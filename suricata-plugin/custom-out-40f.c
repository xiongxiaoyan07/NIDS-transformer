#include "suricata-common.h"
#include "suricata-plugin.h"

#include "output-packet.h"
#include "output-flow.h"
#include "output.h"

#include "decode.h"
#include "decode-ipv4.h"
#include "decode-ipv6.h"
#include "decode-tcp.h"
#include "decode-udp.h"
#include "decode-icmpv4.h"

#include "flow.h"
#include "util-print.h"
#include "util-time.h"
#include "app-layer-protos.h"

#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <stdbool.h>
#include <string.h>
#include <pthread.h>
#include <sys/stat.h>
#include <netinet/in.h>

#define CUSTOM_PACKET_CSV_FILE "/home/xxiong/pcaps/stage1_packets.csv"
#define CUSTOM_FLOW_CSV_FILE   "/home/xxiong/pcaps/stage1_flows.csv"

#define FLOW_AGG_BUCKETS 65536

/*
 * Direction convention:
 *   0 = toserver / source -> destination / IN
 *   1 = toclient / destination -> source / OUT
 *
 * For NF-UQ-like features:
 *   IN_BYTES / IN_PKTS   = source to destination
 *   OUT_BYTES / OUT_PKTS = destination to source
 */

typedef struct FlowAgg_ {
    uint64_t flow_id;

    bool tuple_seen;
    int ip_version;
    char src_ip[46];
    char dst_ip[46];
    uint16_t src_port;
    uint16_t dst_port;
    uint8_t proto;

    uint64_t first_ts_us;
    uint64_t last_ts_us;

    uint64_t first_dir_ts_us[2];
    uint64_t last_dir_ts_us[2];

    uint64_t bytes[2];
    uint64_t pkts[2];

    uint8_t tcp_flags[2];
    uint8_t tcp_flags_total;

    uint16_t tcp_win_max[2];

    uint32_t min_ttl;
    uint32_t max_ttl;

    uint32_t longest_flow_pkt;
    uint32_t shortest_flow_pkt;

    uint32_t min_ip_pkt_len;
    uint32_t max_ip_pkt_len;

    uint64_t num_pkts_up_to_128;
    uint64_t num_pkts_128_to_256;
    uint64_t num_pkts_256_to_512;
    uint64_t num_pkts_512_to_1024;
    uint64_t num_pkts_1024_to_1514;

    bool icmp_seen;
    uint8_t icmp_type;
    uint8_t icmp_code;

    /*
     * Placeholder.
     * Accurate retransmission extraction requires TCP sequence tracking.
     */
    uint64_t retransmitted_bytes[2];
    uint64_t retransmitted_pkts[2];

    /*
     * Suricata-derived weak labels.
     * flow_label = 1 if any packet in this flow triggered at least one alert.
     * alert_packet_count = number of packets in this flow that triggered alerts.
     */
    int flow_label;
    uint64_t alert_packet_count;

    struct FlowAgg_ *next;
} FlowAgg;

typedef struct {
    FILE *packet_fp;
    FILE *flow_fp;
} CustomLoggerThreadData;

static FlowAgg *g_flow_aggs[FLOW_AGG_BUCKETS];

static pthread_mutex_t g_agg_mutex = PTHREAD_MUTEX_INITIALIZER;
static pthread_mutex_t g_file_mutex = PTHREAD_MUTEX_INITIALIZER;

static const char *PACKET_CSV_HEADER =
    "record_type,"
    "packet_id,"
    "timestamp_us,"
    "ts_sec,"
    "ts_usec,"
    "flow_id,"
    "direction,"
    "ip_version,"
    "src_ip,"
    "dst_ip,"
    "src_port,"
    "dst_port,"
    "protocol,"
    "pkt_len,"
    "ip_len,"
    "payload_len,"
    "ttl_or_hop_limit,"
    "tcp_seq,"
    "tcp_ack,"
    "tcp_flags,"
    "tcp_window,"
    "tcp_header_len,"
    "udp_len,"
    "icmp_type,"
    "icmp_code,"
    "l7_proto,"
    "packet_label";

static const char *FLOW_CSV_HEADER =
    "flow_id,"
    "IPV4_SRC_ADDR,"
    "L4_SRC_PORT,"
    "IPV4_DST_ADDR,"
    "L4_DST_PORT,"
    "PROTOCOL,"
    "L7_PROTO,"
    "IN_BYTES,"
    "IN_PKTS,"
    "OUT_BYTES,"
    "OUT_PKTS,"
    "TCP_FLAGS,"
    "CLIENT_TCP_FLAGS,"
    "SERVER_TCP_FLAGS,"
    "FLOW_DURATION_MILLISECONDS,"
    "DURATION_IN,"
    "DURATION_OUT,"
    "MIN_TTL,"
    "MAX_TTL,"
    "LONGEST_FLOW_PKT,"
    "SHORTEST_FLOW_PKT,"
    "MIN_IP_PKT_LEN,"
    "MAX_IP_PKT_LEN,"
    "SRC_TO_DST_SECOND_BYTES,"
    "DST_TO_SRC_SECOND_BYTES,"
    "RETRANSMITTED_IN_BYTES,"
    "RETRANSMITTED_IN_PKTS,"
    "RETRANSMITTED_OUT_BYTES,"
    "RETRANSMITTED_OUT_PKTS,"
    "SRC_TO_DST_AVG_THROUGHPUT,"
    "DST_TO_SRC_AVG_THROUGHPUT,"
    "NUM_PKTS_UP_TO_128_BYTES,"
    "NUM_PKTS_128_TO_256_BYTES,"
    "NUM_PKTS_256_TO_512_BYTES,"
    "NUM_PKTS_512_TO_1024_BYTES,"
    "NUM_PKTS_1024_TO_1514_BYTES,"
    "TCP_WIN_MAX_IN,"
    "TCP_WIN_MAX_OUT,"
    "ICMP_TYPE,"
    "ICMP_IPV4_TYPE,"
    "DNS_QUERY_ID,"
    "DNS_QUERY_TYPE,"
    "DNS_TTL_ANSWER,"
    "FTP_COMMAND_RET_CODE,"
    "Label";
//    "Attack";

static inline uint64_t HashFlowId(uint64_t flow_id)
{
    return flow_id % FLOW_AGG_BUCKETS;
}

static inline uint64_t PacketTsUsec(const Packet *p)
{
    return ((uint64_t)p->ts.secs * 1000000ULL) + (uint64_t)p->ts.usecs;
}

static inline uint64_t DurationUsec(uint64_t first_us, uint64_t last_us)
{
    if (first_us == 0 || last_us == 0 || last_us < first_us)
        return 0;
    return last_us - first_us;
}

static inline double BytesPerSecond(uint64_t bytes, uint64_t duration_us)
{
    if (duration_us == 0)
        return 0.0;
    return ((double)bytes * 1000000.0) / (double)duration_us;
}

static inline double BitsPerSecond(uint64_t bytes, uint64_t duration_us)
{
    return BytesPerSecond(bytes, duration_us) * 8.0;
}

static FILE *OpenCsvFileWithHeader(const char *path, const char *header)
{
    FILE *fp = NULL;
    struct stat st;
    bool need_header = false;

    pthread_mutex_lock(&g_file_mutex);

    if (stat(path, &st) != 0 || st.st_size == 0)
        need_header = true;

    fp = fopen(path, "a");
    if (fp != NULL && need_header) {
        fprintf(fp, "%s\n", header);
        fflush(fp);
    }

    pthread_mutex_unlock(&g_file_mutex);

    return fp;
}

static FlowAgg *FlowAggLookupLocked(uint64_t flow_id)
{
    uint64_t bucket = HashFlowId(flow_id);
    FlowAgg *cur = g_flow_aggs[bucket];

    while (cur != NULL) {
        if (cur->flow_id == flow_id)
            return cur;
        cur = cur->next;
    }

    return NULL;
}

static FlowAgg *FlowAggGetOrCreateLocked(uint64_t flow_id)
{
    FlowAgg *agg = FlowAggLookupLocked(flow_id);
    if (agg != NULL)
        return agg;

    agg = calloc(1, sizeof(*agg));
    if (agg == NULL)
        return NULL;

    agg->flow_id = flow_id;
    agg->min_ttl = UINT32_MAX;
    agg->shortest_flow_pkt = UINT32_MAX;
    agg->min_ip_pkt_len = UINT32_MAX;

    uint64_t bucket = HashFlowId(flow_id);
    agg->next = g_flow_aggs[bucket];
    g_flow_aggs[bucket] = agg;

    return agg;
}

static void FlowAggRemoveLocked(uint64_t flow_id)
{
    uint64_t bucket = HashFlowId(flow_id);
    FlowAgg *cur = g_flow_aggs[bucket];
    FlowAgg *prev = NULL;

    while (cur != NULL) {
        if (cur->flow_id == flow_id) {
            if (prev == NULL)
                g_flow_aggs[bucket] = cur->next;
            else
                prev->next = cur->next;

            free(cur);
            return;
        }

        prev = cur;
        cur = cur->next;
    }
}

static int GetPacketDirection(const Packet *p)
{
#ifdef PKT_IS_TOSERVER
    if (PKT_IS_TOSERVER(p))
        return 0;
#endif

#ifdef PKT_IS_TOCLIENT
    if (PKT_IS_TOCLIENT(p))
        return 1;
#endif

#ifdef FLOW_PKT_TOSERVER
    if (p->flowflags & FLOW_PKT_TOSERVER)
        return 0;
#endif

#ifdef FLOW_PKT_TOCLIENT
    if (p->flowflags & FLOW_PKT_TOCLIENT)
        return 1;
#endif

    return -1;
}

static void GetPacketIPs(const Packet *p, int *ip_version,
                         char *src_ip, size_t src_len,
                         char *dst_ip, size_t dst_len)
{
    *ip_version = 0;
    src_ip[0] = '\0';
    dst_ip[0] = '\0';

    if (PacketIsIPv4(p)) {
        *ip_version = 4;
        PrintInet(AF_INET, &p->src.addr_data32[0], src_ip, src_len);
        PrintInet(AF_INET, &p->dst.addr_data32[0], dst_ip, dst_len);
    } else if (PacketIsIPv6(p)) {
        *ip_version = 6;
        PrintInet(AF_INET6, &p->src.addr_data32[0], src_ip, src_len);
        PrintInet(AF_INET6, &p->dst.addr_data32[0], dst_ip, dst_len);
    }
}

static void GetPacketPorts(const Packet *p, uint16_t *src_port, uint16_t *dst_port)
{
    *src_port = 0;
    *dst_port = 0;

    if (p->proto == IPPROTO_TCP && p->l4.hdrs.tcph != NULL) {
        const TCPHdr *tcp = p->l4.hdrs.tcph;
        *src_port = SCNtohs(tcp->th_sport);
        *dst_port = SCNtohs(tcp->th_dport);
    } else if (p->proto == IPPROTO_UDP && p->l4.hdrs.udph != NULL) {
        const UDPHdr *udp = p->l4.hdrs.udph;
        *src_port = SCNtohs(udp->uh_sport);
        *dst_port = SCNtohs(udp->uh_dport);
    }
}

static uint32_t GetIpPacketLen(const Packet *p)
{
    if (PacketIsIPv4(p)) {
        const IPV4Hdr *ipv4 = PacketGetIPv4(p);
        if (ipv4 != NULL)
            return (uint32_t)SCNtohs(ipv4->ip_len);
    }

    if (PacketIsIPv6(p)) {
        const IPV6Hdr *ipv6 = PacketGetIPv6(p);
        if (ipv6 != NULL) {
            uint32_t payload_len =
                (uint32_t)SCNtohs(ipv6->ip6_hdrun.ip6_un1.ip6_un1_plen);
            return 40U + payload_len;
        }
    }

    return (uint32_t)GET_PKT_LEN(p);
}

static int GetIpTTLOrHopLimit(const Packet *p)
{
    if (PacketIsIPv4(p)) {
        const IPV4Hdr *ipv4 = PacketGetIPv4(p);
        if (ipv4 != NULL)
            return (int)ipv4->ip_ttl;
    }

    if (PacketIsIPv6(p)) {
        const IPV6Hdr *ipv6 = PacketGetIPv6(p);
        if (ipv6 != NULL)
            return (int)ipv6->ip6_hdrun.ip6_un1.ip6_un1_hlim;
    }

    return -1;
}

static void UpdatePacketLengthBins(FlowAgg *agg, uint32_t ip_len)
{
    if (ip_len <= 128) {
        agg->num_pkts_up_to_128++;
    } else if (ip_len <= 256) {
        agg->num_pkts_128_to_256++;
    } else if (ip_len <= 512) {
        agg->num_pkts_256_to_512++;
    } else if (ip_len <= 1024) {
        agg->num_pkts_512_to_1024++;
    } else if (ip_len <= 1514) {
        agg->num_pkts_1024_to_1514++;
    }
}

static void UpdateFlowAggFromPacket(const Packet *p)
{
    if (p == NULL || p->flow == NULL)
        return;

    uint64_t flow_id = FlowGetId(p->flow);
    uint64_t ts_us = PacketTsUsec(p);
    int direction = GetPacketDirection(p);

    if (direction != 0 && direction != 1)
        direction = 0;

    uint32_t ip_len = GetIpPacketLen(p);
    int ttl = GetIpTTLOrHopLimit(p);

  /*
     * Packet-level weak label:
     * 1 means this packet triggered at least one Suricata alert.
     * 0 means this packet triggered no Suricata alert.
     */
    int packet_label = p->alerts.cnt > 0 ? 1 : 0;

    pthread_mutex_lock(&g_agg_mutex);

    FlowAgg *agg = FlowAggGetOrCreateLocked(flow_id);
    if (agg == NULL) {
        pthread_mutex_unlock(&g_agg_mutex);
        return;
    }

    /*
     * Flow-level weak label:
     * If any packet in the flow has alert, mark the whole flow as malicious.
     */
    if (packet_label == 1) {
        agg->flow_label = 1;
        agg->alert_packet_count++;
    }

    if (!agg->tuple_seen) {
        int ip_version = 0;
        char src_ip[46] = {0};
        char dst_ip[46] = {0};
        uint16_t src_port = 0;
        uint16_t dst_port = 0;

        GetPacketIPs(p, &ip_version, src_ip, sizeof(src_ip), dst_ip, sizeof(dst_ip));
        GetPacketPorts(p, &src_port, &dst_port);

        agg->tuple_seen = true;
        agg->ip_version = ip_version;
        snprintf(agg->src_ip, sizeof(agg->src_ip), "%s", src_ip);
        snprintf(agg->dst_ip, sizeof(agg->dst_ip), "%s", dst_ip);
        agg->src_port = src_port;
        agg->dst_port = dst_port;
        agg->proto = p->proto;
    }

    if (agg->first_ts_us == 0 || ts_us < agg->first_ts_us)
        agg->first_ts_us = ts_us;

    if (ts_us > agg->last_ts_us)
        agg->last_ts_us = ts_us;

    if (agg->first_dir_ts_us[direction] == 0 || ts_us < agg->first_dir_ts_us[direction])
        agg->first_dir_ts_us[direction] = ts_us;

    if (ts_us > agg->last_dir_ts_us[direction])
        agg->last_dir_ts_us[direction] = ts_us;

    agg->pkts[direction]++;
    agg->bytes[direction] += ip_len;

    if (ttl >= 0) {
        if ((uint32_t)ttl < agg->min_ttl)
            agg->min_ttl = (uint32_t)ttl;
        if ((uint32_t)ttl > agg->max_ttl)
            agg->max_ttl = (uint32_t)ttl;
    }

    if (ip_len > agg->longest_flow_pkt)
        agg->longest_flow_pkt = ip_len;

    if (ip_len < agg->shortest_flow_pkt)
        agg->shortest_flow_pkt = ip_len;

    if (ip_len < agg->min_ip_pkt_len)
        agg->min_ip_pkt_len = ip_len;

    if (ip_len > agg->max_ip_pkt_len)
        agg->max_ip_pkt_len = ip_len;

    UpdatePacketLengthBins(agg, ip_len);

    if (p->proto == IPPROTO_TCP && p->l4.hdrs.tcph != NULL) {
        const TCPHdr *tcp = p->l4.hdrs.tcph;
        uint8_t flags = tcp->th_flags;
        uint16_t win = SCNtohs(tcp->th_win);

        agg->tcp_flags[direction] |= flags;
        agg->tcp_flags_total |= flags;

        if (win > agg->tcp_win_max[direction])
            agg->tcp_win_max[direction] = win;
    }

    if (p->proto == IPPROTO_ICMP && p->l4.hdrs.icmpv4h != NULL) {
        const ICMPV4Hdr *icmp4 = p->l4.hdrs.icmpv4h;
        agg->icmp_seen = true;
        agg->icmp_type = icmp4->type;
        agg->icmp_code = icmp4->code;
    }

    pthread_mutex_unlock(&g_agg_mutex);
}

static void WritePacketCsvLine(FILE *fp, const Packet *p)
{
    if (fp == NULL || p == NULL)
        return;

    uint64_t ts_us = PacketTsUsec(p);

    uint64_t flow_id = 0;
    if (p->flow != NULL)
        flow_id = FlowGetId(p->flow);

    int direction = GetPacketDirection(p);

    int ip_version = 0;
    char src_ip[46] = {0};
    char dst_ip[46] = {0};
    GetPacketIPs(p, &ip_version, src_ip, sizeof(src_ip), dst_ip, sizeof(dst_ip));

    uint16_t src_port = 0;
    uint16_t dst_port = 0;
    GetPacketPorts(p, &src_port, &dst_port);

    uint32_t pkt_len = (uint32_t)GET_PKT_LEN(p);
    uint32_t ip_len = GetIpPacketLen(p);
    int ttl = GetIpTTLOrHopLimit(p);

    uint32_t tcp_seq = 0;
    uint32_t tcp_ack = 0;
    int tcp_flags = 0;
    int tcp_window = -1;
    int tcp_header_len = -1;

    int udp_len = -1;

    int icmp_type = -1;
    int icmp_code = -1;

    int l7_proto = 0;

    if (p->flow != NULL)
        l7_proto = p->flow->alproto;

    /*
     * Packet-level weak label.
     * Do not use this as an input feature for the Transformer.
     * It should only be used as target/analysis metadata.
     */
    int packet_label = p->alerts.cnt > 0 ? 1 : 0;
//    int packet_alert_count = p->alerts.cnt;

    if (p->proto == IPPROTO_TCP && p->l4.hdrs.tcph != NULL) {
        const TCPHdr *tcp = p->l4.hdrs.tcph;
        tcp_seq = SCNtohl(tcp->th_seq);
        tcp_ack = SCNtohl(tcp->th_ack);
        tcp_flags = tcp->th_flags;
        tcp_window = SCNtohs(tcp->th_win);
        tcp_header_len = TCP_GET_HLEN(p);
    }

    if (p->proto == IPPROTO_UDP && p->l4.hdrs.udph != NULL) {
        const UDPHdr *udp = p->l4.hdrs.udph;
        udp_len = SCNtohs(udp->uh_len);
    }

    if (p->proto == IPPROTO_ICMP && p->l4.hdrs.icmpv4h != NULL) {
        const ICMPV4Hdr *icmp4 = p->l4.hdrs.icmpv4h;
        icmp_type = icmp4->type;
        icmp_code = icmp4->code;
    }

    pthread_mutex_lock(&g_file_mutex);

    fprintf(fp,
        "packet,"
        "%lu,"
        "%lu,"
        "%lu,"
        "%lu,"
        "%lu,"
        "%d,"
        "%d,"
        "%s,"
        "%s,"
        "%u,"
        "%u,"
        "%u,"
        "%u,"
        "%u,"
        "%u,"
        "%d,"
        "%u,"
        "%u,"
        "%d,"
        "%d,"
        "%d,"
        "%d,"
        "%d,"
        "%d,"
        "%d,"
        "%d\n",
        (unsigned long)p->pcap_cnt,
        (unsigned long)ts_us,
        (unsigned long)p->ts.secs,
        (unsigned long)p->ts.usecs,
        (unsigned long)flow_id,
        direction,
        ip_version,
        src_ip,
        dst_ip,
        src_port,
        dst_port,
        p->proto,
        pkt_len,
        ip_len,
        (uint32_t)p->payload_len,
        ttl,
        tcp_seq,
        tcp_ack,
        tcp_flags,
        tcp_window,
        tcp_header_len,
        udp_len,
        icmp_type,
        icmp_code,
        l7_proto,
        packet_label);

    fflush(fp);

    pthread_mutex_unlock(&g_file_mutex);
}

static void GetFlowTuple(Flow *f,
                         char *src_ip, size_t src_len,
                         char *dst_ip, size_t dst_len,
                         uint16_t *src_port,
                         uint16_t *dst_port)
{
    src_ip[0] = '\0';
    dst_ip[0] = '\0';
    *src_port = 0;
    *dst_port = 0;

    if ((f->flags & FLOW_DIR_REVERSED) == 0) {
        if (FLOW_IS_IPV4(f)) {
            PrintInet(AF_INET, (const void *)&(f->src.addr_data32[0]), src_ip, src_len);
            PrintInet(AF_INET, (const void *)&(f->dst.addr_data32[0]), dst_ip, dst_len);
        } else if (FLOW_IS_IPV6(f)) {
            PrintInet(AF_INET6, (const void *)&(f->src.address), src_ip, src_len);
            PrintInet(AF_INET6, (const void *)&(f->dst.address), dst_ip, dst_len);
        }

        *src_port = f->sp;
        *dst_port = f->dp;
    } else {
        if (FLOW_IS_IPV4(f)) {
            PrintInet(AF_INET, (const void *)&(f->dst.addr_data32[0]), src_ip, src_len);
            PrintInet(AF_INET, (const void *)&(f->src.addr_data32[0]), dst_ip, dst_len);
        } else if (FLOW_IS_IPV6(f)) {
            PrintInet(AF_INET6, (const void *)&(f->dst.address), src_ip, src_len);
            PrintInet(AF_INET6, (const void *)&(f->src.address), dst_ip, dst_len);
        }

        *src_port = f->dp;
        *dst_port = f->sp;
    }
}

static void WriteFlowCsvLine(FILE *fp, Flow *f, const FlowAgg *agg)
{
    if (fp == NULL || f == NULL)
        return;

    uint64_t flow_id = FlowGetId(f);

    char src_ip[46] = {0};
    char dst_ip[46] = {0};
    uint16_t src_port = 0;
    uint16_t dst_port = 0;

    GetFlowTuple(f, src_ip, sizeof(src_ip), dst_ip, sizeof(dst_ip),
                 &src_port, &dst_port);

    uint64_t in_bytes = 0;
    uint64_t out_bytes = 0;
    uint64_t in_pkts = 0;
    uint64_t out_pkts = 0;

    uint8_t tcp_flags_total = 0;
    uint8_t client_tcp_flags = 0;
    uint8_t server_tcp_flags = 0;

    uint64_t flow_duration_us = 0;
    uint64_t duration_in_us = 0;
    uint64_t duration_out_us = 0;

    uint32_t min_ttl = 0;
    uint32_t max_ttl = 0;

    uint32_t longest_flow_pkt = 0;
    uint32_t shortest_flow_pkt = 0;
    uint32_t min_ip_pkt_len = 0;
    uint32_t max_ip_pkt_len = 0;

    uint64_t retransmitted_in_bytes = 0;
    uint64_t retransmitted_in_pkts = 0;
    uint64_t retransmitted_out_bytes = 0;
    uint64_t retransmitted_out_pkts = 0;

    uint64_t num_pkts_up_to_128 = 0;
    uint64_t num_pkts_128_to_256 = 0;
    uint64_t num_pkts_256_to_512 = 0;
    uint64_t num_pkts_512_to_1024 = 0;
    uint64_t num_pkts_1024_to_1514 = 0;

    uint16_t tcp_win_max_in = 0;
    uint16_t tcp_win_max_out = 0;

    int icmp_type = 0;
    int icmp_ipv4_type = 0;

    if (agg != NULL) {
        in_bytes = agg->bytes[0];
        out_bytes = agg->bytes[1];
        in_pkts = agg->pkts[0];
        out_pkts = agg->pkts[1];

        tcp_flags_total = agg->tcp_flags_total;
        client_tcp_flags = agg->tcp_flags[0];
        server_tcp_flags = agg->tcp_flags[1];

        flow_duration_us = DurationUsec(agg->first_ts_us, agg->last_ts_us);
        duration_in_us = DurationUsec(agg->first_dir_ts_us[0], agg->last_dir_ts_us[0]);
        duration_out_us = DurationUsec(agg->first_dir_ts_us[1], agg->last_dir_ts_us[1]);

        if (agg->min_ttl != UINT32_MAX)
            min_ttl = agg->min_ttl;
        max_ttl = agg->max_ttl;

        longest_flow_pkt = agg->longest_flow_pkt;

        if (agg->shortest_flow_pkt != UINT32_MAX)
            shortest_flow_pkt = agg->shortest_flow_pkt;

        if (agg->min_ip_pkt_len != UINT32_MAX)
            min_ip_pkt_len = agg->min_ip_pkt_len;

        max_ip_pkt_len = agg->max_ip_pkt_len;

        retransmitted_in_bytes = agg->retransmitted_bytes[0];
        retransmitted_in_pkts = agg->retransmitted_pkts[0];
        retransmitted_out_bytes = agg->retransmitted_bytes[1];
        retransmitted_out_pkts = agg->retransmitted_pkts[1];

        num_pkts_up_to_128 = agg->num_pkts_up_to_128;
        num_pkts_128_to_256 = agg->num_pkts_128_to_256;
        num_pkts_256_to_512 = agg->num_pkts_256_to_512;
        num_pkts_512_to_1024 = agg->num_pkts_512_to_1024;
        num_pkts_1024_to_1514 = agg->num_pkts_1024_to_1514;

        tcp_win_max_in = agg->tcp_win_max[0];
        tcp_win_max_out = agg->tcp_win_max[1];

        if (agg->icmp_seen) {
            icmp_type = ((int)agg->icmp_type * 256) + (int)agg->icmp_code;
            icmp_ipv4_type = (int)agg->icmp_type;
        }
    } else {
        /*
         * Fallback when aggregation is missing.
         * Packet logger should normally create agg before Flow Logger runs.
         */
        in_pkts = f->todstpktcnt;
        out_pkts = f->tosrcpktcnt;
    }

    double src_to_dst_second_bytes = BytesPerSecond(in_bytes, duration_in_us);
    double dst_to_src_second_bytes = BytesPerSecond(out_bytes, duration_out_us);

    double src_to_dst_avg_throughput = BitsPerSecond(in_bytes, duration_in_us);
    double dst_to_src_avg_throughput = BitsPerSecond(out_bytes, duration_out_us);

    /*
     * DNS/FTP are better filled by merging EVE dns/ftp logs by flow_id.
     * Label/Attack should also be filled later from EVE alert or ground truth.
     */
    int dns_query_id = 0;
    int dns_query_type = 0;
    int dns_ttl_answer = 0;
    int ftp_command_ret_code = 0;

    int label = 0;
//    const char *attack = "Unlabeled";

    if (agg != NULL && agg->flow_label == 1) {
        label = 1;
    }

    pthread_mutex_lock(&g_file_mutex);

    fprintf(fp,
            "%lu,"
            "%s,"
            "%u,"
            "%s,"
            "%u,"
            "%u,"
            "%d,"
            "%lu,"
            "%lu,"
            "%lu,"
            "%lu,"
            "%u,"
            "%u,"
            "%u,"
            "%lu,"
            "%lu,"
            "%lu,"
            "%u,"
            "%u,"
            "%u,"
            "%u,"
            "%u,"
            "%u,"
            "%.6f,"
            "%.6f,"
            "%lu,"
            "%lu,"
            "%lu,"
            "%lu,"
            "%.6f,"
            "%.6f,"
            "%lu,"
            "%lu,"
            "%lu,"
            "%lu,"
            "%lu,"
            "%u,"
            "%u,"
            "%d,"
            "%d,"
            "%d,"
            "%d,"
            "%d,"
            "%d,"
            "%d\n",
            (unsigned long)flow_id,
            src_ip,
            src_port,
            dst_ip,
            dst_port,
            f->proto,
            f->alproto,
            (unsigned long)in_bytes,
            (unsigned long)in_pkts,
            (unsigned long)out_bytes,
            (unsigned long)out_pkts,
            tcp_flags_total,
            client_tcp_flags,
            server_tcp_flags,
            (unsigned long)(flow_duration_us / 1000ULL),
            (unsigned long)(duration_in_us / 1000ULL),
            (unsigned long)(duration_out_us / 1000ULL),
            min_ttl,
            max_ttl,
            longest_flow_pkt,
            shortest_flow_pkt,
            min_ip_pkt_len,
            max_ip_pkt_len,
            src_to_dst_second_bytes,
            dst_to_src_second_bytes,
            (unsigned long)retransmitted_in_bytes,
            (unsigned long)retransmitted_in_pkts,
            (unsigned long)retransmitted_out_bytes,
            (unsigned long)retransmitted_out_pkts,
            src_to_dst_avg_throughput,
            dst_to_src_avg_throughput,
            (unsigned long)num_pkts_up_to_128,
            (unsigned long)num_pkts_128_to_256,
            (unsigned long)num_pkts_256_to_512,
            (unsigned long)num_pkts_512_to_1024,
            (unsigned long)num_pkts_1024_to_1514,
            tcp_win_max_in,
            tcp_win_max_out,
            icmp_type,
            icmp_ipv4_type,
            dns_query_id,
            dns_query_type,
            dns_ttl_answer,
            ftp_command_ret_code,
            label);

    fflush(fp);

    pthread_mutex_unlock(&g_file_mutex);
}

static int CustomPacketCsvLogger(ThreadVars *tv, void *thread_data, const Packet *p)
{
    (void)tv;

    CustomLoggerThreadData *tdata = (CustomLoggerThreadData *)thread_data;

    if (tdata == NULL || p == NULL)
        return 0;

    UpdateFlowAggFromPacket(p);
    WritePacketCsvLine(tdata->packet_fp, p);

    return 0;
}

static int CustomFlowCsvLogger(ThreadVars *tv, void *thread_data, Flow *f)
{
    (void)tv;

    CustomLoggerThreadData *tdata = (CustomLoggerThreadData *)thread_data;

    if (tdata == NULL || f == NULL)
        return 0;

    uint64_t flow_id = FlowGetId(f);

    FlowAgg agg_copy;
    FlowAgg *agg_ptr = NULL;

    pthread_mutex_lock(&g_agg_mutex);

    FlowAgg *agg = FlowAggLookupLocked(flow_id);
    if (agg != NULL) {
        agg_copy = *agg;
        agg_copy.next = NULL;
        agg_ptr = &agg_copy;
        FlowAggRemoveLocked(flow_id);
    }

    pthread_mutex_unlock(&g_agg_mutex);

    WriteFlowCsvLine(tdata->flow_fp, f, agg_ptr);

    return 0;
}

static bool CustomPacketLoggerCondition(ThreadVars *tv, void *thread_data, const Packet *p)
{
    (void)tv;
    (void)thread_data;
    (void)p;

    return true;
}

static TmEcode ThreadInit(ThreadVars *tv, const void *initdata, void **data)
{
    (void)tv;
    (void)initdata;

    CustomLoggerThreadData *tdata = calloc(1, sizeof(*tdata));
    if (tdata == NULL) {
        SCLogError("Could not allocate custom CSV logger thread data");
        return TM_ECODE_FAILED;
    }

    tdata->packet_fp = OpenCsvFileWithHeader(CUSTOM_PACKET_CSV_FILE, PACKET_CSV_HEADER);
    if (tdata->packet_fp == NULL) {
        SCLogError("Could not open packet CSV file: %s", CUSTOM_PACKET_CSV_FILE);
        free(tdata);
        return TM_ECODE_FAILED;
    }

    tdata->flow_fp = OpenCsvFileWithHeader(CUSTOM_FLOW_CSV_FILE, FLOW_CSV_HEADER);
    if (tdata->flow_fp == NULL) {
        SCLogError("Could not open flow CSV file: %s", CUSTOM_FLOW_CSV_FILE);
        fclose(tdata->packet_fp);
        free(tdata);
        return TM_ECODE_FAILED;
    }

    *data = tdata;
    return TM_ECODE_OK;
}

static TmEcode ThreadDeinit(ThreadVars *tv, void *data)
{
    (void)tv;

    CustomLoggerThreadData *tdata = (CustomLoggerThreadData *)data;

    if (tdata != NULL) {
        if (tdata->packet_fp != NULL)
            fclose(tdata->packet_fp);

        if (tdata->flow_fp != NULL)
            fclose(tdata->flow_fp);

        free(tdata);
    }

    return TM_ECODE_OK;
}

static void OnLoggingReady(void *arg)
{
    (void)arg;

    SCOutputRegisterPacketLogger(LOGGER_USER,
                                 "custom-packet-logger",
                                 CustomPacketCsvLogger,
                                 CustomPacketLoggerCondition,
                                 NULL,
                                 ThreadInit,
                                 ThreadDeinit);

    SCOutputRegisterFlowLogger("custom-flow-logger",
                               CustomFlowCsvLogger,
                               NULL,
                               ThreadInit,
                               ThreadDeinit);
}

static void Init(void)
{
    SCRegisterOnLoggingReady(OnLoggingReady, NULL);
}

const SCPlugin PluginRegistration = {
    .version = SC_API_VERSION,
    .suricata_version = SC_PACKAGE_VERSION,
    .name = "custom-stage1-csv-logger",
    .plugin_version = "2.1.0",
    .author = "Xiaoyan Xiong",
    .license = "GPLv2",
    .Init = Init,
};

const SCPlugin *SCPluginRegister(void)
{
    return &PluginRegistration;
}