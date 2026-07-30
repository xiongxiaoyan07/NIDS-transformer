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
#include <math.h>
#include <float.h>

#define CUSTOM_PACKET_CSV_FILE "/home/xxiong/pcaps/stage1_packets.csv"
#define CUSTOM_FLOW_CSV_FILE   "/home/xxiong/pcaps/stage1_flows.csv"
#define CUSTOM_NESTED_JSONL_FILE "/home/xxiong/pcaps/stage1_nested_flows.jsonl"

#define FLOW_AGG_BUCKETS 65536

/*
 * Maximum packet records saved inside each nested flow JSON object.
 * 0 means save all packets, but this can create very large memory usage and very large JSONL files.
 * 256 or 512 is usually enough for Transformer sequence experiments.
 */
#define MAX_PACKETS_PER_FLOW 512

/*
 * CICFlowMeter-style active/idle split threshold.
 * If the gap between two packets is larger than this value, the gap is counted as idle time.
 */
#define CIC_ACTIVE_TIMEOUT_US 5000000ULL

#ifndef TH_ECE
#define TH_ECE 0x40
#endif

#ifndef TH_CWR
#define TH_CWR 0x80
#endif

typedef struct {
    uint64_t n;
    double sum;
    double sumsq;
    double min;
    double max;
} StatAgg;

static inline void StatAdd(StatAgg *s, double x)
{
    if (s->n == 0) {
        s->min = x;
        s->max = x;
    } else {
        if (x < s->min)
            s->min = x;
        if (x > s->max)
            s->max = x;
    }

    s->n++;
    s->sum += x;
    s->sumsq += x * x;
}

static inline double StatMean(const StatAgg *s)
{
    if (s == NULL || s->n == 0)
        return 0.0;
    return s->sum / (double)s->n;
}

static inline double StatVar(const StatAgg *s)
{
    if (s == NULL || s->n <= 1)
        return 0.0;

    double n = (double)s->n;
    double var = (s->sumsq - ((s->sum * s->sum) / n)) / (n - 1.0);

    if (var < 0.0)
        var = 0.0;

    return var;
}

static inline double StatStd(const StatAgg *s)
{
    return sqrt(StatVar(s));
}

static inline double StatMin(const StatAgg *s)
{
    if (s == NULL || s->n == 0)
        return 0.0;
    return s->min;
}

static inline double StatMax(const StatAgg *s)
{
    if (s == NULL || s->n == 0)
        return 0.0;
    return s->max;
}

typedef struct PacketFeature_ {
    char record_type[16];

    uint64_t packet_id;
    uint64_t timestamp_us;
    uint64_t ts_sec;
    uint64_t ts_usec;

    uint64_t flow_id;
    int direction;
    int ip_version;

    char src_ip[46];
    char dst_ip[46];

    uint16_t src_port;
    uint16_t dst_port;
    uint8_t protocol;

    uint32_t pkt_len;
    uint32_t ip_len;
    uint32_t payload_len;

    int ttl_or_hop_limit;

    uint32_t tcp_seq;
    uint32_t tcp_ack;
    int tcp_flags;
    int tcp_window;
    int tcp_header_len;

    int udp_len;

    int icmp_type;
    int icmp_code;

    int l7_proto;

    uint64_t flow_iat_us;
    uint64_t direction_iat_us;

    int packet_label;

    struct PacketFeature_ *next;
} PacketFeature;

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

    uint64_t retransmitted_bytes[2];
    uint64_t retransmitted_pkts[2];

    int flow_label;
    uint64_t alert_packet_count;

    StatAgg pkt_len_stats_all;
    StatAgg pkt_len_stats[2];

    StatAgg flow_iat_stats;
    StatAgg dir_iat_stats[2];

    uint64_t last_pkt_ts_us;
    uint64_t last_iat_ts_us[2];

    uint64_t header_len[2];

    uint64_t fin_count;
    uint64_t syn_count;
    uint64_t rst_count;
    uint64_t psh_count;
    uint64_t ack_count;
    uint64_t urg_count;
    uint64_t cwe_count;
    uint64_t ece_count;

    uint64_t fwd_psh_flags;
    uint64_t bwd_psh_flags;
    uint64_t fwd_urg_flags;
    uint64_t bwd_urg_flags;

    int init_win_bytes_forward;
    int init_win_bytes_backward;
    bool init_win_fwd_seen;
    bool init_win_bwd_seen;

    uint64_t act_data_pkt_fwd;
    uint32_t min_seg_size_forward;

    uint64_t active_start_us;
    uint64_t active_end_us;
    StatAgg active_stats;
    StatAgg idle_stats;

    PacketFeature *pkt_head;
    PacketFeature *pkt_tail;
    uint64_t packet_total_count;
    uint64_t packet_saved_count;
    bool packet_truncated;

    struct FlowAgg_ *next;
} FlowAgg;

typedef struct {
    FILE *packet_fp;
    FILE *flow_fp;
    FILE *nested_fp;
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
    "flow_iat_us,"
    "direction_iat_us,"
    "packet_label";

static const char *FLOW_CSV_HEADER =
    "flow_id,"
    "source_ip,"
    "source_port,"
    "destination_ip,"
    "destination_port,"
    "protocol,"
    "flow_duration,"
    "total_fwd_packets,"
    "total_backward_packets,"
    "total_length_of_fwd_packets,"
    "total_length_of_bwd_packets,"
    "fwd_packet_length_max,"
    "fwd_packet_length_min,"
    "fwd_packet_length_mean,"
    "fwd_packet_length_std,"
    "bwd_packet_length_max,"
    "bwd_packet_length_min,"
    "bwd_packet_length_mean,"
    "bwd_packet_length_std,"
    "flow_bytes_per_s,"
    "flow_packets_per_s,"
    "flow_iat_mean,"
    "flow_iat_std,"
    "flow_iat_max,"
    "flow_iat_min,"
    "fwd_iat_total,"
    "fwd_iat_mean,"
    "fwd_iat_std,"
    "fwd_iat_max,"
    "fwd_iat_min,"
    "bwd_iat_total,"
    "bwd_iat_mean,"
    "bwd_iat_std,"
    "bwd_iat_max,"
    "bwd_iat_min,"
    "fwd_psh_flags,"
    "bwd_psh_flags,"
    "fwd_urg_flags,"
    "bwd_urg_flags,"
    "fwd_header_length,"
    "bwd_header_length,"
    "fwd_packets_per_s,"
    "bwd_packets_per_s,"
    "min_packet_length,"
    "max_packet_length,"
    "packet_length_mean,"
    "packet_length_std,"
    "packet_length_variance,"
    "fin_flag_count,"
    "syn_flag_count,"
    "rst_flag_count,"
    "psh_flag_count,"
    "ack_flag_count,"
    "urg_flag_count,"
    "cwe_flag_count,"
    "ece_flag_count,"
    "down_up_ratio,"
    "average_packet_size,"
    "avg_fwd_segment_size,"
    "avg_bwd_segment_size,"
    "init_win_bytes_forward,"
    "init_win_bytes_backward,"
    "act_data_pkt_fwd,"
    "min_seg_size_forward,"
    "active_mean,"
    "active_std,"
    "active_max,"
    "active_min,"
    "idle_mean,"
    "idle_std,"
    "idle_max,"
    "idle_min,"
    "label";

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

static void JsonWriteEscapedString(FILE *fp, const char *s)
{
    fputc('"', fp);

    if (s != NULL) {
        const unsigned char *p = (const unsigned char *)s;

        while (*p != '\0') {
            unsigned char c = *p;

            switch (c) {
                case '"':
                    fputs("\\\"", fp);
                    break;
                case '\\':
                    fputs("\\\\", fp);
                    break;
                case '\b':
                    fputs("\\b", fp);
                    break;
                case '\f':
                    fputs("\\f", fp);
                    break;
                case '\n':
                    fputs("\\n", fp);
                    break;
                case '\r':
                    fputs("\\r", fp);
                    break;
                case '\t':
                    fputs("\\t", fp);
                    break;
                default:
                    if (c < 0x20)
                        fprintf(fp, "\\u%04x", c);
                    else
                        fputc(c, fp);
                    break;
            }

            p++;
        }
    }

    fputc('"', fp);
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
    agg->init_win_bytes_forward = -1;
    agg->init_win_bytes_backward = -1;
    agg->min_seg_size_forward = UINT32_MAX;

    uint64_t bucket = HashFlowId(flow_id);
    agg->next = g_flow_aggs[bucket];
    g_flow_aggs[bucket] = agg;

    return agg;
}

static FlowAgg *FlowAggDetachLocked(uint64_t flow_id)
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

            cur->next = NULL;
            return cur;
        }

        prev = cur;
        cur = cur->next;
    }

    return NULL;
}

static void FreePacketFeatureList(PacketFeature *pkt)
{
    while (pkt != NULL) {
        PacketFeature *next = pkt->next;
        free(pkt);
        pkt = next;
    }
}

static void FreeFlowAgg(FlowAgg *agg)
{
    if (agg == NULL)
        return;

    FreePacketFeatureList(agg->pkt_head);
    free(agg);
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

static uint32_t GetCicPacketLen(const Packet *p)
{
    return (uint32_t)p->payload_len;
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

static uint32_t GetIpHeaderLen(const Packet *p)
{
    if (PacketIsIPv4(p)) {
        const IPV4Hdr *ipv4 = PacketGetIPv4(p);
        if (ipv4 != NULL)
            return (uint32_t)IPV4_GET_RAW_HLEN(ipv4);
    }

    if (PacketIsIPv6(p))
        return 40U;

    return 0;
}

static uint32_t GetL4HeaderLen(const Packet *p)
{
    if (p->proto == IPPROTO_TCP && p->l4.hdrs.tcph != NULL)
        return (uint32_t)TCP_GET_HLEN(p);

    if (p->proto == IPPROTO_UDP && p->l4.hdrs.udph != NULL)
        return 8U;

    if (p->proto == IPPROTO_ICMP && p->l4.hdrs.icmpv4h != NULL)
        return 8U;

    return 0;
}

static uint32_t GetHeaderLenForCIC(const Packet *p)
{
    return GetIpHeaderLen(p) + GetL4HeaderLen(p);
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

static void UpdateActiveIdleStats(FlowAgg *agg, uint64_t ts_us, uint64_t flow_iat_us)
{
    if (agg->active_start_us == 0) {
        agg->active_start_us = ts_us;
        agg->active_end_us = ts_us;
        return;
    }

    if (flow_iat_us > CIC_ACTIVE_TIMEOUT_US) {
        if (agg->active_end_us >= agg->active_start_us) {
            StatAdd(&agg->active_stats,
                    (double)(agg->active_end_us - agg->active_start_us));
        }

        StatAdd(&agg->idle_stats, (double)flow_iat_us);

        agg->active_start_us = ts_us;
        agg->active_end_us = ts_us;
    } else {
        agg->active_end_us = ts_us;
    }
}

static void AppendPacketFeatureLocked(FlowAgg *agg,
                                      const Packet *p,
                                      int direction,
                                      uint64_t flow_iat_us,
                                      uint64_t direction_iat_us,
                                      int packet_label)
{
    if (agg == NULL || p == NULL)
        return;

    agg->packet_total_count++;

    if (MAX_PACKETS_PER_FLOW > 0 &&
        agg->packet_saved_count >= MAX_PACKETS_PER_FLOW) {
        agg->packet_truncated = true;
        return;
    }

    PacketFeature *pf = calloc(1, sizeof(*pf));
    if (pf == NULL) {
        agg->packet_truncated = true;
        return;
    }

    snprintf(pf->record_type, sizeof(pf->record_type), "%s", "packet");

    pf->packet_id = (uint64_t)p->pcap_cnt;
    pf->timestamp_us = PacketTsUsec(p);
    pf->ts_sec = (uint64_t)p->ts.secs;
    pf->ts_usec = (uint64_t)p->ts.usecs;

    pf->flow_id = agg->flow_id;
    pf->direction = direction;

    GetPacketIPs(p, &pf->ip_version,
                 pf->src_ip, sizeof(pf->src_ip),
                 pf->dst_ip, sizeof(pf->dst_ip));

    GetPacketPorts(p, &pf->src_port, &pf->dst_port);

    pf->protocol = p->proto;
    pf->pkt_len = (uint32_t)GET_PKT_LEN(p);
    pf->ip_len = GetIpPacketLen(p);
    pf->payload_len = (uint32_t)p->payload_len;
    pf->ttl_or_hop_limit = GetIpTTLOrHopLimit(p);

    pf->tcp_seq = 0;
    pf->tcp_ack = 0;
    pf->tcp_flags = 0;
    pf->tcp_window = -1;
    pf->tcp_header_len = -1;

    pf->udp_len = -1;

    pf->icmp_type = -1;
    pf->icmp_code = -1;

    pf->l7_proto = 0;

    if (p->flow != NULL)
        pf->l7_proto = p->flow->alproto;

    if (p->proto == IPPROTO_TCP && p->l4.hdrs.tcph != NULL) {
        const TCPHdr *tcp = p->l4.hdrs.tcph;
        pf->tcp_seq = SCNtohl(tcp->th_seq);
        pf->tcp_ack = SCNtohl(tcp->th_ack);
        pf->tcp_flags = tcp->th_flags;
        pf->tcp_window = SCNtohs(tcp->th_win);
        pf->tcp_header_len = TCP_GET_HLEN(p);
    }

    if (p->proto == IPPROTO_UDP && p->l4.hdrs.udph != NULL) {
        const UDPHdr *udp = p->l4.hdrs.udph;
        pf->udp_len = SCNtohs(udp->uh_len);
    }

    if (p->proto == IPPROTO_ICMP && p->l4.hdrs.icmpv4h != NULL) {
        const ICMPV4Hdr *icmp4 = p->l4.hdrs.icmpv4h;
        pf->icmp_type = icmp4->type;
        pf->icmp_code = icmp4->code;
    }

    pf->flow_iat_us = flow_iat_us;
    pf->direction_iat_us = direction_iat_us;
    pf->packet_label = packet_label;

    if (agg->pkt_tail == NULL) {
        agg->pkt_head = pf;
        agg->pkt_tail = pf;
    } else {
        agg->pkt_tail->next = pf;
        agg->pkt_tail = pf;
    }

    agg->packet_saved_count++;
}

static void UpdateFlowAggFromPacket(const Packet *p,
                                    uint64_t *packet_flow_iat_us,
                                    uint64_t *packet_direction_iat_us)
{
    if (packet_flow_iat_us != NULL)
        *packet_flow_iat_us = 0;
    if (packet_direction_iat_us != NULL)
        *packet_direction_iat_us = 0;

    if (p == NULL || p->flow == NULL)
        return;

    uint64_t flow_id = FlowGetId(p->flow);
    uint64_t ts_us = PacketTsUsec(p);
    int direction = GetPacketDirection(p);

    if (direction != 0 && direction != 1)
        direction = 0;

    uint32_t ip_len = GetIpPacketLen(p);
    uint32_t cic_len = GetCicPacketLen(p);
    uint32_t header_len = GetHeaderLenForCIC(p);
    int ttl = GetIpTTLOrHopLimit(p);

    int packet_label = p->alerts.cnt > 0 ? 1 : 0;

    uint64_t local_flow_iat_us = 0;
    uint64_t local_direction_iat_us = 0;

    pthread_mutex_lock(&g_agg_mutex);

    FlowAgg *agg = FlowAggGetOrCreateLocked(flow_id);
    if (agg == NULL) {
        pthread_mutex_unlock(&g_agg_mutex);
        return;
    }

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

    if (agg->last_pkt_ts_us != 0 && ts_us >= agg->last_pkt_ts_us) {
        uint64_t iat = ts_us - agg->last_pkt_ts_us;
        local_flow_iat_us = iat;

        if (packet_flow_iat_us != NULL)
            *packet_flow_iat_us = iat;

        StatAdd(&agg->flow_iat_stats, (double)iat);
        UpdateActiveIdleStats(agg, ts_us, iat);
    } else {
        UpdateActiveIdleStats(agg, ts_us, 0);
    }
    agg->last_pkt_ts_us = ts_us;

    if (agg->last_iat_ts_us[direction] != 0 && ts_us >= agg->last_iat_ts_us[direction]) {
        uint64_t dir_iat = ts_us - agg->last_iat_ts_us[direction];
        local_direction_iat_us = dir_iat;

        if (packet_direction_iat_us != NULL)
            *packet_direction_iat_us = dir_iat;

        StatAdd(&agg->dir_iat_stats[direction], (double)dir_iat);
    }
    agg->last_iat_ts_us[direction] = ts_us;

    if (agg->first_ts_us == 0 || ts_us < agg->first_ts_us)
        agg->first_ts_us = ts_us;

    if (ts_us > agg->last_ts_us)
        agg->last_ts_us = ts_us;

    if (agg->first_dir_ts_us[direction] == 0 || ts_us < agg->first_dir_ts_us[direction])
        agg->first_dir_ts_us[direction] = ts_us;

    if (ts_us > agg->last_dir_ts_us[direction])
        agg->last_dir_ts_us[direction] = ts_us;

    agg->pkts[direction]++;
    agg->bytes[direction] += cic_len;

    StatAdd(&agg->pkt_len_stats_all, (double)cic_len);
    StatAdd(&agg->pkt_len_stats[direction], (double)cic_len);

    agg->header_len[direction] += header_len;

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

        if (flags & TH_FIN)
            agg->fin_count++;
        if (flags & TH_SYN)
            agg->syn_count++;
        if (flags & TH_RST)
            agg->rst_count++;
        if (flags & TH_PUSH)
            agg->psh_count++;
        if (flags & TH_ACK)
            agg->ack_count++;
        if (flags & TH_URG)
            agg->urg_count++;
        if (flags & TH_CWR)
            agg->cwe_count++;
        if (flags & TH_ECE)
            agg->ece_count++;

        if (direction == 0 && (flags & TH_PUSH))
            agg->fwd_psh_flags++;
        if (direction == 1 && (flags & TH_PUSH))
            agg->bwd_psh_flags++;

        if (direction == 0 && (flags & TH_URG))
            agg->fwd_urg_flags++;
        if (direction == 1 && (flags & TH_URG))
            agg->bwd_urg_flags++;

        if (direction == 0 && !agg->init_win_fwd_seen) {
            agg->init_win_bytes_forward = win;
            agg->init_win_fwd_seen = true;
        }

        if (direction == 1 && !agg->init_win_bwd_seen) {
            agg->init_win_bytes_backward = win;
            agg->init_win_bwd_seen = true;
        }

        if (direction == 0 && p->payload_len > 0)
            agg->act_data_pkt_fwd++;

        if (direction == 0) {
            uint32_t seg_size = GetL4HeaderLen(p);
            if (seg_size > 0 && seg_size < agg->min_seg_size_forward)
                agg->min_seg_size_forward = seg_size;
        }
    }

    if (p->proto == IPPROTO_ICMP && p->l4.hdrs.icmpv4h != NULL) {
        const ICMPV4Hdr *icmp4 = p->l4.hdrs.icmpv4h;
        agg->icmp_seen = true;
        agg->icmp_type = icmp4->type;
        agg->icmp_code = icmp4->code;
    }

    AppendPacketFeatureLocked(agg,
                              p,
                              direction,
                              local_flow_iat_us,
                              local_direction_iat_us,
                              packet_label);

    pthread_mutex_unlock(&g_agg_mutex);
}

static void WritePacketCsvLine(FILE *fp, const Packet *p,
                               uint64_t flow_iat_us,
                               uint64_t direction_iat_us)
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

    int packet_label = p->alerts.cnt > 0 ? 1 : 0;

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
            "%lu,"
            "%lu,"
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
            (unsigned long)flow_iat_us,
            (unsigned long)direction_iat_us,
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

static void ComputeFlowOutputValues(Flow *f, const FlowAgg *agg,
                                    uint64_t *fwd_pkts,
                                    uint64_t *bwd_pkts,
                                    double *fwd_bytes,
                                    double *bwd_bytes,
                                    uint64_t *flow_duration_us,
                                    uint64_t *fwd_iat_total,
                                    uint64_t *bwd_iat_total,
                                    int *label,
                                    StatAgg *active_tmp)
{
    *fwd_pkts = 0;
    *bwd_pkts = 0;
    *fwd_bytes = 0.0;
    *bwd_bytes = 0.0;
    *flow_duration_us = 0;
    *fwd_iat_total = 0;
    *bwd_iat_total = 0;
    *label = 0;
    memset(active_tmp, 0, sizeof(*active_tmp));

    if (agg != NULL) {
        *fwd_pkts = agg->pkts[0];
        *bwd_pkts = agg->pkts[1];
        *fwd_bytes = agg->pkt_len_stats[0].sum;
        *bwd_bytes = agg->pkt_len_stats[1].sum;
        *flow_duration_us = DurationUsec(agg->first_ts_us, agg->last_ts_us);
        *fwd_iat_total = DurationUsec(agg->first_dir_ts_us[0], agg->last_dir_ts_us[0]);
        *bwd_iat_total = DurationUsec(agg->first_dir_ts_us[1], agg->last_dir_ts_us[1]);

        *active_tmp = agg->active_stats;
        if (agg->active_start_us != 0 && agg->active_end_us >= agg->active_start_us) {
            StatAdd(active_tmp, (double)(agg->active_end_us - agg->active_start_us));
        }

        if (agg->flow_label == 1)
            *label = 1;
    } else {
        *fwd_pkts = f->todstpktcnt;
        *bwd_pkts = f->tosrcpktcnt;
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

    uint64_t fwd_pkts = 0;
    uint64_t bwd_pkts = 0;
    double fwd_bytes = 0.0;
    double bwd_bytes = 0.0;
    uint64_t flow_duration_us = 0;
    uint64_t fwd_iat_total = 0;
    uint64_t bwd_iat_total = 0;
    int label = 0;
    StatAgg active_tmp;

    ComputeFlowOutputValues(f, agg,
                            &fwd_pkts,
                            &bwd_pkts,
                            &fwd_bytes,
                            &bwd_bytes,
                            &flow_duration_us,
                            &fwd_iat_total,
                            &bwd_iat_total,
                            &label,
                            &active_tmp);

    double total_bytes = fwd_bytes + bwd_bytes;
    double total_pkts = (double)(fwd_pkts + bwd_pkts);
    double flow_duration = (double)flow_duration_us;

    double flow_bytes_s = 0.0;
    double flow_packets_s = 0.0;
    double fwd_packets_s = 0.0;
    double bwd_packets_s = 0.0;

    if (flow_duration_us > 0) {
        double dur_s = flow_duration_us / 1000000.0;
        flow_bytes_s = total_bytes / dur_s;
        flow_packets_s = total_pkts / dur_s;
        fwd_packets_s = (double)fwd_pkts / dur_s;
        bwd_packets_s = (double)bwd_pkts / dur_s;
    }

    double down_up_ratio = 0.0;
    if (fwd_pkts > 0)
        down_up_ratio = (double)bwd_pkts / (double)fwd_pkts;

    double average_packet_size = 0.0;
    if (total_pkts > 0.0)
        average_packet_size = total_bytes / total_pkts;

    double avg_fwd_segment_size = agg != NULL ? StatMean(&agg->pkt_len_stats[0]) : 0.0;
    double avg_bwd_segment_size = agg != NULL ? StatMean(&agg->pkt_len_stats[1]) : 0.0;

    uint32_t min_seg_size_forward = 0;
    if (agg != NULL && agg->min_seg_size_forward != UINT32_MAX)
        min_seg_size_forward = agg->min_seg_size_forward;

    pthread_mutex_lock(&g_file_mutex);

    fprintf(fp,
            "%lu,"
            "%s,"
            "%u,"
            "%s,"
            "%u,"
            "%u,"
            "%.0f,"
            "%lu,"
            "%lu,"
            "%.0f,"
            "%.0f,"
            "%.0f,"
            "%.0f,"
            "%.6f,"
            "%.6f,"
            "%.0f,"
            "%.0f,"
            "%.6f,"
            "%.6f,"
            "%.6f,"
            "%.6f,"
            "%.6f,"
            "%.6f,"
            "%.0f,"
            "%.0f,"
            "%lu,"
            "%.6f,"
            "%.6f,"
            "%.0f,"
            "%.0f,"
            "%lu,"
            "%.6f,"
            "%.6f,"
            "%.0f,"
            "%.0f,"
            "%lu,"
            "%lu,"
            "%lu,"
            "%lu,"
            "%lu,"
            "%lu,"
            "%.6f,"
            "%.6f,"
            "%.0f,"
            "%.0f,"
            "%.6f,"
            "%.6f,"
            "%.6f,"
            "%lu,"
            "%lu,"
            "%lu,"
            "%lu,"
            "%lu,"
            "%lu,"
            "%lu,"
            "%lu,"
            "%.6f,"
            "%.6f,"
            "%.6f,"
            "%.6f,"
            "%d,"
            "%d,"
            "%lu,"
            "%u,"
            "%.6f,"
            "%.6f,"
            "%.0f,"
            "%.0f,"
            "%.6f,"
            "%.6f,"
            "%.0f,"
            "%.0f,"
            "%d\n",
            (unsigned long)flow_id,
            src_ip,
            src_port,
            dst_ip,
            dst_port,
            f->proto,
            flow_duration,
            (unsigned long)fwd_pkts,
            (unsigned long)bwd_pkts,
            fwd_bytes,
            bwd_bytes,
            agg != NULL ? StatMax(&agg->pkt_len_stats[0]) : 0.0,
            agg != NULL ? StatMin(&agg->pkt_len_stats[0]) : 0.0,
            agg != NULL ? StatMean(&agg->pkt_len_stats[0]) : 0.0,
            agg != NULL ? StatStd(&agg->pkt_len_stats[0]) : 0.0,
            agg != NULL ? StatMax(&agg->pkt_len_stats[1]) : 0.0,
            agg != NULL ? StatMin(&agg->pkt_len_stats[1]) : 0.0,
            agg != NULL ? StatMean(&agg->pkt_len_stats[1]) : 0.0,
            agg != NULL ? StatStd(&agg->pkt_len_stats[1]) : 0.0,
            flow_bytes_s,
            flow_packets_s,
            agg != NULL ? StatMean(&agg->flow_iat_stats) : 0.0,
            agg != NULL ? StatStd(&agg->flow_iat_stats) : 0.0,
            agg != NULL ? StatMax(&agg->flow_iat_stats) : 0.0,
            agg != NULL ? StatMin(&agg->flow_iat_stats) : 0.0,
            (unsigned long)fwd_iat_total,
            agg != NULL ? StatMean(&agg->dir_iat_stats[0]) : 0.0,
            agg != NULL ? StatStd(&agg->dir_iat_stats[0]) : 0.0,
            agg != NULL ? StatMax(&agg->dir_iat_stats[0]) : 0.0,
            agg != NULL ? StatMin(&agg->dir_iat_stats[0]) : 0.0,
            (unsigned long)bwd_iat_total,
            agg != NULL ? StatMean(&agg->dir_iat_stats[1]) : 0.0,
            agg != NULL ? StatStd(&agg->dir_iat_stats[1]) : 0.0,
            agg != NULL ? StatMax(&agg->dir_iat_stats[1]) : 0.0,
            agg != NULL ? StatMin(&agg->dir_iat_stats[1]) : 0.0,
            agg != NULL ? (unsigned long)agg->fwd_psh_flags : 0UL,
            agg != NULL ? (unsigned long)agg->bwd_psh_flags : 0UL,
            agg != NULL ? (unsigned long)agg->fwd_urg_flags : 0UL,
            agg != NULL ? (unsigned long)agg->bwd_urg_flags : 0UL,
            agg != NULL ? (unsigned long)agg->header_len[0] : 0UL,
            agg != NULL ? (unsigned long)agg->header_len[1] : 0UL,
            fwd_packets_s,
            bwd_packets_s,
            agg != NULL ? StatMin(&agg->pkt_len_stats_all) : 0.0,
            agg != NULL ? StatMax(&agg->pkt_len_stats_all) : 0.0,
            agg != NULL ? StatMean(&agg->pkt_len_stats_all) : 0.0,
            agg != NULL ? StatStd(&agg->pkt_len_stats_all) : 0.0,
            agg != NULL ? StatVar(&agg->pkt_len_stats_all) : 0.0,
            agg != NULL ? (unsigned long)agg->fin_count : 0UL,
            agg != NULL ? (unsigned long)agg->syn_count : 0UL,
            agg != NULL ? (unsigned long)agg->rst_count : 0UL,
            agg != NULL ? (unsigned long)agg->psh_count : 0UL,
            agg != NULL ? (unsigned long)agg->ack_count : 0UL,
            agg != NULL ? (unsigned long)agg->urg_count : 0UL,
            agg != NULL ? (unsigned long)agg->cwe_count : 0UL,
            agg != NULL ? (unsigned long)agg->ece_count : 0UL,
            down_up_ratio,
            average_packet_size,
            avg_fwd_segment_size,
            avg_bwd_segment_size,
            agg != NULL ? agg->init_win_bytes_forward : -1,
            agg != NULL ? agg->init_win_bytes_backward : -1,
            agg != NULL ? (unsigned long)agg->act_data_pkt_fwd : 0UL,
            min_seg_size_forward,
            StatMean(&active_tmp),
            StatStd(&active_tmp),
            StatMax(&active_tmp),
            StatMin(&active_tmp),
            agg != NULL ? StatMean(&agg->idle_stats) : 0.0,
            agg != NULL ? StatStd(&agg->idle_stats) : 0.0,
            agg != NULL ? StatMax(&agg->idle_stats) : 0.0,
            agg != NULL ? StatMin(&agg->idle_stats) : 0.0,
            label);

    fflush(fp);

    pthread_mutex_unlock(&g_file_mutex);
}

static void WritePacketFeatureJson(FILE *fp, const PacketFeature *pf)
{
    fprintf(fp, "{");

    fprintf(fp, "\"record_type\":");
    JsonWriteEscapedString(fp, pf->record_type);

    fprintf(fp, ",\"packet_id\":%lu", (unsigned long)pf->packet_id);
    fprintf(fp, ",\"timestamp_us\":%lu", (unsigned long)pf->timestamp_us);
    fprintf(fp, ",\"ts_sec\":%lu", (unsigned long)pf->ts_sec);
    fprintf(fp, ",\"ts_usec\":%lu", (unsigned long)pf->ts_usec);
    fprintf(fp, ",\"flow_id\":%lu", (unsigned long)pf->flow_id);
    fprintf(fp, ",\"direction\":%d", pf->direction);
    fprintf(fp, ",\"ip_version\":%d", pf->ip_version);

    fprintf(fp, ",\"src_ip\":");
    JsonWriteEscapedString(fp, pf->src_ip);

    fprintf(fp, ",\"dst_ip\":");
    JsonWriteEscapedString(fp, pf->dst_ip);

    fprintf(fp, ",\"src_port\":%u", pf->src_port);
    fprintf(fp, ",\"dst_port\":%u", pf->dst_port);
    fprintf(fp, ",\"protocol\":%u", pf->protocol);
    fprintf(fp, ",\"pkt_len\":%u", pf->pkt_len);
    fprintf(fp, ",\"ip_len\":%u", pf->ip_len);
    fprintf(fp, ",\"payload_len\":%u", pf->payload_len);
    fprintf(fp, ",\"ttl_or_hop_limit\":%d", pf->ttl_or_hop_limit);
    fprintf(fp, ",\"tcp_seq\":%u", pf->tcp_seq);
    fprintf(fp, ",\"tcp_ack\":%u", pf->tcp_ack);
    fprintf(fp, ",\"tcp_flags\":%d", pf->tcp_flags);
    fprintf(fp, ",\"tcp_window\":%d", pf->tcp_window);
    fprintf(fp, ",\"tcp_header_len\":%d", pf->tcp_header_len);
    fprintf(fp, ",\"udp_len\":%d", pf->udp_len);
    fprintf(fp, ",\"icmp_type\":%d", pf->icmp_type);
    fprintf(fp, ",\"icmp_code\":%d", pf->icmp_code);
    fprintf(fp, ",\"l7_proto\":%d", pf->l7_proto);
    fprintf(fp, ",\"flow_iat_us\":%lu", (unsigned long)pf->flow_iat_us);
    fprintf(fp, ",\"direction_iat_us\":%lu", (unsigned long)pf->direction_iat_us);
    fprintf(fp, ",\"packet_label\":%d", pf->packet_label);

    fprintf(fp, "}");
}

static void WriteNestedFlowJsonLine(FILE *fp, Flow *f, const FlowAgg *agg)
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

    uint64_t fwd_pkts = 0;
    uint64_t bwd_pkts = 0;
    double fwd_bytes = 0.0;
    double bwd_bytes = 0.0;
    uint64_t flow_duration_us = 0;
    uint64_t fwd_iat_total = 0;
    uint64_t bwd_iat_total = 0;
    int label = 0;
    StatAgg active_tmp;

    ComputeFlowOutputValues(f, agg,
                            &fwd_pkts,
                            &bwd_pkts,
                            &fwd_bytes,
                            &bwd_bytes,
                            &flow_duration_us,
                            &fwd_iat_total,
                            &bwd_iat_total,
                            &label,
                            &active_tmp);

    double total_bytes = fwd_bytes + bwd_bytes;
    double total_pkts = (double)(fwd_pkts + bwd_pkts);
    double flow_duration = (double)flow_duration_us;

    double flow_bytes_s = 0.0;
    double flow_packets_s = 0.0;
    double fwd_packets_s = 0.0;
    double bwd_packets_s = 0.0;

    if (flow_duration_us > 0) {
        double dur_s = flow_duration_us / 1000000.0;
        flow_bytes_s = total_bytes / dur_s;
        flow_packets_s = total_pkts / dur_s;
        fwd_packets_s = (double)fwd_pkts / dur_s;
        bwd_packets_s = (double)bwd_pkts / dur_s;
    }

    double down_up_ratio = 0.0;
    if (fwd_pkts > 0)
        down_up_ratio = (double)bwd_pkts / (double)fwd_pkts;

    double average_packet_size = 0.0;
    if (total_pkts > 0.0)
        average_packet_size = total_bytes / total_pkts;

    double avg_fwd_segment_size = agg != NULL ? StatMean(&agg->pkt_len_stats[0]) : 0.0;
    double avg_bwd_segment_size = agg != NULL ? StatMean(&agg->pkt_len_stats[1]) : 0.0;

    uint32_t min_seg_size_forward = 0;
    if (agg != NULL && agg->min_seg_size_forward != UINT32_MAX)
        min_seg_size_forward = agg->min_seg_size_forward;

    pthread_mutex_lock(&g_file_mutex);

    fprintf(fp, "{");

    fprintf(fp, "\"flow_id\":%lu", (unsigned long)flow_id);

    fprintf(fp, ",\"source_ip\":");
    JsonWriteEscapedString(fp, src_ip);

    fprintf(fp, ",\"source_port\":%u", src_port);

    fprintf(fp, ",\"destination_ip\":");
    JsonWriteEscapedString(fp, dst_ip);

    fprintf(fp, ",\"destination_port\":%u", dst_port);
    fprintf(fp, ",\"protocol\":%u", f->proto);
    fprintf(fp, ",\"flow_duration\":%.0f", flow_duration);
    fprintf(fp, ",\"total_fwd_packets\":%lu", (unsigned long)fwd_pkts);
    fprintf(fp, ",\"total_backward_packets\":%lu", (unsigned long)bwd_pkts);
    fprintf(fp, ",\"total_length_of_fwd_packets\":%.0f", fwd_bytes);
    fprintf(fp, ",\"total_length_of_bwd_packets\":%.0f", bwd_bytes);

    fprintf(fp, ",\"fwd_packet_length_max\":%.0f", agg ? StatMax(&agg->pkt_len_stats[0]) : 0.0);
    fprintf(fp, ",\"fwd_packet_length_min\":%.0f", agg ? StatMin(&agg->pkt_len_stats[0]) : 0.0);
    fprintf(fp, ",\"fwd_packet_length_mean\":%.6f", agg ? StatMean(&agg->pkt_len_stats[0]) : 0.0);
    fprintf(fp, ",\"fwd_packet_length_std\":%.6f", agg ? StatStd(&agg->pkt_len_stats[0]) : 0.0);

    fprintf(fp, ",\"bwd_packet_length_max\":%.0f", agg ? StatMax(&agg->pkt_len_stats[1]) : 0.0);
    fprintf(fp, ",\"bwd_packet_length_min\":%.0f", agg ? StatMin(&agg->pkt_len_stats[1]) : 0.0);
    fprintf(fp, ",\"bwd_packet_length_mean\":%.6f", agg ? StatMean(&agg->pkt_len_stats[1]) : 0.0);
    fprintf(fp, ",\"bwd_packet_length_std\":%.6f", agg ? StatStd(&agg->pkt_len_stats[1]) : 0.0);

    fprintf(fp, ",\"flow_bytes_per_s\":%.6f", flow_bytes_s);
    fprintf(fp, ",\"flow_packets_per_s\":%.6f", flow_packets_s);

    fprintf(fp, ",\"flow_iat_mean\":%.6f", agg ? StatMean(&agg->flow_iat_stats) : 0.0);
    fprintf(fp, ",\"flow_iat_std\":%.6f", agg ? StatStd(&agg->flow_iat_stats) : 0.0);
    fprintf(fp, ",\"flow_iat_max\":%.0f", agg ? StatMax(&agg->flow_iat_stats) : 0.0);
    fprintf(fp, ",\"flow_iat_min\":%.0f", agg ? StatMin(&agg->flow_iat_stats) : 0.0);

    fprintf(fp, ",\"fwd_iat_total\":%lu", (unsigned long)fwd_iat_total);
    fprintf(fp, ",\"fwd_iat_mean\":%.6f", agg ? StatMean(&agg->dir_iat_stats[0]) : 0.0);
    fprintf(fp, ",\"fwd_iat_std\":%.6f", agg ? StatStd(&agg->dir_iat_stats[0]) : 0.0);
    fprintf(fp, ",\"fwd_iat_max\":%.0f", agg ? StatMax(&agg->dir_iat_stats[0]) : 0.0);
    fprintf(fp, ",\"fwd_iat_min\":%.0f", agg ? StatMin(&agg->dir_iat_stats[0]) : 0.0);

    fprintf(fp, ",\"bwd_iat_total\":%lu", (unsigned long)bwd_iat_total);
    fprintf(fp, ",\"bwd_iat_mean\":%.6f", agg ? StatMean(&agg->dir_iat_stats[1]) : 0.0);
    fprintf(fp, ",\"bwd_iat_std\":%.6f", agg ? StatStd(&agg->dir_iat_stats[1]) : 0.0);
    fprintf(fp, ",\"bwd_iat_max\":%.0f", agg ? StatMax(&agg->dir_iat_stats[1]) : 0.0);
    fprintf(fp, ",\"bwd_iat_min\":%.0f", agg ? StatMin(&agg->dir_iat_stats[1]) : 0.0);

    fprintf(fp, ",\"fwd_psh_flags\":%lu", agg ? (unsigned long)agg->fwd_psh_flags : 0UL);
    fprintf(fp, ",\"bwd_psh_flags\":%lu", agg ? (unsigned long)agg->bwd_psh_flags : 0UL);
    fprintf(fp, ",\"fwd_urg_flags\":%lu", agg ? (unsigned long)agg->fwd_urg_flags : 0UL);
    fprintf(fp, ",\"bwd_urg_flags\":%lu", agg ? (unsigned long)agg->bwd_urg_flags : 0UL);

    fprintf(fp, ",\"fwd_header_length\":%lu", agg ? (unsigned long)agg->header_len[0] : 0UL);
    fprintf(fp, ",\"bwd_header_length\":%lu", agg ? (unsigned long)agg->header_len[1] : 0UL);

    fprintf(fp, ",\"fwd_packets_per_s\":%.6f", fwd_packets_s);
    fprintf(fp, ",\"bwd_packets_per_s\":%.6f", bwd_packets_s);

    fprintf(fp, ",\"min_packet_length\":%.0f", agg ? StatMin(&agg->pkt_len_stats_all) : 0.0);
    fprintf(fp, ",\"max_packet_length\":%.0f", agg ? StatMax(&agg->pkt_len_stats_all) : 0.0);
    fprintf(fp, ",\"packet_length_mean\":%.6f", agg ? StatMean(&agg->pkt_len_stats_all) : 0.0);
    fprintf(fp, ",\"packet_length_std\":%.6f", agg ? StatStd(&agg->pkt_len_stats_all) : 0.0);
    fprintf(fp, ",\"packet_length_variance\":%.6f", agg ? StatVar(&agg->pkt_len_stats_all) : 0.0);

    fprintf(fp, ",\"fin_flag_count\":%lu", agg ? (unsigned long)agg->fin_count : 0UL);
    fprintf(fp, ",\"syn_flag_count\":%lu", agg ? (unsigned long)agg->syn_count : 0UL);
    fprintf(fp, ",\"rst_flag_count\":%lu", agg ? (unsigned long)agg->rst_count : 0UL);
    fprintf(fp, ",\"psh_flag_count\":%lu", agg ? (unsigned long)agg->psh_count : 0UL);
    fprintf(fp, ",\"ack_flag_count\":%lu", agg ? (unsigned long)agg->ack_count : 0UL);
    fprintf(fp, ",\"urg_flag_count\":%lu", agg ? (unsigned long)agg->urg_count : 0UL);
    fprintf(fp, ",\"cwe_flag_count\":%lu", agg ? (unsigned long)agg->cwe_count : 0UL);
    fprintf(fp, ",\"ece_flag_count\":%lu", agg ? (unsigned long)agg->ece_count : 0UL);

    fprintf(fp, ",\"down_up_ratio\":%.6f", down_up_ratio);
    fprintf(fp, ",\"average_packet_size\":%.6f", average_packet_size);
    fprintf(fp, ",\"avg_fwd_segment_size\":%.6f", avg_fwd_segment_size);
    fprintf(fp, ",\"avg_bwd_segment_size\":%.6f", avg_bwd_segment_size);

    fprintf(fp, ",\"init_win_bytes_forward\":%d", agg ? agg->init_win_bytes_forward : -1);
    fprintf(fp, ",\"init_win_bytes_backward\":%d", agg ? agg->init_win_bytes_backward : -1);
    fprintf(fp, ",\"act_data_pkt_fwd\":%lu", agg ? (unsigned long)agg->act_data_pkt_fwd : 0UL);
    fprintf(fp, ",\"min_seg_size_forward\":%u", min_seg_size_forward);

    fprintf(fp, ",\"active_mean\":%.6f", StatMean(&active_tmp));
    fprintf(fp, ",\"active_std\":%.6f", StatStd(&active_tmp));
    fprintf(fp, ",\"active_max\":%.0f", StatMax(&active_tmp));
    fprintf(fp, ",\"active_min\":%.0f", StatMin(&active_tmp));

    fprintf(fp, ",\"idle_mean\":%.6f", agg ? StatMean(&agg->idle_stats) : 0.0);
    fprintf(fp, ",\"idle_std\":%.6f", agg ? StatStd(&agg->idle_stats) : 0.0);
    fprintf(fp, ",\"idle_max\":%.0f", agg ? StatMax(&agg->idle_stats) : 0.0);
    fprintf(fp, ",\"idle_min\":%.0f", agg ? StatMin(&agg->idle_stats) : 0.0);

    fprintf(fp, ",\"label\":%d", label);

    fprintf(fp, ",\"packet_total_count\":%lu", agg ? (unsigned long)agg->packet_total_count : 0UL);
    fprintf(fp, ",\"packet_saved_count\":%lu", agg ? (unsigned long)agg->packet_saved_count : 0UL);
    fprintf(fp, ",\"packet_truncated\":%s", (agg && agg->packet_truncated) ? "true" : "false");

    fprintf(fp, ",\"packet\":[");

    if (agg != NULL) {
        const PacketFeature *pf = agg->pkt_head;
        bool first = true;

        while (pf != NULL) {
            if (!first)
                fprintf(fp, ",");

            WritePacketFeatureJson(fp, pf);

            first = false;
            pf = pf->next;
        }
    }

    fprintf(fp, "]}");
    fprintf(fp, "\n");

    fflush(fp);

    pthread_mutex_unlock(&g_file_mutex);
}

static int CustomPacketCsvLogger(ThreadVars *tv, void *thread_data, const Packet *p)
{
    (void)tv;

    CustomLoggerThreadData *tdata = (CustomLoggerThreadData *)thread_data;

    if (tdata == NULL || p == NULL)
        return 0;

    uint64_t flow_iat_us = 0;
    uint64_t direction_iat_us = 0;

    UpdateFlowAggFromPacket(p, &flow_iat_us, &direction_iat_us);
    WritePacketCsvLine(tdata->packet_fp, p, flow_iat_us, direction_iat_us);

    return 0;
}

static int CustomFlowCsvLogger(ThreadVars *tv, void *thread_data, Flow *f)
{
    (void)tv;

    CustomLoggerThreadData *tdata = (CustomLoggerThreadData *)thread_data;

    if (tdata == NULL || f == NULL)
        return 0;

    uint64_t flow_id = FlowGetId(f);

    pthread_mutex_lock(&g_agg_mutex);
    FlowAgg *agg_ptr = FlowAggDetachLocked(flow_id);
    pthread_mutex_unlock(&g_agg_mutex);

    WriteFlowCsvLine(tdata->flow_fp, f, agg_ptr);
    WriteNestedFlowJsonLine(tdata->nested_fp, f, agg_ptr);

    FreeFlowAgg(agg_ptr);

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
        SCLogError("Could not allocate custom CIC CSV/JSONL logger thread data");
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

    tdata->nested_fp = fopen(CUSTOM_NESTED_JSONL_FILE, "a");
    if (tdata->nested_fp == NULL) {
        SCLogError("Could not open nested JSONL file: %s", CUSTOM_NESTED_JSONL_FILE);
        fclose(tdata->flow_fp);
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

        if (tdata->nested_fp != NULL)
            fclose(tdata->nested_fp);

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
    .name = "custom-cic-style-csv-jsonl-logger",
    .plugin_version = "4.0.0",
    .author = "Xiaoyan Xiong",
    .license = "GPLv2",
    .Init = Init,
};

const SCPlugin *SCPluginRegister(void)
{
    return &PluginRegistration;
}
