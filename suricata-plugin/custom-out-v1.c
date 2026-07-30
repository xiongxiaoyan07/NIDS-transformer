#include "suricata-common.h"
#include "suricata-plugin.h"
#include "output-packet.h"
#include "util-print.h"
#include "decode.h"
#include "decode-ipv4.h"
#include "decode-ipv6.h"
#include "decode-tcp.h"
#include "decode-udp.h"
#include "flow.h"
#include "util-time.h"
#include "jansson.h"

#include <stdio.h>
#include <pthread.h>
#include <string.h>
#include <time.h>

#define CUSTOM_PACKET_LOG_FILE "/home/xxiong/pcaps/custom-nids-features.json"

static pthread_mutex_t tdata_mutex = PTHREAD_MUTEX_INITIALIZER;

typedef struct {
    FILE *fp;
} CustomLoggerThreadData;

static void SetDefaultFields(json_t *js)
{
    json_object_set_new(js, "packet_id", json_integer(-1));
    json_object_set_new(js, "timestamp", json_string(""));
    json_object_set_new(js, "ts_sec", json_integer(-1));
    json_object_set_new(js, "ts_usec", json_integer(-1));

    json_object_set_new(js, "pkt_len", json_integer(-1));
    json_object_set_new(js, "payload_len", json_integer(0));
    json_object_set_new(js, "l4_proto", json_integer(-1));
    json_object_set_new(js, "packet_label", json_integer(0));
    json_object_set_new(js, "flow_label", json_integer(0));

    json_object_set_new(js, "flow_id", json_integer(-1));
    json_object_set_new(js, "flow_pkts_tosrc", json_integer(0));
    json_object_set_new(js, "flow_pkts_todst", json_integer(0));
    json_object_set_new(js, "flow_pkts_total", json_integer(0));
//    json_object_set_new(js, "flow_bytes_tosrc", json_integer(-1));
//    json_object_set_new(js, "flow_bytes_todst", json_integer(-1));
//    json_object_set_new(js, "flow_bytes_total", json_integer(-1));
    json_object_set_new(js, "flow_duration_so_far", json_real(0.0));
    json_object_set_new(js, "flow_pkts_per_sec", json_real(0.0));

    json_object_set_new(js, "ip_version", json_integer(0));
    json_object_set_new(js, "src_ip", json_string(""));
    json_object_set_new(js, "dst_ip", json_string(""));

    json_object_set_new(js, "ip_header_len", json_integer(-1));
    json_object_set_new(js, "ip_tos", json_integer(-1));
    json_object_set_new(js, "ip_len", json_integer(-1));
    json_object_set_new(js, "ip_id", json_integer(-1));
    json_object_set_new(js, "ip_ttl", json_integer(-1));
    json_object_set_new(js, "ip_proto", json_integer(-1));
    json_object_set_new(js, "ip_checksum", json_integer(-1));
    json_object_set_new(js, "ip_frag_offset", json_integer(0));
    json_object_set_new(js, "ip_flag_df", json_integer(0));
    json_object_set_new(js, "ip_flag_mf", json_integer(0));

    json_object_set_new(js, "ipv6_traffic_class", json_integer(-1));
    json_object_set_new(js, "ipv6_flow_label", json_integer(-1));
    json_object_set_new(js, "ipv6_payload_len", json_integer(-1));
    json_object_set_new(js, "ipv6_next_header", json_integer(-1));
    json_object_set_new(js, "ipv6_hop_limit", json_integer(-1));

    json_object_set_new(js, "src_port", json_integer(-1));
    json_object_set_new(js, "dst_port", json_integer(-1));

    json_object_set_new(js, "tcp_seq", json_integer(-1));
    json_object_set_new(js, "tcp_ack", json_integer(-1));
    json_object_set_new(js, "tcp_flags", json_integer(0));
    json_object_set_new(js, "tcp_flag_fin", json_integer(0));
    json_object_set_new(js, "tcp_flag_syn", json_integer(0));
    json_object_set_new(js, "tcp_flag_rst", json_integer(0));
    json_object_set_new(js, "tcp_flag_psh", json_integer(0));
    json_object_set_new(js, "tcp_flag_ack", json_integer(0));
    json_object_set_new(js, "tcp_flag_urg", json_integer(0));
    json_object_set_new(js, "tcp_flag_ece", json_integer(0));
    json_object_set_new(js, "tcp_flag_cwr", json_integer(0));
    json_object_set_new(js, "tcp_window", json_integer(-1));
    json_object_set_new(js, "tcp_checksum", json_integer(-1));
    json_object_set_new(js, "tcp_urgent_ptr", json_integer(-1));
    json_object_set_new(js, "tcp_header_len", json_integer(-1));

    json_object_set_new(js, "udp_len", json_integer(-1));
    json_object_set_new(js, "udp_checksum", json_integer(-1));

    json_object_set_new(js, "decode_status", json_string("unknown"));
}

static TmEcode ThreadInit(ThreadVars *tv, const void *initdata, void **data)
{
    (void)tv;
    (void)initdata;

    CustomLoggerThreadData *tdata = SCMalloc(sizeof(CustomLoggerThreadData));
    if (!tdata) {
        SCLogError("Could not allocate thread data for custom packet logger");
        return TM_ECODE_FAILED;
    }

    tdata->fp = fopen(CUSTOM_PACKET_LOG_FILE, "a");
    if (!tdata->fp) {
        SCLogError("Could not open file %s for writing", CUSTOM_PACKET_LOG_FILE);
        SCFree(tdata);
        return TM_ECODE_FAILED;
    }

    *data = tdata;
    return TM_ECODE_OK;
}

static TmEcode ThreadDeInit(ThreadVars *tv, void *data)
{
    (void)tv;

    CustomLoggerThreadData *tdata = data;
    if (tdata && tdata->fp) {
        fclose(tdata->fp);
        tdata->fp = NULL;
    }
    SCFree(tdata);
    return TM_ECODE_OK;
}

static void AddTimestamp(json_t *js, const Packet *p)
{
    time_t secs = (time_t)p->ts.secs;
    uint32_t usecs = (uint32_t)p->ts.usecs;
    struct tm timeinfo;

    gmtime_r(&secs, &timeinfo);

    char timestamp[64];
    strftime(timestamp, sizeof(timestamp), "%Y-%m-%dT%H:%M:%S", &timeinfo);
    snprintf(timestamp + strlen(timestamp),
             sizeof(timestamp) - strlen(timestamp),
             ".%06uZ", usecs);

    json_object_set_new(js, "timestamp", json_string(timestamp));
    json_object_set_new(js, "ts_sec", json_integer((json_int_t)p->ts.secs));
    json_object_set_new(js, "ts_usec", json_integer((json_int_t)p->ts.usecs));
}

static void AddBasicPacketFeatures(json_t *js, const Packet *p)
{
    json_object_set_new(js, "packet_id", json_integer((json_int_t)p->pcap_cnt));
    json_object_set_new(js, "pkt_len", json_integer((json_int_t)GET_PKT_LEN(p)));
    json_object_set_new(js, "payload_len", json_integer((json_int_t)p->payload_len));
    json_object_set_new(js, "l4_proto", json_integer((json_int_t)p->proto));
//    json_object_set_new(js, "label", json_integer(p->alerts.cnt > 0 ? 1 : 0));
    int packet_label = p->alerts.cnt > 0 ? 1 : 0;
    json_object_set_new(js, "packet_label", json_integer(packet_label));
}

static void AddFlowFeatures(json_t *js, const Packet *p)
{
    if (p->flow == NULL)
        return;

    Flow *f = p->flow;

    uint64_t pkts_tosrc = f->tosrcpktcnt;
    uint64_t pkts_todst = f->todstpktcnt;
    uint64_t pkts_total = pkts_tosrc + pkts_todst;

    json_object_set_new(js, "flow_id", json_integer((json_int_t)FlowGetId(f)));
    json_object_set_new(js, "flow_pkts_tosrc", json_integer((json_int_t)pkts_tosrc));
    json_object_set_new(js, "flow_pkts_todst", json_integer((json_int_t)pkts_todst));
    json_object_set_new(js, "flow_pkts_total", json_integer((json_int_t)pkts_total));

//#ifdef HAVE_FLOW_BYTE_COUNTERS
//    json_object_set_new(js, "flow_bytes_tosrc", json_integer((json_int_t)f->tosrcbytecnt));
//    json_object_set_new(js, "flow_bytes_todst", json_integer((json_int_t)f->todstbytecnt));
//    json_object_set_new(js, "flow_bytes_total",
//                        json_integer((json_int_t)(f->tosrcbytecnt + f->todstbytecnt)));
//#endif

    time_t start_s = SCTIME_SECS(f->startts);
    uint32_t start_us = SCTIME_USECS(f->startts);

    double duration =
        (double)(p->ts.secs - start_s) +
        ((double)((int32_t)p->ts.usecs - (int32_t)start_us) / 1000000.0);

    if (duration < 0.0)
        duration = 0.0;

    json_object_set_new(js, "flow_duration_so_far", json_real(duration));

    if (duration > 0.0) {
        json_object_set_new(js, "flow_pkts_per_sec",
                            json_real((double)pkts_total / duration));
    }
}

static void AddIPv4Features(json_t *js, const Packet *p)
{
    const IPV4Hdr *ipv4 = PacketGetIPv4(p);
    if (ipv4 == NULL)
        return;

    char src_ip[46] = {0};
    char dst_ip[46] = {0};

    PrintInet(AF_INET, &p->src.addr_data32[0], src_ip, sizeof(src_ip));
    PrintInet(AF_INET, &p->dst.addr_data32[0], dst_ip, sizeof(dst_ip));

    uint16_t ip_off = SCNtohs(ipv4->ip_off);

    json_object_set_new(js, "decode_status", json_string("ipv4"));
    json_object_set_new(js, "ip_version", json_integer(4));
    json_object_set_new(js, "src_ip", json_string(src_ip));
    json_object_set_new(js, "dst_ip", json_string(dst_ip));

    json_object_set_new(js, "ip_header_len", json_integer(IPV4_GET_RAW_HLEN(ipv4)));
    json_object_set_new(js, "ip_tos", json_integer(ipv4->ip_tos));
    json_object_set_new(js, "ip_len", json_integer(SCNtohs(ipv4->ip_len)));
    json_object_set_new(js, "ip_id", json_integer(SCNtohs(ipv4->ip_id)));
    json_object_set_new(js, "ip_ttl", json_integer(ipv4->ip_ttl));
    json_object_set_new(js, "ip_proto", json_integer(ipv4->ip_proto));
    json_object_set_new(js, "ip_checksum", json_integer(SCNtohs(ipv4->ip_csum)));

    json_object_set_new(js, "ip_frag_offset", json_integer(ip_off & 0x1FFF));
    json_object_set_new(js, "ip_flag_df", json_integer((ip_off & 0x4000) ? 1 : 0));
    json_object_set_new(js, "ip_flag_mf", json_integer((ip_off & 0x2000) ? 1 : 0));
}

static void AddIPv6Features(json_t *js, const Packet *p)
{
    const IPV6Hdr *ipv6 = PacketGetIPv6(p);
    if (ipv6 == NULL)
        return;

    char src_ip[46] = {0};
    char dst_ip[46] = {0};

    PrintInet(AF_INET6, &p->src.addr_data32[0], src_ip, sizeof(src_ip));
    PrintInet(AF_INET6, &p->dst.addr_data32[0], dst_ip, sizeof(dst_ip));

    uint32_t flow = SCNtohl(ipv6->ip6_hdrun.ip6_un1.ip6_un1_flow);

    json_object_set_new(js, "decode_status", json_string("ipv6"));
    json_object_set_new(js, "ip_version", json_integer(6));
    json_object_set_new(js, "src_ip", json_string(src_ip));
    json_object_set_new(js, "dst_ip", json_string(dst_ip));

    json_object_set_new(js, "ip_header_len", json_integer(40));
    json_object_set_new(js, "ipv6_traffic_class", json_integer((flow >> 20) & 0xFF));
    json_object_set_new(js, "ipv6_flow_label", json_integer(flow & 0x000FFFFF));
    json_object_set_new(js, "ipv6_payload_len",
                        json_integer(SCNtohs(ipv6->ip6_hdrun.ip6_un1.ip6_un1_plen)));
    json_object_set_new(js, "ipv6_next_header",
                        json_integer(ipv6->ip6_hdrun.ip6_un1.ip6_un1_nxt));
    json_object_set_new(js, "ipv6_hop_limit",
                        json_integer(ipv6->ip6_hdrun.ip6_un1.ip6_un1_hlim));
}

static void AddTCPFeatures(json_t *js, const Packet *p)
{
    if (p->l4.hdrs.tcph == NULL)
        return;

    const TCPHdr *tcp = p->l4.hdrs.tcph;
    uint8_t flags = tcp->th_flags;

    json_object_set_new(js, "src_port", json_integer(SCNtohs(tcp->th_sport)));
    json_object_set_new(js, "dst_port", json_integer(SCNtohs(tcp->th_dport)));

    json_object_set_new(js, "tcp_seq", json_integer((json_int_t)SCNtohl(tcp->th_seq)));
    json_object_set_new(js, "tcp_ack", json_integer((json_int_t)SCNtohl(tcp->th_ack)));
    json_object_set_new(js, "tcp_flags", json_integer(flags));

    json_object_set_new(js, "tcp_flag_fin", json_integer((flags & TH_FIN) ? 1 : 0));
    json_object_set_new(js, "tcp_flag_syn", json_integer((flags & TH_SYN) ? 1 : 0));
    json_object_set_new(js, "tcp_flag_rst", json_integer((flags & TH_RST) ? 1 : 0));
    json_object_set_new(js, "tcp_flag_psh", json_integer((flags & TH_PUSH) ? 1 : 0));
    json_object_set_new(js, "tcp_flag_ack", json_integer((flags & TH_ACK) ? 1 : 0));
    json_object_set_new(js, "tcp_flag_urg", json_integer((flags & TH_URG) ? 1 : 0));

//#ifdef TH_ECE
//    json_object_set_new(js, "tcp_flag_ece", json_integer((flags & TH_ECE) ? 1 : 0));
//#endif
//
//#ifdef TH_CWR
//    json_object_set_new(js, "tcp_flag_cwr", json_integer((flags & TH_CWR) ? 1 : 0));
//#endif

    json_object_set_new(js, "tcp_window", json_integer(SCNtohs(tcp->th_win)));
    json_object_set_new(js, "tcp_checksum", json_integer(SCNtohs(tcp->th_sum)));
    json_object_set_new(js, "tcp_urgent_ptr", json_integer(SCNtohs(tcp->th_urp)));
//    json_object_set_new(js, "tcp_header_len", json_integer(TCP_GET_HLEN(tcp)));
    /* 修正这里：TCP_GET_HLEN 需要 Packet *p */
    json_object_set_new(js, "tcp_header_len", json_integer(TCP_GET_HLEN(p)));
}

static void AddUDPFeatures(json_t *js, const Packet *p)
{
    if (p->l4.hdrs.udph == NULL)
        return;

    const UDPHdr *udp = p->l4.hdrs.udph;

    json_object_set_new(js, "src_port", json_integer(SCNtohs(udp->uh_sport)));
    json_object_set_new(js, "dst_port", json_integer(SCNtohs(udp->uh_dport)));
    json_object_set_new(js, "udp_len", json_integer(SCNtohs(udp->uh_len)));
    json_object_set_new(js, "udp_checksum", json_integer(SCNtohs(udp->uh_sum)));
}

static int CustomPacketJsonLogger(ThreadVars *tv, void *thread_data, const Packet *p)
{
    (void)tv;

    CustomLoggerThreadData *tdata = (CustomLoggerThreadData *)thread_data;

    if (tdata == NULL || tdata->fp == NULL || p == NULL)
        return TM_ECODE_OK;

    json_t *js = json_object();
    if (js == NULL)
        return TM_ECODE_OK;

    SetDefaultFields(js);

    AddTimestamp(js, p);
    AddFlowFeatures(js, p);
    AddBasicPacketFeatures(js, p);

    if (PacketIsIPv4(p)) {
        AddIPv4Features(js, p);
    } else if (PacketIsIPv6(p)) {
        AddIPv6Features(js, p);
    } else {
        json_object_set_new(js, "decode_status", json_string("non_ip"));
    }

    if (p->proto == IPPROTO_TCP)
        AddTCPFeatures(js, p);

    if (p->proto == IPPROTO_UDP)
        AddUDPFeatures(js, p);

//    AddTCPFeatures(js, p);
//    AddUDPFeatures(js, p);

//    char *json_str = json_dumps(js, JSON_COMPACT);
//    if (json_str != NULL) {
//        pthread_mutex_lock(&tdata_mutex);
//        fprintf(tdata->fp, "%s\n", json_str);
//        fflush(tdata->fp);
//        pthread_mutex_unlock(&tdata_mutex);
//
//        free(json_str);
//    }
    char *json_str = json_dumps(js, 0);
    if (json_str) {
        pthread_mutex_lock(&tdata_mutex);
        if (tdata->fp) {
            fprintf(tdata->fp, "%s\n", json_str);
            fflush(tdata->fp);
        }
        pthread_mutex_unlock(&tdata_mutex);
        free(json_str);
    }

    json_decref(js);
    return TM_ECODE_OK;
}

static bool CustomPacketLoggerCondition(ThreadVars *tv, void *thread_data, const Packet *p)
{
    (void)tv;
    (void)thread_data;
    (void)p;
    return true;
}

//custom-nids-feature-logger
static void Init(void)
{
    SCOutputRegisterPacketLogger(LOGGER_USER,
                                 "custom-packet-logger",
                                 CustomPacketJsonLogger,
                                 CustomPacketLoggerCondition,
                                 NULL,
                                 ThreadInit,
                                 ThreadDeInit);
}

//custom-nids-feature-logger
const SCPlugin PluginRegistration = {
    .version = SC_API_VERSION,
    .suricata_version = SC_PACKAGE_VERSION,
    .name = "custom-packet-logger",
    .plugin_version = "1.1.0",
    .author = "A",
    .license = "GPLv2",
    .Init = Init,
};

const SCPlugin *SCPluginRegister(void)
{
    return &PluginRegistration;
}