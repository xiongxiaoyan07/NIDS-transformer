#include "suricata-common.h"
#include "suricata-plugin.h"
#include "output-packet.h"
#include "output-flow.h"
#include "output-tx.h"
#include "util-print.h"
#include "output.h"
#include "jansson.h"
#include "decode.h"
#include "decode-ipv4.h"
#include "decode-ipv6.h"
#include "decode-tcp.h"
#include "decode-udp.h"
#include "util-misc.h"
#include "flow.h"
#include "util-atomic.h"
#include "util-time.h"
#include <stdio.h>
#include <pthread.h>

#define CUSTOM_PACKET_LOG_FILE "/home/xxiong/pcaps/custom-packet-v5.json"
static pthread_mutex_t tdata_mutex = PTHREAD_MUTEX_INITIALIZER;

typedef struct {
    FILE *fp;
} CustomLoggerThreadData;

static TmEcode ThreadInit(ThreadVars *tv, const void *initdata, void **data)
{
    (void)tv; (void)initdata; // 消除警告
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
    (void)tv; // 消除警告
    CustomLoggerThreadData *tdata = data;
    if (tdata && tdata->fp) {
        fclose(tdata->fp);
        tdata->fp = NULL;
    }
    SCFree(tdata);
    return TM_ECODE_OK;
}

static int CustomPacketJsonLogger(ThreadVars *tv, void *thread_data, const Packet *p)
{
//    CustomLoggerThreadData *tdata = thread_data;
//    if (!tdata || !tdata->fp) return TM_ECODE_OK;
    (void)tv;
    CustomLoggerThreadData *tdata = (CustomLoggerThreadData *)thread_data;
    // 基础检查
    if (tdata == NULL || tdata->fp == NULL || p == NULL)
        return TM_ECODE_OK;

    json_t* p_object = json_object();
    if (p_object == NULL)
        return TM_ECODE_OK;
//    if (!p_object) return TM_ECODE_OK;

    char src_ip[46] = {0}, dst_ip[46] = {0};

    // --- 准确的包时间提取 ---
    // p->ts.secs 是捕获时的秒数，p->ts.usecs 是微秒
    time_t secs = (time_t)p->ts.secs;
    uint32_t usecs = (uint32_t)p->ts.usecs;
    struct tm timeinfo;
    gmtime_r(&secs, &timeinfo); // 修正：使用线程安全版本 gmtime_r

    char timestamp_str[64];
    // 格式化基本时间
    strftime(timestamp_str, sizeof(timestamp_str), "%Y-%m-%dT%H:%M:%S", &timeinfo);
    // 拼接微秒和时区标识
    snprintf(timestamp_str + strlen(timestamp_str),
             sizeof(timestamp_str) - strlen(timestamp_str),
             ".%06uZ", usecs);

    // 设置 ID 和 准确的时间戳
    // packet_id 建议包含 pcap_cnt 保证唯一性，或使用时间戳组合
    uint64_t packet_id = (uint64_t)p->pcap_cnt;
    json_object_set_new(p_object, "packet_id", json_integer(packet_id));
    json_object_set_new(p_object, "timestamp", json_string(timestamp_str));

    // --- Flow 信息提取 (针对 Suricata 8.0.0-dev 修正) ---
    // --- Flow 信息提取 (针对 Suricata 8.0 生产环境修正) ---
    if (p->flow != NULL) {
        /* 1. 调用源码中的函数动态获取 Flow ID
         * 既然 flow.h 中定义了 FlowGetId，这是最标准的方法
         */
        uint64_t flow_id = FlowGetId(p->flow);
        json_object_set_new(p_object, "flow_id", json_integer(flow_id));

        /* 2. 获取数据包计数
         * 使用编译器之前提示的正确成员名：tosrcpktcnt 和 todstpktcnt
         */
        uint64_t total_pkts = p->flow->tosrcpktcnt + p->flow->todstpktcnt;
        json_object_set_new(p_object, "flow_packet_cnt", json_integer(total_pkts));

        /* 3. 时间差计算
         */
        time_t start_s = SCTIME_SECS(p->flow->startts);
        uint32_t start_us = SCTIME_USECS(p->flow->startts);

        double duration = (double)(p->ts.secs - start_s) +
                          (double)((int32_t)p->ts.usecs - (int32_t)start_us) / 1000000.0;
        json_object_set_new(p_object, "flow_start_offset", json_real(duration));
    } else {
        json_object_set_new(p_object, "flow_id", json_null());
        json_object_set_new(p_object, "flow_packet_cnt", json_integer(0));
    }

    if (PacketIsIPv4(p)) {
        const IPV4Hdr* ipv4 = PacketGetIPv4(p);
        PrintInet(AF_INET, (const void*)&(p->src.addr_data32[0]), src_ip, sizeof(src_ip));
        PrintInet(AF_INET, (const void*)&(p->dst.addr_data32[0]), dst_ip, sizeof(dst_ip));
        json_object_set_new(p_object, "Header Length", json_integer(IPV4_GET_RAW_HLEN(ipv4)));
        json_object_set_new(p_object, "Source IP", json_string(src_ip));
        json_object_set_new(p_object, "Destination IP", json_string(dst_ip));
        json_object_set_new(p_object, "Protocol", json_integer(ipv4->ip_proto));
        json_object_set_new(p_object, "ToS", json_integer(ipv4->ip_tos));
        //json_object_set_new(p_object, "Length", json_integer(ipv4->ip_len));
        json_object_set_new(p_object, "Length", json_integer(SCNtohs(ipv4->ip_len)));
        json_object_set_new(p_object, "TTL", json_integer(ipv4->ip_ttl));
        //json_object_set_new(p_object, "Checksum", json_integer(ipv4->ip_csum));
        json_object_set_new(p_object, "Checksum", json_integer(SCNtohs(ipv4->ip_csum)));
        json_object_set_new(p_object, "IP Version", json_integer(4));
        if (p->l4.hdrs.tcph != NULL) {
            TCPHdr* tcp = p->l4.hdrs.tcph;
            json_object_set_new(p_object, "Source Port", json_integer(SCNtohs(tcp->th_sport)));
            json_object_set_new(p_object, "Destination Port", json_integer(SCNtohs(tcp->th_dport)));
        }
        if (p->l4.hdrs.udph != NULL) {
            UDPHdr* udp = p->l4.hdrs.udph;
            json_object_set_new(p_object, "Source Port", json_integer(SCNtohs(udp->uh_sport)));
            json_object_set_new(p_object, "Destination Port", json_integer(SCNtohs(udp->uh_dport)));
        }
    }else if (PacketIsIPv6(p)) {
        const IPV6Hdr* ipv6 = PacketGetIPv6(p);
        PrintInet(AF_INET6, (const void*)&(p->src.addr_data32[0]), src_ip, sizeof(src_ip));
        PrintInet(AF_INET6, (const void*)&(p->dst.addr_data32[0]), dst_ip, sizeof(dst_ip));
        json_object_set_new(p_object, "Header Length", json_integer(40));
        json_object_set_new(p_object, "Source IP", json_string(src_ip));
        json_object_set_new(p_object, "Destination IP", json_string(dst_ip));
        json_object_set_new(p_object, "Next Header", json_integer(ipv6->ip6_hdrun.ip6_un1.ip6_un1_nxt));

        //json_object_set_new(p_object, "Traffic Class", json_integer(((ipv6->ip6_hdrun.ip6_un1.ip6_un1_flow) >> 20) & 0xFF));
        uint32_t flow = SCNtohl(ipv6->ip6_hdrun.ip6_un1.ip6_un1_flow);
        uint8_t tc = (flow >> 20) & 0xFF;
        json_object_set_new(p_object, "Traffic Class", json_integer(tc));
        //json_object_set_new(p_object, "Payload Length", json_integer(ipv6->ip6_hdrun.ip6_un1.ip6_un1_plen));
        json_object_set_new(p_object, "Payload Length",json_integer(SCNtohs(ipv6->ip6_hdrun.ip6_un1.ip6_un1_plen)));

        json_object_set_new(p_object, "Hop Limit", json_integer(ipv6->ip6_hdrun.ip6_un1.ip6_un1_hlim));
        json_object_set_new(p_object, "IP Version", json_integer(6));
        if (p->l4.hdrs.tcph != NULL) {
            TCPHdr* tcp = p->l4.hdrs.tcph;
            json_object_set_new(p_object, "Source Port", json_integer(SCNtohs(tcp->th_sport)));
            json_object_set_new(p_object, "Destination Port", json_integer(SCNtohs(tcp->th_dport)));
        }
        if (p->l4.hdrs.udph != NULL) {
            UDPHdr* udp = p->l4.hdrs.udph;
            json_object_set_new(p_object, "Source Port", json_integer(SCNtohs(udp->uh_sport)));
            json_object_set_new(p_object, "Destination Port", json_integer(SCNtohs(udp->uh_dport)));
        }
    }else {
        char proto_str[10];
        sprintf(proto_str, "%u", p->proto);
        json_object_set_new(p_object, "proto_name", json_string(proto_str));
        json_object_set_new(p_object, "decode_status", json_string("Non-IP"));
    }
    json_object_set_new(p_object, "pkt_len", json_integer(GET_PKT_LEN(p)));
    json_object_set_new(p_object, "payload_len", json_integer(p->payload_len));

    if (p->alerts.cnt > 0) {
        json_object_set_new(p_object, "label", json_integer(1));
    } else {
        json_object_set_new(p_object, "label", json_integer(0));
    }

    char *json_str = json_dumps(p_object, 0);
    if (json_str) {
        pthread_mutex_lock(&tdata_mutex);
        if (tdata->fp) {
            fprintf(tdata->fp, "%s\n", json_str);
            fflush(tdata->fp);
        }
        pthread_mutex_unlock(&tdata_mutex);
        free(json_str);
    }

    json_decref(p_object);
    return TM_ECODE_OK; // 返回标准的 TmEcode
}

static bool CustomPacketLoggerCondition(ThreadVars *tv, void *thread_data, const Packet *)
{
    (void)tv; (void)thread_data;
    return true;
}

static void Init(void)
{
    // 在 8.0 版本中，建议确保模块名称没有任何拼写差异
    SCOutputRegisterPacketLogger(LOGGER_USER, "custom-packet-logger",
                                 CustomPacketJsonLogger,
                                 CustomPacketLoggerCondition,
                                 NULL,
                                 ThreadInit,
                                 ThreadDeInit);
}

const SCPlugin PluginRegistration = {
        .version = SC_API_VERSION,
        .suricata_version = SC_PACKAGE_VERSION,
        .name = "custom-packet-logger",
        .plugin_version = "1.0.6",
        .author = "A",
        .license = "GPLv2",
        .Init = Init,
};

const SCPlugin *SCPluginRegister(void) {
    return &PluginRegistration;
}