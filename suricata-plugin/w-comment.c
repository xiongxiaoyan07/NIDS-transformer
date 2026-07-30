/*
 * custom_out_csv_annotated.c
 *
 * Purpose:
 *   This Suricata plugin writes two CSV files at the same time:
 *
 *   1) stage1_packets.csv
 *      - One row per packet.
 *      - Used to build Xi = {x_i,1, ..., x_i,T_i} for the Stage-I Packet Transformer.
 *
 *   2) stage1_flows.csv
 *      - One row per flow.
 *      - Used as s_i, the flow-level statistical feature vector for the MLP baseline.
 *
 * Main idea:
 *   - The Packet Logger is called once for every packet.
 *   - Every packet updates an in-memory flow aggregation table.
 *   - The Flow Logger is called when Suricata logs/expires a flow.
 *   - At that time, the aggregated flow features are written as one CSV row.
 *
 * Important research note:
 *   - Packet and flow features can be model inputs.
 *   - Alert/signature fields should not be used as model inputs because they leak Suricata rule logic.
 *   - Label and Attack are left as placeholders and should be filled later from EVE alert logs or ground truth.
 */

#include "suricata-common.h"      /* Core Suricata definitions, memory helpers, logging macros, common types. */
#include "suricata-plugin.h"      /* Plugin API definitions such as SCPlugin and SCPluginRegister. */

#include "output-packet.h"        /* Packet logging API, including SCOutputRegisterPacketLogger. */
#include "output-flow.h"          /* Flow logging API, including SCOutputRegisterFlowLogger. */

#include "decode.h"               /* Generic packet decode structures and macros such as Packet. */
#include "decode-ipv4.h"          /* IPv4 header structure and helper functions. */
#include "decode-ipv6.h"          /* IPv6 header structure and helper functions. */
#include "decode-tcp.h"           /* TCP header structure and helper macros. */
#include "decode-udp.h"           /* UDP header structure. */
#include "decode-icmpv4.h"        /* ICMPv4 header structure. */

#include "flow.h"                 /* Flow structure, flow identifiers, and flow counters. */
#include "util-print.h"           /* PrintInet helper for converting binary IP addresses to strings. */
#include "util-time.h"            /* Suricata time helper macros. */
#include "app-layer-protos.h"     /* Application-layer protocol identifiers, such as HTTP, DNS, TLS. */

#include <stdio.h>                 /* FILE, fopen, fprintf, fflush, fclose. */
#include <stdlib.h>                /* calloc, free. */
#include <stdint.h>                /* uint8_t, uint16_t, uint32_t, uint64_t. */
#include <stdbool.h>               /* bool, true, false. */
#include <string.h>                /* snprintf, string handling. */
#include <pthread.h>               /* pthread_mutex_t and mutex functions for thread safety. */
#include <sys/stat.h>              /* stat, used to check whether a CSV file already exists or is empty. */
#include <netinet/in.h>            /* IPPROTO_TCP, IPPROTO_UDP, IPPROTO_ICMP, AF_INET, AF_INET6. */

#define CUSTOM_PACKET_CSV_FILE "/home/xxiong/pcaps/stage1_packets.csv" /* Output CSV for packet-level sequence features. */
#define CUSTOM_FLOW_CSV_FILE   "/home/xxiong/pcaps/stage1_flows.csv"   /* Output CSV for flow-level baseline features. */

#define FLOW_AGG_BUCKETS 65536     /* Number of buckets in the simple hash table used to store flow aggregations. */

/*
 * Direction convention used by this plugin:
 *   direction = 0 means toserver / source-to-destination / IN direction.
 *   direction = 1 means toclient / destination-to-source / OUT direction.
 *
 * For NF-UQ-like flow features:
 *   IN_BYTES and IN_PKTS are direction 0.
 *   OUT_BYTES and OUT_PKTS are direction 1.
 */

typedef struct FlowAgg_ {          /* Per-flow aggregation structure maintained by this plugin. */
    uint64_t flow_id;              /* Suricata flow identifier used to join packets, flows, and EVE alerts. */

    bool tuple_seen;               /* Whether the 5-tuple has already been copied into this aggregation. */
    int ip_version;                /* IP version: 4 for IPv4, 6 for IPv6, 0 if unknown. */
    char src_ip[46];               /* Source IP string; 46 bytes are enough for IPv6 textual format. */
    char dst_ip[46];               /* Destination IP string. */
    uint16_t src_port;             /* Source transport-layer port. */
    uint16_t dst_port;             /* Destination transport-layer port. */
    uint8_t proto;                 /* L4 protocol number: TCP=6, UDP=17, ICMP=1. */

    uint64_t first_ts_us;          /* Timestamp of the first packet in this flow, in microseconds. */
    uint64_t last_ts_us;           /* Timestamp of the last packet seen in this flow, in microseconds. */

    uint64_t first_dir_ts_us[2];   /* First packet timestamp per direction: index 0=in, index 1=out. */
    uint64_t last_dir_ts_us[2];    /* Last packet timestamp per direction. */

    uint64_t bytes[2];             /* Byte counters per direction, using IP packet length. */
    uint64_t pkts[2];              /* Packet counters per direction. */

    uint8_t tcp_flags[2];          /* OR-aggregated TCP flags per direction. */
    uint8_t tcp_flags_total;       /* OR-aggregated TCP flags for the whole bidirectional flow. */

    uint16_t tcp_win_max[2];       /* Maximum TCP window value per direction. */

    uint32_t min_ttl;              /* Minimum observed IPv4 TTL or IPv6 hop limit. */
    uint32_t max_ttl;              /* Maximum observed IPv4 TTL or IPv6 hop limit. */

    uint32_t longest_flow_pkt;     /* Largest IP packet length observed in this flow. */
    uint32_t shortest_flow_pkt;    /* Smallest IP packet length observed in this flow. */

    uint32_t min_ip_pkt_len;       /* Minimum IP packet length. */
    uint32_t max_ip_pkt_len;       /* Maximum IP packet length. */

    uint64_t num_pkts_up_to_128;       /* Number of packets with IP length <= 128 bytes. */
    uint64_t num_pkts_128_to_256;      /* Number of packets with 128 < IP length <= 256 bytes. */
    uint64_t num_pkts_256_to_512;      /* Number of packets with 256 < IP length <= 512 bytes. */
    uint64_t num_pkts_512_to_1024;     /* Number of packets with 512 < IP length <= 1024 bytes. */
    uint64_t num_pkts_1024_to_1514;    /* Number of packets with 1024 < IP length <= 1514 bytes. */

    bool icmp_seen;                /* Whether this flow contains an ICMPv4 packet. */
    uint8_t icmp_type;             /* Last observed ICMPv4 type. */
    uint8_t icmp_code;             /* Last observed ICMPv4 code. */

    uint64_t retransmitted_bytes[2]; /* Placeholder for retransmitted bytes per direction. */
    uint64_t retransmitted_pkts[2];  /* Placeholder for retransmitted packets per direction. */

    struct FlowAgg_ *next;         /* Next item in the hash bucket linked list. */
} FlowAgg;                         /* End of FlowAgg definition. */

typedef struct {                   /* Per-thread logger data passed by Suricata into logger callbacks. */
    FILE *packet_fp;               /* File handle for stage1_packets.csv. */
    FILE *flow_fp;                 /* File handle for stage1_flows.csv. */
} CustomLoggerThreadData;          /* End of CustomLoggerThreadData definition. */

static FlowAgg *g_flow_aggs[FLOW_AGG_BUCKETS]; /* Global hash table storing all active flow aggregations. */

static pthread_mutex_t g_agg_mutex = PTHREAD_MUTEX_INITIALIZER;  /* Protects g_flow_aggs from concurrent access. */
static pthread_mutex_t g_file_mutex = PTHREAD_MUTEX_INITIALIZER; /* Protects CSV file writes from concurrent threads. */

static const char *PACKET_CSV_HEADER =                      /* CSV header for packet-level output. */
    "record_type,"                                          /* Constant string: packet. */
    "packet_id,"                                            /* Packet number from PCAP processing. */
    "timestamp_us,"                                         /* Packet timestamp in microseconds. */
    "ts_sec,"                                               /* Packet timestamp seconds part. */
    "ts_usec,"                                              /* Packet timestamp microseconds part. */
    "flow_id,"                                              /* Suricata flow id. */
    "direction,"                                            /* Packet direction: 0=toserver, 1=toclient, -1=unknown. */
    "ip_version,"                                           /* 4, 6, or 0. */
    "src_ip,"                                               /* Packet source IP. */
    "dst_ip,"                                               /* Packet destination IP. */
    "src_port,"                                             /* Packet source port. */
    "dst_port,"                                             /* Packet destination port. */
    "protocol,"                                             /* L4 protocol number. */
    "pkt_len,"                                              /* Full captured packet length. */
    "ip_len,"                                               /* IP-layer packet length. */
    "payload_len,"                                          /* Suricata decoded payload length. */
    "ttl_or_hop_limit,"                                     /* IPv4 TTL or IPv6 hop limit. */
    "tcp_seq,"                                              /* TCP sequence number; 0 for non-TCP. */
    "tcp_ack,"                                              /* TCP ACK number; 0 for non-TCP. */
    "tcp_flags,"                                            /* TCP flags integer; 0 for non-TCP. */
    "tcp_window,"                                           /* TCP window size; -1 for non-TCP. */
    "tcp_header_len,"                                       /* TCP header length; -1 for non-TCP. */
    "udp_len,"                                              /* UDP datagram length; -1 for non-UDP. */
    "icmp_type,"                                            /* ICMP type; -1 for non-ICMP. */
    "icmp_code,"                                            /* ICMP code; -1 for non-ICMP. */
    "l7_proto";                                             /* Suricata application-layer protocol id. */

static const char *FLOW_CSV_HEADER =                        /* CSV header for flow-level output. */
    "flow_id,"                                              /* Suricata flow id. */
    "IPV4_SRC_ADDR,"                                        /* Source IP address; name follows NF-UQ style. */
    "L4_SRC_PORT,"                                          /* Source L4 port. */
    "IPV4_DST_ADDR,"                                        /* Destination IP address. */
    "L4_DST_PORT,"                                          /* Destination L4 port. */
    "PROTOCOL,"                                             /* L4 protocol number. */
    "L7_PROTO,"                                             /* Suricata app-layer protocol id. */
    "IN_BYTES,"                                             /* Source-to-destination bytes. */
    "IN_PKTS,"                                              /* Source-to-destination packets. */
    "OUT_BYTES,"                                            /* Destination-to-source bytes. */
    "OUT_PKTS,"                                             /* Destination-to-source packets. */
    "TCP_FLAGS,"                                            /* OR of all TCP flags in the flow. */
    "CLIENT_TCP_FLAGS,"                                     /* OR of TCP flags in client/toserver direction. */
    "SERVER_TCP_FLAGS,"                                     /* OR of TCP flags in server/toclient direction. */
    "FLOW_DURATION_MILLISECONDS,"                           /* Whole flow duration in milliseconds. */
    "DURATION_IN,"                                          /* Source-to-destination duration in milliseconds. */
    "DURATION_OUT,"                                         /* Destination-to-source duration in milliseconds. */
    "MIN_TTL,"                                              /* Minimum TTL/hop-limit in the flow. */
    "MAX_TTL,"                                              /* Maximum TTL/hop-limit in the flow. */
    "LONGEST_FLOW_PKT,"                                     /* Maximum IP packet length. */
    "SHORTEST_FLOW_PKT,"                                    /* Minimum IP packet length. */
    "MIN_IP_PKT_LEN,"                                       /* Minimum IP packet length. */
    "MAX_IP_PKT_LEN,"                                       /* Maximum IP packet length. */
    "SRC_TO_DST_SECOND_BYTES,"                              /* Source-to-destination bytes per second. */
    "DST_TO_SRC_SECOND_BYTES,"                              /* Destination-to-source bytes per second. */
    "RETRANSMITTED_IN_BYTES,"                               /* Placeholder for source-to-destination retransmitted bytes. */
    "RETRANSMITTED_IN_PKTS,"                                /* Placeholder for source-to-destination retransmitted packets. */
    "RETRANSMITTED_OUT_BYTES,"                              /* Placeholder for destination-to-source retransmitted bytes. */
    "RETRANSMITTED_OUT_PKTS,"                               /* Placeholder for destination-to-source retransmitted packets. */
    "SRC_TO_DST_AVG_THROUGHPUT,"                            /* Source-to-destination average throughput in bits/s. */
    "DST_TO_SRC_AVG_THROUGHPUT,"                            /* Destination-to-source average throughput in bits/s. */
    "NUM_PKTS_UP_TO_128_BYTES,"                             /* Packet size bucket <=128. */
    "NUM_PKTS_128_TO_256_BYTES,"                            /* Packet size bucket 129-256. */
    "NUM_PKTS_256_TO_512_BYTES,"                            /* Packet size bucket 257-512. */
    "NUM_PKTS_512_TO_1024_BYTES,"                           /* Packet size bucket 513-1024. */
    "NUM_PKTS_1024_TO_1514_BYTES,"                          /* Packet size bucket 1025-1514. */
    "TCP_WIN_MAX_IN,"                                       /* Maximum TCP window in source-to-destination direction. */
    "TCP_WIN_MAX_OUT,"                                      /* Maximum TCP window in destination-to-source direction. */
    "ICMP_TYPE,"                                            /* ICMP type-code combination: type*256+code. */
    "ICMP_IPV4_TYPE,"                                       /* ICMP type only. */
    "DNS_QUERY_ID,"                                         /* Placeholder; merge from EVE DNS logs later. */
    "DNS_QUERY_TYPE,"                                       /* Placeholder; merge from EVE DNS logs later. */
    "DNS_TTL_ANSWER,"                                       /* Placeholder; merge from EVE DNS logs later. */
    "FTP_COMMAND_RET_CODE,"                                 /* Placeholder; merge from EVE FTP logs later. */
    "Label,"                                                /* Placeholder label; fill in preprocessing. */
    "Attack";                                               /* Placeholder attack class; fill in preprocessing. */

static inline uint64_t HashFlowId(uint64_t flow_id)          /* Computes hash bucket index for a flow id. */
{                                                             /* Start of HashFlowId. */
    return flow_id % FLOW_AGG_BUCKETS;                       /* Use modulo to map flow id to a bucket. */
}                                                             /* End of HashFlowId. */

static inline uint64_t PacketTsUsec(const Packet *p)          /* Converts packet timestamp to microseconds. */
{                                                             /* Start of PacketTsUsec. */
    return ((uint64_t)p->ts.secs * 1000000ULL) +              /* Convert seconds to microseconds. */
           (uint64_t)p->ts.usecs;                             /* Add microsecond part. */
}                                                             /* End of PacketTsUsec. */

static inline uint64_t DurationUsec(uint64_t first_us, uint64_t last_us) /* Computes non-negative duration in microseconds. */
{                                                             /* Start of DurationUsec. */
    if (first_us == 0 || last_us == 0 || last_us < first_us)  /* Check invalid or reversed timestamps. */
        return 0;                                             /* Return zero duration for invalid cases. */
    return last_us - first_us;                                /* Return elapsed microseconds. */
}                                                             /* End of DurationUsec. */

static inline double BytesPerSecond(uint64_t bytes, uint64_t duration_us) /* Computes bytes/second. */
{                                                             /* Start of BytesPerSecond. */
    if (duration_us == 0)                                     /* Avoid division by zero. */
        return 0.0;                                           /* No duration means zero rate. */
    return ((double)bytes * 1000000.0) / (double)duration_us; /* Convert microsecond duration to seconds. */
}                                                             /* End of BytesPerSecond. */

static inline double BitsPerSecond(uint64_t bytes, uint64_t duration_us) /* Computes bits/second. */
{                                                             /* Start of BitsPerSecond. */
    return BytesPerSecond(bytes, duration_us) * 8.0;          /* One byte equals eight bits. */
}                                                             /* End of BitsPerSecond. */

static FILE *OpenCsvFileWithHeader(const char *path, const char *header) /* Opens a CSV and writes header if needed. */
{                                                             /* Start of OpenCsvFileWithHeader. */
    FILE *fp = NULL;                                          /* File pointer to return. */
    struct stat st;                                           /* File metadata structure. */
    bool need_header = false;                                 /* Whether to write the CSV header. */

    pthread_mutex_lock(&g_file_mutex);                        /* Lock file operations to avoid races. */

    if (stat(path, &st) != 0 || st.st_size == 0)              /* If file does not exist or is empty. */
        need_header = true;                                   /* Mark that header should be written. */

    fp = fopen(path, "a");                                    /* Open file in append mode. */
    if (fp != NULL && need_header) {                          /* If opened and header is needed. */
        fprintf(fp, "%s\n", header);                         /* Write one header line. */
        fflush(fp);                                           /* Flush header to disk immediately. */
    }                                                         /* End header writing block. */

    pthread_mutex_unlock(&g_file_mutex);                      /* Unlock file operations. */

    return fp;                                                /* Return opened file pointer or NULL. */
}                                                             /* End of OpenCsvFileWithHeader. */

static FlowAgg *FlowAggLookupLocked(uint64_t flow_id)          /* Finds a flow aggregation; caller must hold g_agg_mutex. */
{                                                             /* Start of FlowAggLookupLocked. */
    uint64_t bucket = HashFlowId(flow_id);                    /* Compute hash bucket. */
    FlowAgg *cur = g_flow_aggs[bucket];                       /* Start with first item in bucket. */

    while (cur != NULL) {                                     /* Walk linked list. */
        if (cur->flow_id == flow_id)                          /* Check whether this item matches the flow id. */
            return cur;                                       /* Return matching aggregation. */
        cur = cur->next;                                      /* Move to next item. */
    }                                                         /* End loop. */

    return NULL;                                              /* Not found. */
}                                                             /* End of FlowAggLookupLocked. */

static FlowAgg *FlowAggGetOrCreateLocked(uint64_t flow_id)    /* Finds or creates a flow aggregation; caller must hold lock. */
{                                                             /* Start of FlowAggGetOrCreateLocked. */
    FlowAgg *agg = FlowAggLookupLocked(flow_id);              /* Try to find existing aggregation. */
    if (agg != NULL)                                          /* If it exists. */
        return agg;                                           /* Return it. */

    agg = calloc(1, sizeof(*agg));                            /* Allocate and zero-initialize new aggregation. */
    if (agg == NULL)                                          /* If allocation failed. */
        return NULL;                                          /* Return NULL to signal failure. */

    agg->flow_id = flow_id;                                   /* Store the flow id. */
    agg->min_ttl = UINT32_MAX;                                /* Initialize minimum TTL to very large value. */
    agg->shortest_flow_pkt = UINT32_MAX;                      /* Initialize shortest packet length to very large value. */
    agg->min_ip_pkt_len = UINT32_MAX;                         /* Initialize minimum IP packet length to very large value. */

    uint64_t bucket = HashFlowId(flow_id);                    /* Compute target hash bucket. */
    agg->next = g_flow_aggs[bucket];                          /* Insert new item at head of bucket linked list. */
    g_flow_aggs[bucket] = agg;                                /* Update bucket head. */

    return agg;                                               /* Return new aggregation. */
}                                                             /* End of FlowAggGetOrCreateLocked. */

static void FlowAggRemoveLocked(uint64_t flow_id)             /* Removes and frees an aggregation; caller must hold lock. */
{                                                             /* Start of FlowAggRemoveLocked. */
    uint64_t bucket = HashFlowId(flow_id);                    /* Compute hash bucket. */
    FlowAgg *cur = g_flow_aggs[bucket];                       /* Current linked-list item. */
    FlowAgg *prev = NULL;                                     /* Previous linked-list item. */

    while (cur != NULL) {                                     /* Walk linked list. */
        if (cur->flow_id == flow_id) {                        /* If this is the item to remove. */
            if (prev == NULL)                                 /* If removing bucket head. */
                g_flow_aggs[bucket] = cur->next;              /* Replace bucket head. */
            else                                              /* Otherwise. */
                prev->next = cur->next;                       /* Skip current item. */

            free(cur);                                        /* Free memory. */
            return;                                           /* Done. */
        }                                                     /* End match block. */

        prev = cur;                                           /* Advance previous pointer. */
        cur = cur->next;                                      /* Advance current pointer. */
    }                                                         /* End loop. */
}                                                             /* End of FlowAggRemoveLocked. */

static int GetPacketDirection(const Packet *p)                /* Returns packet direction as 0, 1, or -1. */
{                                                             /* Start of GetPacketDirection. */
#ifdef PKT_IS_TOSERVER                                       /* Use macro if available in this Suricata version. */
    if (PKT_IS_TOSERVER(p))                                  /* Check whether packet goes to server. */
        return 0;                                             /* Direction 0 means source-to-destination. */
#endif                                                        /* End conditional compilation block. */

#ifdef PKT_IS_TOCLIENT                                       /* Use macro if available in this Suricata version. */
    if (PKT_IS_TOCLIENT(p))                                  /* Check whether packet goes to client. */
        return 1;                                             /* Direction 1 means destination-to-source. */
#endif                                                        /* End conditional compilation block. */

#ifdef FLOW_PKT_TOSERVER                                     /* Fallback using flow flags if macro exists. */
    if (p->flowflags & FLOW_PKT_TOSERVER)                    /* Check toserver flag. */
        return 0;                                             /* Direction 0. */
#endif                                                        /* End conditional compilation block. */

#ifdef FLOW_PKT_TOCLIENT                                     /* Fallback using flow flags if macro exists. */
    if (p->flowflags & FLOW_PKT_TOCLIENT)                    /* Check toclient flag. */
        return 1;                                             /* Direction 1. */
#endif                                                        /* End conditional compilation block. */

    return -1;                                                /* Direction unknown. */
}                                                             /* End of GetPacketDirection. */

static void GetPacketIPs(const Packet *p, int *ip_version,    /* Extracts source/destination IP strings from packet. */
                         char *src_ip, size_t src_len,        /* Output buffer for source IP. */
                         char *dst_ip, size_t dst_len)        /* Output buffer for destination IP. */
{                                                             /* Start of GetPacketIPs. */
    *ip_version = 0;                                          /* Default IP version is unknown. */
    src_ip[0] = '\0';                                         /* Initialize source IP string as empty. */
    dst_ip[0] = '\0';                                         /* Initialize destination IP string as empty. */

    if (PacketIsIPv4(p)) {                                    /* If packet is IPv4. */
        *ip_version = 4;                                      /* Set IP version to 4. */
        PrintInet(AF_INET, &p->src.addr_data32[0], src_ip, src_len); /* Convert source IPv4 to string. */
        PrintInet(AF_INET, &p->dst.addr_data32[0], dst_ip, dst_len); /* Convert destination IPv4 to string. */
    } else if (PacketIsIPv6(p)) {                             /* If packet is IPv6. */
        *ip_version = 6;                                      /* Set IP version to 6. */
        PrintInet(AF_INET6, &p->src.addr_data32[0], src_ip, src_len); /* Convert source IPv6 to string. */
        PrintInet(AF_INET6, &p->dst.addr_data32[0], dst_ip, dst_len); /* Convert destination IPv6 to string. */
    }                                                         /* End IP version branch. */
}                                                             /* End of GetPacketIPs. */

static void GetPacketPorts(const Packet *p, uint16_t *src_port, uint16_t *dst_port) /* Extracts TCP/UDP ports. */
{                                                             /* Start of GetPacketPorts. */
    *src_port = 0;                                            /* Default source port is 0. */
    *dst_port = 0;                                            /* Default destination port is 0. */

    if (p->proto == IPPROTO_TCP && p->l4.hdrs.tcph != NULL) { /* If packet is TCP and TCP header exists. */
        const TCPHdr *tcp = p->l4.hdrs.tcph;                  /* Get TCP header pointer. */
        *src_port = SCNtohs(tcp->th_sport);                   /* Convert TCP source port from network to host order. */
        *dst_port = SCNtohs(tcp->th_dport);                   /* Convert TCP destination port from network to host order. */
    } else if (p->proto == IPPROTO_UDP && p->l4.hdrs.udph != NULL) { /* If packet is UDP and UDP header exists. */
        const UDPHdr *udp = p->l4.hdrs.udph;                  /* Get UDP header pointer. */
        *src_port = SCNtohs(udp->uh_sport);                   /* Convert UDP source port from network to host order. */
        *dst_port = SCNtohs(udp->uh_dport);                   /* Convert UDP destination port from network to host order. */
    }                                                         /* End protocol branch. */
}                                                             /* End of GetPacketPorts. */

static uint32_t GetIpPacketLen(const Packet *p)               /* Returns IP-layer packet length. */
{                                                             /* Start of GetIpPacketLen. */
    if (PacketIsIPv4(p)) {                                    /* If packet is IPv4. */
        const IPV4Hdr *ipv4 = PacketGetIPv4(p);               /* Get IPv4 header pointer. */
        if (ipv4 != NULL)                                     /* If header exists. */
            return (uint32_t)SCNtohs(ipv4->ip_len);           /* Return IPv4 total length. */
    }                                                         /* End IPv4 branch. */

    if (PacketIsIPv6(p)) {                                    /* If packet is IPv6. */
        const IPV6Hdr *ipv6 = PacketGetIPv6(p);               /* Get IPv6 header pointer. */
        if (ipv6 != NULL) {                                   /* If header exists. */
            uint32_t payload_len =                            /* IPv6 header stores payload length, not total length. */
                (uint32_t)SCNtohs(ipv6->ip6_hdrun.ip6_un1.ip6_un1_plen); /* Convert IPv6 payload length. */
            return 40U + payload_len;                         /* IPv6 base header is 40 bytes, so total is 40 + payload. */
        }                                                     /* End IPv6 header block. */
    }                                                         /* End IPv6 branch. */

    return (uint32_t)GET_PKT_LEN(p);                          /* Fallback to captured packet length. */
}                                                             /* End of GetIpPacketLen. */

static int GetIpTTLOrHopLimit(const Packet *p)                /* Returns IPv4 TTL or IPv6 hop limit. */
{                                                             /* Start of GetIpTTLOrHopLimit. */
    if (PacketIsIPv4(p)) {                                    /* If packet is IPv4. */
        const IPV4Hdr *ipv4 = PacketGetIPv4(p);               /* Get IPv4 header. */
        if (ipv4 != NULL)                                     /* If header exists. */
            return (int)ipv4->ip_ttl;                         /* Return TTL. */
    }                                                         /* End IPv4 branch. */

    if (PacketIsIPv6(p)) {                                    /* If packet is IPv6. */
        const IPV6Hdr *ipv6 = PacketGetIPv6(p);               /* Get IPv6 header. */
        if (ipv6 != NULL)                                     /* If header exists. */
            return (int)ipv6->ip6_hdrun.ip6_un1.ip6_un1_hlim; /* Return hop limit. */
    }                                                         /* End IPv6 branch. */

    return -1;                                                /* Unknown or non-IP packet. */
}                                                             /* End of GetIpTTLOrHopLimit. */

static void UpdatePacketLengthBins(FlowAgg *agg, uint32_t ip_len) /* Updates packet-size histogram buckets. */
{                                                             /* Start of UpdatePacketLengthBins. */
    if (ip_len <= 128) {                                      /* Packet length bucket 1. */
        agg->num_pkts_up_to_128++;                           /* Increment <=128 counter. */
    } else if (ip_len <= 256) {                               /* Packet length bucket 2. */
        agg->num_pkts_128_to_256++;                          /* Increment 129-256 counter. */
    } else if (ip_len <= 512) {                               /* Packet length bucket 3. */
        agg->num_pkts_256_to_512++;                          /* Increment 257-512 counter. */
    } else if (ip_len <= 1024) {                              /* Packet length bucket 4. */
        agg->num_pkts_512_to_1024++;                         /* Increment 513-1024 counter. */
    } else if (ip_len <= 1514) {                              /* Packet length bucket 5. */
        agg->num_pkts_1024_to_1514++;                        /* Increment 1025-1514 counter. */
    }                                                         /* Packets larger than 1514 are not counted in these buckets. */
}                                                             /* End of UpdatePacketLengthBins. */

static void UpdateFlowAggFromPacket(const Packet *p)          /* Updates flow-level aggregation using one packet. */
{                                                             /* Start of UpdateFlowAggFromPacket. */
    if (p == NULL || p->flow == NULL)                         /* If packet or flow pointer is missing. */
        return;                                               /* Cannot aggregate this packet. */

    uint64_t flow_id = FlowGetId(p->flow);                    /* Get Suricata flow id. */
    uint64_t ts_us = PacketTsUsec(p);                         /* Get packet timestamp in microseconds. */
    int direction = GetPacketDirection(p);                    /* Get packet direction. */

    if (direction != 0 && direction != 1)                     /* If direction is unknown. */
        direction = 0;                                        /* Put it into direction 0 as fallback. */

    uint32_t ip_len = GetIpPacketLen(p);                      /* Get IP packet length. */
    int ttl = GetIpTTLOrHopLimit(p);                          /* Get TTL or hop limit. */

    pthread_mutex_lock(&g_agg_mutex);                         /* Lock aggregation table. */

    FlowAgg *agg = FlowAggGetOrCreateLocked(flow_id);         /* Get existing aggregation or create a new one. */
    if (agg == NULL) {                                        /* If allocation failed. */
        pthread_mutex_unlock(&g_agg_mutex);                   /* Unlock aggregation table. */
        return;                                               /* Stop processing this packet. */
    }                                                         /* End allocation failure branch. */

    if (!agg->tuple_seen) {                                   /* If this is the first packet with tuple information. */
        int ip_version = 0;                                   /* Temporary IP version. */
        char src_ip[46] = {0};                                /* Temporary source IP string. */
        char dst_ip[46] = {0};                                /* Temporary destination IP string. */
        uint16_t src_port = 0;                                /* Temporary source port. */
        uint16_t dst_port = 0;                                /* Temporary destination port. */

        GetPacketIPs(p, &ip_version, src_ip, sizeof(src_ip), dst_ip, sizeof(dst_ip)); /* Extract IPs. */
        GetPacketPorts(p, &src_port, &dst_port);              /* Extract ports. */

        agg->tuple_seen = true;                               /* Mark tuple as stored. */
        agg->ip_version = ip_version;                         /* Store IP version. */
        snprintf(agg->src_ip, sizeof(agg->src_ip), "%s", src_ip); /* Store source IP safely. */
        snprintf(agg->dst_ip, sizeof(agg->dst_ip), "%s", dst_ip); /* Store destination IP safely. */
        agg->src_port = src_port;                             /* Store source port. */
        agg->dst_port = dst_port;                             /* Store destination port. */
        agg->proto = p->proto;                                /* Store protocol number. */
    }                                                         /* End tuple initialization block. */

    if (agg->first_ts_us == 0 || ts_us < agg->first_ts_us)    /* If this packet is earlier than current first timestamp. */
        agg->first_ts_us = ts_us;                             /* Update first timestamp. */

    if (ts_us > agg->last_ts_us)                              /* If this packet is later than current last timestamp. */
        agg->last_ts_us = ts_us;                              /* Update last timestamp. */

    if (agg->first_dir_ts_us[direction] == 0 || ts_us < agg->first_dir_ts_us[direction]) /* If first timestamp for this direction is empty/later. */
        agg->first_dir_ts_us[direction] = ts_us;              /* Update first timestamp for this direction. */

    if (ts_us > agg->last_dir_ts_us[direction])               /* If this is latest timestamp in this direction. */
        agg->last_dir_ts_us[direction] = ts_us;               /* Update last timestamp for this direction. */

    agg->pkts[direction]++;                                   /* Increment packet counter for direction. */
    agg->bytes[direction] += ip_len;                          /* Add IP packet length to byte counter. */

    if (ttl >= 0) {                                           /* If TTL/hop-limit is valid. */
        if ((uint32_t)ttl < agg->min_ttl)                     /* If smaller than current minimum. */
            agg->min_ttl = (uint32_t)ttl;                     /* Update minimum TTL. */
        if ((uint32_t)ttl > agg->max_ttl)                     /* If larger than current maximum. */
            agg->max_ttl = (uint32_t)ttl;                     /* Update maximum TTL. */
    }                                                         /* End TTL update block. */

    if (ip_len > agg->longest_flow_pkt)                       /* If this packet is the largest so far. */
        agg->longest_flow_pkt = ip_len;                       /* Update largest packet length. */

    if (ip_len < agg->shortest_flow_pkt)                      /* If this packet is the smallest so far. */
        agg->shortest_flow_pkt = ip_len;                      /* Update smallest packet length. */

    if (ip_len < agg->min_ip_pkt_len)                         /* If this packet length is smaller than min IP length. */
        agg->min_ip_pkt_len = ip_len;                         /* Update min IP packet length. */

    if (ip_len > agg->max_ip_pkt_len)                         /* If this packet length is larger than max IP length. */
        agg->max_ip_pkt_len = ip_len;                         /* Update max IP packet length. */

    UpdatePacketLengthBins(agg, ip_len);                      /* Update packet-size histogram buckets. */

    if (p->proto == IPPROTO_TCP && p->l4.hdrs.tcph != NULL) { /* If packet is TCP. */
        const TCPHdr *tcp = p->l4.hdrs.tcph;                  /* Get TCP header. */
        uint8_t flags = tcp->th_flags;                        /* Read TCP flags. */
        uint16_t win = SCNtohs(tcp->th_win);                  /* Read TCP window in host byte order. */

        agg->tcp_flags[direction] |= flags;                   /* OR flags into per-direction flag set. */
        agg->tcp_flags_total |= flags;                        /* OR flags into whole-flow flag set. */

        if (win > agg->tcp_win_max[direction])                /* If this window is larger than current maximum. */
            agg->tcp_win_max[direction] = win;                /* Update max window for direction. */
    }                                                         /* End TCP update block. */

    if (p->proto == IPPROTO_ICMP && p->l4.hdrs.icmpv4h != NULL) { /* If packet is ICMPv4. */
        const ICMPV4Hdr *icmp4 = p->l4.hdrs.icmpv4h;          /* Get ICMPv4 header. */
        agg->icmp_seen = true;                                /* Mark ICMP as observed. */
        agg->icmp_type = icmp4->type;                         /* Store ICMP type. */
        agg->icmp_code = icmp4->code;                         /* Store ICMP code. */
    }                                                         /* End ICMP update block. */

    pthread_mutex_unlock(&g_agg_mutex);                       /* Unlock aggregation table. */
}                                                             /* End of UpdateFlowAggFromPacket. */

static void WritePacketCsvLine(FILE *fp, const Packet *p)     /* Writes one packet row to stage1_packets.csv. */
{                                                             /* Start of WritePacketCsvLine. */
    if (fp == NULL || p == NULL)                              /* Check invalid input. */
        return;                                               /* Nothing to write. */

    uint64_t ts_us = PacketTsUsec(p);                         /* Packet timestamp in microseconds. */

    uint64_t flow_id = 0;                                     /* Default flow id is 0 if unavailable. */
    if (p->flow != NULL)                                      /* If packet belongs to a Suricata flow. */
        flow_id = FlowGetId(p->flow);                         /* Store the flow id. */

    int direction = GetPacketDirection(p);                    /* Packet direction. */

    int ip_version = 0;                                       /* IP version placeholder. */
    char src_ip[46] = {0};                                    /* Source IP string. */
    char dst_ip[46] = {0};                                    /* Destination IP string. */
    GetPacketIPs(p, &ip_version, src_ip, sizeof(src_ip), dst_ip, sizeof(dst_ip)); /* Extract IP address fields. */

    uint16_t src_port = 0;                                    /* Source port placeholder. */
    uint16_t dst_port = 0;                                    /* Destination port placeholder. */
    GetPacketPorts(p, &src_port, &dst_port);                  /* Extract source and destination ports. */

    uint32_t pkt_len = (uint32_t)GET_PKT_LEN(p);              /* Captured packet length. */
    uint32_t ip_len = GetIpPacketLen(p);                      /* IP-layer packet length. */
    int ttl = GetIpTTLOrHopLimit(p);                          /* TTL or hop limit. */

    uint32_t tcp_seq = 0;                                     /* TCP sequence number, default 0. */
    uint32_t tcp_ack = 0;                                     /* TCP acknowledgement number, default 0. */
    int tcp_flags = 0;                                        /* TCP flags, default 0. */
    int tcp_window = -1;                                      /* TCP window, default -1 for non-TCP. */
    int tcp_header_len = -1;                                  /* TCP header length, default -1 for non-TCP. */

    int udp_len = -1;                                         /* UDP length, default -1 for non-UDP. */

    int icmp_type = -1;                                       /* ICMP type, default -1 for non-ICMP. */
    int icmp_code = -1;                                       /* ICMP code, default -1 for non-ICMP. */

    int l7_proto = 0;                                         /* Suricata app-layer protocol id, default 0. */

    if (p->flow != NULL)                                      /* If flow exists. */
        l7_proto = p->flow->alproto;                          /* Copy app-layer protocol id from flow. */

    if (p->proto == IPPROTO_TCP && p->l4.hdrs.tcph != NULL) { /* If packet is TCP. */
        const TCPHdr *tcp = p->l4.hdrs.tcph;                  /* Get TCP header pointer. */
        tcp_seq = SCNtohl(tcp->th_seq);                       /* Convert TCP sequence number to host byte order. */
        tcp_ack = SCNtohl(tcp->th_ack);                       /* Convert TCP ACK number to host byte order. */
        tcp_flags = tcp->th_flags;                            /* Read TCP flags. */
        tcp_window = SCNtohs(tcp->th_win);                    /* Convert TCP window to host byte order. */
        tcp_header_len = TCP_GET_HLEN(p);                     /* Get TCP header length. */
    }                                                         /* End TCP block. */

    if (p->proto == IPPROTO_UDP && p->l4.hdrs.udph != NULL) { /* If packet is UDP. */
        const UDPHdr *udp = p->l4.hdrs.udph;                  /* Get UDP header pointer. */
        udp_len = SCNtohs(udp->uh_len);                       /* Convert UDP length to host byte order. */
    }                                                         /* End UDP block. */

    if (p->proto == IPPROTO_ICMP && p->l4.hdrs.icmpv4h != NULL) { /* If packet is ICMPv4. */
        const ICMPV4Hdr *icmp4 = p->l4.hdrs.icmpv4h;          /* Get ICMPv4 header pointer. */
        icmp_type = icmp4->type;                              /* Read ICMP type. */
        icmp_code = icmp4->code;                              /* Read ICMP code. */
    }                                                         /* End ICMP block. */

    pthread_mutex_lock(&g_file_mutex);                        /* Lock file writing. */

    fprintf(fp,                                                /* Write one CSV row. */
            "packet,"                                         /* record_type. */
            "%lu,"                                           /* packet_id. */
            "%lu,"                                           /* timestamp_us. */
            "%lu,"                                           /* ts_sec. */
            "%lu,"                                           /* ts_usec. */
            "%lu,"                                           /* flow_id. */
            "%d,"                                            /* direction. */
            "%d,"                                            /* ip_version. */
            "%s,"                                            /* src_ip. */
            "%s,"                                            /* dst_ip. */
            "%u,"                                            /* src_port. */
            "%u,"                                            /* dst_port. */
            "%u,"                                            /* protocol. */
            "%u,"                                            /* pkt_len. */
            "%u,"                                            /* ip_len. */
            "%u,"                                            /* payload_len. */
            "%d,"                                            /* ttl_or_hop_limit. */
            "%u,"                                            /* tcp_seq. */
            "%u,"                                            /* tcp_ack. */
            "%d,"                                            /* tcp_flags. */
            "%d,"                                            /* tcp_window. */
            "%d,"                                            /* tcp_header_len. */
            "%d,"                                            /* udp_len. */
            "%d,"                                            /* icmp_type. */
            "%d,"                                            /* icmp_code. */
            "%d\n",                                          /* l7_proto. */
            (unsigned long)p->pcap_cnt,                       /* packet_id value. */
            (unsigned long)ts_us,                             /* timestamp_us value. */
            (unsigned long)p->ts.secs,                        /* ts_sec value. */
            (unsigned long)p->ts.usecs,                       /* ts_usec value. */
            (unsigned long)flow_id,                           /* flow_id value. */
            direction,                                        /* direction value. */
            ip_version,                                       /* ip_version value. */
            src_ip,                                           /* src_ip value. */
            dst_ip,                                           /* dst_ip value. */
            src_port,                                         /* src_port value. */
            dst_port,                                         /* dst_port value. */
            p->proto,                                         /* protocol value. */
            pkt_len,                                          /* pkt_len value. */
            ip_len,                                           /* ip_len value. */
            (uint32_t)p->payload_len,                         /* payload_len value. */
            ttl,                                              /* ttl_or_hop_limit value. */
            tcp_seq,                                          /* tcp_seq value. */
            tcp_ack,                                          /* tcp_ack value. */
            tcp_flags,                                        /* tcp_flags value. */
            tcp_window,                                       /* tcp_window value. */
            tcp_header_len,                                   /* tcp_header_len value. */
            udp_len,                                          /* udp_len value. */
            icmp_type,                                        /* icmp_type value. */
            icmp_code,                                        /* icmp_code value. */
            l7_proto);                                        /* l7_proto value. */

    fflush(fp);                                               /* Force row to disk. */

    pthread_mutex_unlock(&g_file_mutex);                      /* Unlock file writing. */
}                                                             /* End of WritePacketCsvLine. */

static void GetFlowTuple(Flow *f,                             /* Extracts flow tuple in consistent direction. */
                         char *src_ip, size_t src_len,        /* Output buffer for source IP. */
                         char *dst_ip, size_t dst_len,        /* Output buffer for destination IP. */
                         uint16_t *src_port,                  /* Output source port. */
                         uint16_t *dst_port)                  /* Output destination port. */
{                                                             /* Start of GetFlowTuple. */
    src_ip[0] = '\0';                                         /* Initialize source IP. */
    dst_ip[0] = '\0';                                         /* Initialize destination IP. */
    *src_port = 0;                                            /* Initialize source port. */
    *dst_port = 0;                                            /* Initialize destination port. */

    if ((f->flags & FLOW_DIR_REVERSED) == 0) {                /* If Suricata flow direction is not reversed. */
        if (FLOW_IS_IPV4(f)) {                                /* If flow is IPv4. */
            PrintInet(AF_INET, (const void *)&(f->src.addr_data32[0]), src_ip, src_len); /* Convert source IPv4. */
            PrintInet(AF_INET, (const void *)&(f->dst.addr_data32[0]), dst_ip, dst_len); /* Convert destination IPv4. */
        } else if (FLOW_IS_IPV6(f)) {                         /* If flow is IPv6. */
            PrintInet(AF_INET6, (const void *)&(f->src.address), src_ip, src_len); /* Convert source IPv6. */
            PrintInet(AF_INET6, (const void *)&(f->dst.address), dst_ip, dst_len); /* Convert destination IPv6. */
        }                                                     /* End IP version block. */

        *src_port = f->sp;                                    /* Use Suricata source port. */
        *dst_port = f->dp;                                    /* Use Suricata destination port. */
    } else {                                                  /* If flow direction is reversed. */
        if (FLOW_IS_IPV4(f)) {                                /* If flow is IPv4. */
            PrintInet(AF_INET, (const void *)&(f->dst.addr_data32[0]), src_ip, src_len); /* Reverse destination as source. */
            PrintInet(AF_INET, (const void *)&(f->src.addr_data32[0]), dst_ip, dst_len); /* Reverse source as destination. */
        } else if (FLOW_IS_IPV6(f)) {                         /* If flow is IPv6. */
            PrintInet(AF_INET6, (const void *)&(f->dst.address), src_ip, src_len); /* Reverse destination IPv6 as source. */
            PrintInet(AF_INET6, (const void *)&(f->src.address), dst_ip, dst_len); /* Reverse source IPv6 as destination. */
        }                                                     /* End IP version block. */

        *src_port = f->dp;                                    /* Reverse destination port as source port. */
        *dst_port = f->sp;                                    /* Reverse source port as destination port. */
    }                                                         /* End direction branch. */
}                                                             /* End of GetFlowTuple. */

static void WriteFlowCsvLine(FILE *fp, Flow *f, const FlowAgg *agg) /* Writes one flow row to stage1_flows.csv. */
{                                                             /* Start of WriteFlowCsvLine. */
    if (fp == NULL || f == NULL)                              /* Check invalid input. */
        return;                                               /* Nothing to write. */

    uint64_t flow_id = FlowGetId(f);                          /* Get Suricata flow id. */

    char src_ip[46] = {0};                                    /* Source IP string. */
    char dst_ip[46] = {0};                                    /* Destination IP string. */
    uint16_t src_port = 0;                                    /* Source port. */
    uint16_t dst_port = 0;                                    /* Destination port. */

    GetFlowTuple(f, src_ip, sizeof(src_ip), dst_ip, sizeof(dst_ip), &src_port, &dst_port); /* Extract flow tuple. */

    uint64_t in_bytes = 0;                                    /* Source-to-destination bytes. */
    uint64_t out_bytes = 0;                                   /* Destination-to-source bytes. */
    uint64_t in_pkts = 0;                                     /* Source-to-destination packets. */
    uint64_t out_pkts = 0;                                    /* Destination-to-source packets. */

    uint8_t tcp_flags_total = 0;                              /* Whole-flow TCP flags. */
    uint8_t client_tcp_flags = 0;                             /* Client/toserver TCP flags. */
    uint8_t server_tcp_flags = 0;                             /* Server/toclient TCP flags. */

    uint64_t flow_duration_us = 0;                            /* Whole-flow duration in microseconds. */
    uint64_t duration_in_us = 0;                              /* Source-to-destination duration. */
    uint64_t duration_out_us = 0;                             /* Destination-to-source duration. */

    uint32_t min_ttl = 0;                                     /* Minimum TTL/hop-limit. */
    uint32_t max_ttl = 0;                                     /* Maximum TTL/hop-limit. */

    uint32_t longest_flow_pkt = 0;                            /* Maximum packet length. */
    uint32_t shortest_flow_pkt = 0;                           /* Minimum packet length. */
    uint32_t min_ip_pkt_len = 0;                              /* Minimum IP packet length. */
    uint32_t max_ip_pkt_len = 0;                              /* Maximum IP packet length. */

    uint64_t retransmitted_in_bytes = 0;                      /* Placeholder retransmitted bytes in direction 0. */
    uint64_t retransmitted_in_pkts = 0;                       /* Placeholder retransmitted packets in direction 0. */
    uint64_t retransmitted_out_bytes = 0;                     /* Placeholder retransmitted bytes in direction 1. */
    uint64_t retransmitted_out_pkts = 0;                      /* Placeholder retransmitted packets in direction 1. */

    uint64_t num_pkts_up_to_128 = 0;                          /* Packet size bucket counter. */
    uint64_t num_pkts_128_to_256 = 0;                         /* Packet size bucket counter. */
    uint64_t num_pkts_256_to_512 = 0;                         /* Packet size bucket counter. */
    uint64_t num_pkts_512_to_1024 = 0;                        /* Packet size bucket counter. */
    uint64_t num_pkts_1024_to_1514 = 0;                       /* Packet size bucket counter. */

    uint16_t tcp_win_max_in = 0;                              /* Maximum TCP window direction 0. */
    uint16_t tcp_win_max_out = 0;                             /* Maximum TCP window direction 1. */

    int icmp_type = 0;                                        /* ICMP type-code combined value. */
    int icmp_ipv4_type = 0;                                   /* ICMP type only. */

    if (agg != NULL) {                                        /* If packet aggregation exists. */
        in_bytes = agg->bytes[0];                             /* Copy source-to-destination bytes. */
        out_bytes = agg->bytes[1];                            /* Copy destination-to-source bytes. */
        in_pkts = agg->pkts[0];                               /* Copy source-to-destination packets. */
        out_pkts = agg->pkts[1];                              /* Copy destination-to-source packets. */

        tcp_flags_total = agg->tcp_flags_total;               /* Copy whole-flow TCP flags. */
        client_tcp_flags = agg->tcp_flags[0];                 /* Copy client/toserver TCP flags. */
        server_tcp_flags = agg->tcp_flags[1];                 /* Copy server/toclient TCP flags. */

        flow_duration_us = DurationUsec(agg->first_ts_us, agg->last_ts_us); /* Compute whole-flow duration. */
        duration_in_us = DurationUsec(agg->first_dir_ts_us[0], agg->last_dir_ts_us[0]); /* Compute in-direction duration. */
        duration_out_us = DurationUsec(agg->first_dir_ts_us[1], agg->last_dir_ts_us[1]); /* Compute out-direction duration. */

        if (agg->min_ttl != UINT32_MAX)                       /* If minimum TTL was updated. */
            min_ttl = agg->min_ttl;                           /* Copy minimum TTL. */
        max_ttl = agg->max_ttl;                               /* Copy maximum TTL. */

        longest_flow_pkt = agg->longest_flow_pkt;             /* Copy longest packet length. */

        if (agg->shortest_flow_pkt != UINT32_MAX)             /* If shortest packet length was updated. */
            shortest_flow_pkt = agg->shortest_flow_pkt;       /* Copy shortest packet length. */

        if (agg->min_ip_pkt_len != UINT32_MAX)                /* If minimum IP length was updated. */
            min_ip_pkt_len = agg->min_ip_pkt_len;             /* Copy minimum IP length. */

        max_ip_pkt_len = agg->max_ip_pkt_len;                 /* Copy maximum IP length. */

        retransmitted_in_bytes = agg->retransmitted_bytes[0]; /* Copy retransmission placeholder. */
        retransmitted_in_pkts = agg->retransmitted_pkts[0];   /* Copy retransmission placeholder. */
        retransmitted_out_bytes = agg->retransmitted_bytes[1]; /* Copy retransmission placeholder. */
        retransmitted_out_pkts = agg->retransmitted_pkts[1];  /* Copy retransmission placeholder. */

        num_pkts_up_to_128 = agg->num_pkts_up_to_128;         /* Copy packet length bucket counter. */
        num_pkts_128_to_256 = agg->num_pkts_128_to_256;       /* Copy packet length bucket counter. */
        num_pkts_256_to_512 = agg->num_pkts_256_to_512;       /* Copy packet length bucket counter. */
        num_pkts_512_to_1024 = agg->num_pkts_512_to_1024;     /* Copy packet length bucket counter. */
        num_pkts_1024_to_1514 = agg->num_pkts_1024_to_1514;   /* Copy packet length bucket counter. */

        tcp_win_max_in = agg->tcp_win_max[0];                 /* Copy max TCP window in direction 0. */
        tcp_win_max_out = agg->tcp_win_max[1];                /* Copy max TCP window in direction 1. */

        if (agg->icmp_seen) {                                 /* If ICMP packet was observed. */
            icmp_type = ((int)agg->icmp_type * 256) + (int)agg->icmp_code; /* Encode ICMP type and code. */
            icmp_ipv4_type = (int)agg->icmp_type;             /* Store ICMP type only. */
        }                                                     /* End ICMP block. */
    } else {                                                  /* If no aggregation exists. */
        in_pkts = f->todstpktcnt;                             /* Fallback to Suricata to-destination packet counter. */
        out_pkts = f->tosrcpktcnt;                            /* Fallback to Suricata to-source packet counter. */
    }                                                         /* End aggregation branch. */

    double src_to_dst_second_bytes = BytesPerSecond(in_bytes, duration_in_us); /* Compute direction 0 bytes/sec. */
    double dst_to_src_second_bytes = BytesPerSecond(out_bytes, duration_out_us); /* Compute direction 1 bytes/sec. */

    double src_to_dst_avg_throughput = BitsPerSecond(in_bytes, duration_in_us); /* Compute direction 0 bits/sec. */
    double dst_to_src_avg_throughput = BitsPerSecond(out_bytes, duration_out_us); /* Compute direction 1 bits/sec. */

    int dns_query_id = 0;                                     /* Placeholder; fill later from EVE DNS. */
    int dns_query_type = 0;                                   /* Placeholder; fill later from EVE DNS. */
    int dns_ttl_answer = 0;                                   /* Placeholder; fill later from EVE DNS. */
    int ftp_command_ret_code = 0;                             /* Placeholder; fill later from EVE FTP. */

    int label = -1;                                           /* Placeholder label; fill later from alerts or ground truth. */
    const char *attack = "Unlabeled";                         /* Placeholder attack class. */

    pthread_mutex_lock(&g_file_mutex);                        /* Lock file writing. */

    fprintf(fp,                                                /* Write one flow CSV row. */
            "%lu,"                                           /* flow_id. */
            "%s,"                                            /* IPV4_SRC_ADDR. */
            "%u,"                                            /* L4_SRC_PORT. */
            "%s,"                                            /* IPV4_DST_ADDR. */
            "%u,"                                            /* L4_DST_PORT. */
            "%u,"                                            /* PROTOCOL. */
            "%d,"                                            /* L7_PROTO. */
            "%lu,"                                           /* IN_BYTES. */
            "%lu,"                                           /* IN_PKTS. */
            "%lu,"                                           /* OUT_BYTES. */
            "%lu,"                                           /* OUT_PKTS. */
            "%u,"                                            /* TCP_FLAGS. */
            "%u,"                                            /* CLIENT_TCP_FLAGS. */
            "%u,"                                            /* SERVER_TCP_FLAGS. */
            "%lu,"                                           /* FLOW_DURATION_MILLISECONDS. */
            "%lu,"                                           /* DURATION_IN. */
            "%lu,"                                           /* DURATION_OUT. */
            "%u,"                                            /* MIN_TTL. */
            "%u,"                                            /* MAX_TTL. */
            "%u,"                                            /* LONGEST_FLOW_PKT. */
            "%u,"                                            /* SHORTEST_FLOW_PKT. */
            "%u,"                                            /* MIN_IP_PKT_LEN. */
            "%u,"                                            /* MAX_IP_PKT_LEN. */
            "%.6f,"                                          /* SRC_TO_DST_SECOND_BYTES. */
            "%.6f,"                                          /* DST_TO_SRC_SECOND_BYTES. */
            "%lu,"                                           /* RETRANSMITTED_IN_BYTES. */
            "%lu,"                                           /* RETRANSMITTED_IN_PKTS. */
            "%lu,"                                           /* RETRANSMITTED_OUT_BYTES. */
            "%lu,"                                           /* RETRANSMITTED_OUT_PKTS. */
            "%.6f,"                                          /* SRC_TO_DST_AVG_THROUGHPUT. */
            "%.6f,"                                          /* DST_TO_SRC_AVG_THROUGHPUT. */
            "%lu,"                                           /* NUM_PKTS_UP_TO_128_BYTES. */
            "%lu,"                                           /* NUM_PKTS_128_TO_256_BYTES. */
            "%lu,"                                           /* NUM_PKTS_256_TO_512_BYTES. */
            "%lu,"                                           /* NUM_PKTS_512_TO_1024_BYTES. */
            "%lu,"                                           /* NUM_PKTS_1024_TO_1514_BYTES. */
            "%u,"                                            /* TCP_WIN_MAX_IN. */
            "%u,"                                            /* TCP_WIN_MAX_OUT. */
            "%d,"                                            /* ICMP_TYPE. */
            "%d,"                                            /* ICMP_IPV4_TYPE. */
            "%d,"                                            /* DNS_QUERY_ID. */
            "%d,"                                            /* DNS_QUERY_TYPE. */
            "%d,"                                            /* DNS_TTL_ANSWER. */
            "%d,"                                            /* FTP_COMMAND_RET_CODE. */
            "%d,"                                            /* Label. */
            "%s\n",                                          /* Attack. */
            (unsigned long)flow_id,                           /* flow_id value. */
            src_ip,                                           /* IPV4_SRC_ADDR value. */
            src_port,                                         /* L4_SRC_PORT value. */
            dst_ip,                                           /* IPV4_DST_ADDR value. */
            dst_port,                                         /* L4_DST_PORT value. */
            f->proto,                                         /* PROTOCOL value. */
            f->alproto,                                       /* L7_PROTO value. */
            (unsigned long)in_bytes,                          /* IN_BYTES value. */
            (unsigned long)in_pkts,                           /* IN_PKTS value. */
            (unsigned long)out_bytes,                         /* OUT_BYTES value. */
            (unsigned long)out_pkts,                          /* OUT_PKTS value. */
            tcp_flags_total,                                  /* TCP_FLAGS value. */
            client_tcp_flags,                                 /* CLIENT_TCP_FLAGS value. */
            server_tcp_flags,                                 /* SERVER_TCP_FLAGS value. */
            (unsigned long)(flow_duration_us / 1000ULL),      /* FLOW_DURATION_MILLISECONDS value. */
            (unsigned long)(duration_in_us / 1000ULL),        /* DURATION_IN value. */
            (unsigned long)(duration_out_us / 1000ULL),       /* DURATION_OUT value. */
            min_ttl,                                          /* MIN_TTL value. */
            max_ttl,                                          /* MAX_TTL value. */
            longest_flow_pkt,                                 /* LONGEST_FLOW_PKT value. */
            shortest_flow_pkt,                                /* SHORTEST_FLOW_PKT value. */
            min_ip_pkt_len,                                   /* MIN_IP_PKT_LEN value. */
            max_ip_pkt_len,                                   /* MAX_IP_PKT_LEN value. */
            src_to_dst_second_bytes,                          /* SRC_TO_DST_SECOND_BYTES value. */
            dst_to_src_second_bytes,                          /* DST_TO_SRC_SECOND_BYTES value. */
            (unsigned long)retransmitted_in_bytes,            /* RETRANSMITTED_IN_BYTES value. */
            (unsigned long)retransmitted_in_pkts,             /* RETRANSMITTED_IN_PKTS value. */
            (unsigned long)retransmitted_out_bytes,           /* RETRANSMITTED_OUT_BYTES value. */
            (unsigned long)retransmitted_out_pkts,            /* RETRANSMITTED_OUT_PKTS value. */
            src_to_dst_avg_throughput,                        /* SRC_TO_DST_AVG_THROUGHPUT value. */
            dst_to_src_avg_throughput,                        /* DST_TO_SRC_AVG_THROUGHPUT value. */
            (unsigned long)num_pkts_up_to_128,                /* NUM_PKTS_UP_TO_128_BYTES value. */
            (unsigned long)num_pkts_128_to_256,               /* NUM_PKTS_128_TO_256_BYTES value. */
            (unsigned long)num_pkts_256_to_512,               /* NUM_PKTS_256_TO_512_BYTES value. */
            (unsigned long)num_pkts_512_to_1024,              /* NUM_PKTS_512_TO_1024_BYTES value. */
            (unsigned long)num_pkts_1024_to_1514,             /* NUM_PKTS_1024_TO_1514_BYTES value. */
            tcp_win_max_in,                                   /* TCP_WIN_MAX_IN value. */
            tcp_win_max_out,                                  /* TCP_WIN_MAX_OUT value. */
            icmp_type,                                        /* ICMP_TYPE value. */
            icmp_ipv4_type,                                   /* ICMP_IPV4_TYPE value. */
            dns_query_id,                                     /* DNS_QUERY_ID value. */
            dns_query_type,                                   /* DNS_QUERY_TYPE value. */
            dns_ttl_answer,                                   /* DNS_TTL_ANSWER value. */
            ftp_command_ret_code,                             /* FTP_COMMAND_RET_CODE value. */
            label,                                            /* Label value. */
            attack);                                          /* Attack value. */

    fflush(fp);                                               /* Flush row to disk. */

    pthread_mutex_unlock(&g_file_mutex);                      /* Unlock file writing. */
}                                                             /* End of WriteFlowCsvLine. */

static int CustomPacketCsvLogger(ThreadVars *tv, void *thread_data, const Packet *p) /* Packet logger callback. */
{                                                             /* Start of CustomPacketCsvLogger. */
    (void)tv;                                                 /* Mark unused parameter to avoid compiler warnings. */

    CustomLoggerThreadData *tdata = (CustomLoggerThreadData *)thread_data; /* Cast logger thread data. */

    if (tdata == NULL || p == NULL)                           /* Check invalid input. */
        return 0;                                             /* Return success without doing anything. */

    UpdateFlowAggFromPacket(p);                               /* Update flow aggregation using this packet. */
    WritePacketCsvLine(tdata->packet_fp, p);                  /* Write this packet as one CSV row. */

    return 0;                                                 /* Return success. */
}                                                             /* End of CustomPacketCsvLogger. */

static int CustomFlowCsvLogger(ThreadVars *tv, void *thread_data, Flow *f) /* Flow logger callback. */
{                                                             /* Start of CustomFlowCsvLogger. */
    (void)tv;                                                 /* Mark unused parameter. */

    CustomLoggerThreadData *tdata = (CustomLoggerThreadData *)thread_data; /* Cast logger thread data. */

    if (tdata == NULL || f == NULL)                           /* Check invalid input. */
        return 0;                                             /* Return success without writing. */

    uint64_t flow_id = FlowGetId(f);                          /* Get flow id. */

    FlowAgg agg_copy;                                         /* Local copy of aggregation to use after unlocking. */
    FlowAgg *agg_ptr = NULL;                                  /* Pointer to local copy if aggregation exists. */

    pthread_mutex_lock(&g_agg_mutex);                         /* Lock aggregation table. */

    FlowAgg *agg = FlowAggLookupLocked(flow_id);              /* Find aggregation for this flow. */
    if (agg != NULL) {                                        /* If aggregation exists. */
        agg_copy = *agg;                                      /* Copy aggregation to stack. */
        agg_copy.next = NULL;                                 /* Clear linked-list pointer in the copy. */
        agg_ptr = &agg_copy;                                  /* Use copied aggregation after unlock. */
        FlowAggRemoveLocked(flow_id);                         /* Remove and free original aggregation. */
    }                                                         /* End aggregation copy block. */

    pthread_mutex_unlock(&g_agg_mutex);                       /* Unlock aggregation table. */

    WriteFlowCsvLine(tdata->flow_fp, f, agg_ptr);             /* Write one flow CSV row. */

    return 0;                                                 /* Return success. */
}                                                             /* End of CustomFlowCsvLogger. */

static bool CustomPacketLoggerCondition(ThreadVars *tv, void *thread_data, const Packet *p) /* Decides whether to log a packet. */
{                                                             /* Start of CustomPacketLoggerCondition. */
    (void)tv;                                                 /* Mark unused parameter. */
    (void)thread_data;                                        /* Mark unused parameter. */
    (void)p;                                                  /* Mark unused parameter. */

    return true;                                              /* Log every packet. */
}                                                             /* End of CustomPacketLoggerCondition. */

static TmEcode ThreadInit(ThreadVars *tv, const void *initdata, void **data) /* Initializes logger thread data. */
{                                                             /* Start of ThreadInit. */
    (void)tv;                                                 /* Mark unused parameter. */
    (void)initdata;                                           /* Mark unused parameter. */

    CustomLoggerThreadData *tdata = calloc(1, sizeof(*tdata)); /* Allocate per-thread data. */
    if (tdata == NULL) {                                      /* If allocation failed. */
        SCLogError("Could not allocate custom CSV logger thread data"); /* Log error. */
        return TM_ECODE_FAILED;                               /* Tell Suricata initialization failed. */
    }                                                         /* End allocation error block. */

    tdata->packet_fp = OpenCsvFileWithHeader(CUSTOM_PACKET_CSV_FILE, PACKET_CSV_HEADER); /* Open packet CSV. */
    if (tdata->packet_fp == NULL) {                           /* If packet CSV could not be opened. */
        SCLogError("Could not open packet CSV file: %s", CUSTOM_PACKET_CSV_FILE); /* Log error. */
        free(tdata);                                          /* Free allocated thread data. */
        return TM_ECODE_FAILED;                               /* Tell Suricata initialization failed. */
    }                                                         /* End packet file error block. */

    tdata->flow_fp = OpenCsvFileWithHeader(CUSTOM_FLOW_CSV_FILE, FLOW_CSV_HEADER); /* Open flow CSV. */
    if (tdata->flow_fp == NULL) {                             /* If flow CSV could not be opened. */
        SCLogError("Could not open flow CSV file: %s", CUSTOM_FLOW_CSV_FILE); /* Log error. */
        fclose(tdata->packet_fp);                             /* Close already opened packet CSV. */
        free(tdata);                                          /* Free thread data. */
        return TM_ECODE_FAILED;                               /* Tell Suricata initialization failed. */
    }                                                         /* End flow file error block. */

    *data = tdata;                                            /* Return thread data to Suricata. */
    return TM_ECODE_OK;                                       /* Initialization succeeded. */
}                                                             /* End of ThreadInit. */

static TmEcode ThreadDeinit(ThreadVars *tv, void *data)       /* Cleans up logger thread data. */
{                                                             /* Start of ThreadDeinit. */
    (void)tv;                                                 /* Mark unused parameter. */

    CustomLoggerThreadData *tdata = (CustomLoggerThreadData *)data; /* Cast thread data. */

    if (tdata != NULL) {                                      /* If thread data exists. */
        if (tdata->packet_fp != NULL)                         /* If packet CSV is open. */
            fclose(tdata->packet_fp);                         /* Close packet CSV. */

        if (tdata->flow_fp != NULL)                           /* If flow CSV is open. */
            fclose(tdata->flow_fp);                           /* Close flow CSV. */

        free(tdata);                                          /* Free thread data. */
    }                                                         /* End cleanup block. */

    return TM_ECODE_OK;                                       /* Cleanup succeeded. */
}                                                             /* End of ThreadDeinit. */

static void OnLoggingReady(void *arg)                         /* Callback invoked when Suricata logging is ready. */
{                                                             /* Start of OnLoggingReady. */
    (void)arg;                                                /* Mark unused parameter. */

    SCOutputRegisterPacketLogger(LOGGER_USER,                 /* Register a user packet logger. */
                                 "custom-packet-logger",     /* YAML output name for packet logger. */
                                 CustomPacketCsvLogger,       /* Packet callback function. */
                                 CustomPacketLoggerCondition, /* Condition function: log packet or not. */
                                 NULL,                        /* Initialization data; not used. */
                                 ThreadInit,                  /* Per-thread initialization function. */
                                 ThreadDeinit);               /* Per-thread cleanup function. */

    SCOutputRegisterFlowLogger("custom-flow-logger",          /* Register a flow logger with this YAML name. */
                               CustomFlowCsvLogger,           /* Flow callback function. */
                               NULL,                          /* Initialization data; not used. */
                               ThreadInit,                    /* Per-thread initialization function. */
                               ThreadDeinit);                 /* Per-thread cleanup function. */
}                                                             /* End of OnLoggingReady. */

static void Init(void)                                        /* Plugin initialization function called by Suricata. */
{                                                             /* Start of Init. */
    SCRegisterOnLoggingReady(OnLoggingReady, NULL);           /* Delay logger registration until logging subsystem is ready. */
}                                                             /* End of Init. */

const SCPlugin PluginRegistration = {                         /* Static plugin metadata object. */
    .version = SC_API_VERSION,                                /* Suricata plugin API version expected by this plugin. */
    .suricata_version = SC_PACKAGE_VERSION,                   /* Suricata version used to build this plugin. */
    .name = "custom-stage1-csv-logger",                       /* Plugin name. */
    .plugin_version = "2.1.0-annotated",                      /* Plugin version string. */
    .author = "Xiaoyan Xiong",                               /* Plugin author. */
    .license = "GPLv2",                                      /* License string. */
    .Init = Init,                                             /* Pointer to plugin initialization function. */
};                                                            /* End of plugin metadata object. */

const SCPlugin *SCPluginRegister(void)                        /* Required exported function for Suricata to load plugin. */
{                                                             /* Start of SCPluginRegister. */
    return &PluginRegistration;                               /* Return plugin metadata to Suricata. */
}                                                             /* End of SCPluginRegister. */
