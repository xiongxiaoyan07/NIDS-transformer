CATEGORIES = [
    (
        "1. Research framing, research questions, and contributions",
        "研究定位、研究问题与贡献",
        [
            (
                "A",
                "Can you explain your thesis in one sentence?",
                "你能用一句话解释你的论文吗？",
                "My thesis tests whether a network detector can improve flow-level decisions by modelling two levels of structure: packet order and elapsed time inside each flow, and selected historical context between related flows.",
                "先说最终任务仍然是 flow-level detection，再说两个层次：flow 内部的 packet order/time，以及 flow 之间的 historical context。不要先陷入模型细节。",
                "Do not say the model detects every attack. Say it improves the evaluated alert-associated classification task."
            ),
            (
                "A",
                "What is the central research problem?",
                "论文的核心研究问题是什么？",
                "The central problem is how to use packet-level temporal information and cross-flow context without losing the natural hierarchy of network traffic. Many systems either compress a flow too early or classify every flow independently. My work studies both information losses in one controlled framework.",
                "核心不是“使用 Transformer”，而是同时解决 aggregation 丢失 flow 内部过程信息，以及 independent classification 丢失 flow 间行为信息。",
                "Keep the claim at the level of the evaluated framework, not all NIDS systems."
            ),
            (
                "A",
                "What is your first research question?",
                "你的第一个研究问题是什么？",
                "The exact wording is: Does time-aware packet-sequence modelling produce more informative flow representations than flow-level aggregation alone? I answer it by comparing the packet-sequence models with a flow-statistics MLP and by using encoding and fusion ablations on shared test manifests.",
                "必须使用论文原话。回答时补充证据来源：flow MLP 对照、position/time 消融和 fusion 消融。",
                "Do not answer RQ1 only with the best score; mention the controlled comparison."
            ),
            (
                "A",
                "What is your second research question?",
                "你的第二个研究问题是什么？",
                "The exact wording is: Does incorporating cross-flow context improve flow-level intrusion detection compared with independent per-flow classification? I test this by freezing the Stage 1 embeddings and comparing independent classification with causal, relation-specific Stage 2 context on the same flows.",
                "必须使用论文原话。强调固定 Stage 1 embedding，并在相同 flow 上比较 independent prediction 与 Stage 2 context。",
                "Use the word start-ordered when discussing causality; it is not strict completion-time streaming."
            ),
            (
                "A",
                "Why is the model hierarchical?",
                "为什么你的模型是 hierarchical 的？",
                "Because network traffic has at least two natural levels. Packets form a flow, and related flows can form host-level or endpoint-level behaviour. Stage 1 models the first level, and Stage 2 models the second. The final prediction is still made for one target flow.",
                "hierarchical 指数据结构和建模层次，不是简单地把两个网络堆在一起。最终预测粒度仍然是 flow。",
                "Do not describe Stage 2 as classifying an attack campaign; it still outputs a flow label."
            ),
            (
                "A",
                "What is genuinely novel in your work?",
                "你的真正创新点是什么？",
                "The novelty is the integrated and controlled two-stage design. Stage 1 combines ordinal position, cumulative elapsed time, packet metadata, and bounded statistical conditioning. Stage 2 lets the current flow query selected earlier flow embeddings. The main contribution is this hierarchical decomposition and its evaluation, not the invention of self-attention or FiLM by themselves.",
                "主动承认 Transformer、sinusoidal encoding 和 FiLM 都不是你发明的。创新在于面向 NIDS 的组合方式、stage interface、target-query context 和受控评估。",
                "Avoid saying that no earlier study has ever combined similar ideas unless you can prove it exhaustively."
            ),
            (
                "B",
                "What are the main contributions?",
                "论文的主要贡献有哪些？",
                "I would group them into four items. First, an aligned PCAP-to-learning pipeline using a custom Suricata output module. Second, a time-aware Stage 1 packet representation with bounded flow-statistical conditioning. Third, a causal target-query Stage 2 model over relation-specific histories. Fourth, chronological internal and external evaluation, including CICIDS2017 windows and a separate company server.",
                "答辩中最好明确说四项：数据流程、Stage 1、Stage 2、外部评估。这样与 PPT 内容一致。",
                "If the committee treats external evaluation as evidence rather than a technical contribution, accept that classification."
            ),
            (
                "A",
                "Your thesis says 'three principal contributions' but then lists four bullets. Is that an error?",
                "论文写着 three principal contributions，但后面列了四项，这是错误吗？",
                "Yes, that is an editorial inconsistency. The clearest correction is to say four principal contributions. Another possible structure is three technical contributions plus one evaluation contribution. The experiments and conclusions are unchanged, but the wording should be corrected in the final text.",
                "这是一个真实的文字不一致，不能强行辩解。直接承认，然后解释更清楚的分类方式。",
                "Do not deny the visible mismatch. A short, honest correction is the strongest answer."
            ),
            (
                "A",
                "Why did you train the two stages separately instead of end to end?",
                "为什么两个 stage 分开训练，而不是端到端训练？",
                "Separate training makes the research questions easier to test. I can first evaluate the quality of the intra-flow representation and then measure the additional value of cross-flow context using frozen embeddings. It also reduces computational complexity. The trade-off is that Stage 1 cannot adapt to the final Stage 2 objective, so end-to-end fine-tuning is future work.",
                "优点是可解释的实验归因、可复用 embedding、资源可控；缺点是失去 end-to-end optimization。必须同时说优点和代价。",
                "Do not claim separate training is always better. It was an experimental design choice."
            ),
            (
                "B",
                "What exactly does the system predict?",
                "系统最终预测的对象是什么？",
                "It produces a binary probability for each target flow. Class 1 means that at least one packet in that flow was associated with a configured Suricata alert. The model does not directly predict an attack family, a host compromise, or a complete multi-stage campaign.",
                "预测粒度是 flow，标签是 alert-associated binary class。不是 packet 分类、attack family 分类或 campaign detection。",
                "Avoid calling class 1 verified malicious ground truth."
            ),
        ],
    ),
    (
        "2. Networking foundations and relation to prior work",
        "网络基础概念与相关工作",
        [
            (
                "A",
                "What is the difference between a packet and a flow?",
                "Packet 和 flow 有什么区别？",
                "A packet is one transmitted network unit with header and length information. A flow is a group of related packets belonging to one bidirectional communication. One packet is a local event, while a flow describes a conversation over time.",
                "用“信封”和“对话”解释最直观。补充 flow 包含一组按时间排序、双向的 packet。",
                "Do not define a flow only as five tuple in this thesis; Suricata supplies a bidirectional flow identifier."
            ),
            (
                "B",
                "What do you mean by a bidirectional flow?",
                "什么是 bidirectional flow？",
                "A bidirectional flow contains packets in both directions of the same conversation. In this dataset, Suricata records to-server and to-client directions under a common flow identifier. This allows the model to observe direction changes and responses, not only outbound traffic.",
                "强调同一 flow_id 下包含 to-server 与 to-client。direction 是 packet-level binary input 之一。",
                "Do not assume every flow has packets in both directions; incomplete or one-way flows can still occur."
            ),
            (
                "B",
                "What information is contained in flow statistics?",
                "Flow statistics 包含哪些信息？",
                "They include duration, forward and backward packet counts, byte totals, packet-length distributions, rates, inter-arrival summaries, TCP flag counts, header lengths, initial windows, and active or idle measurements. They provide a compact global summary of one flow.",
                "答出几个 feature family 即可，不需要背完 77 个 raw columns。强调它们是 global summary。",
                "Remember that the raw 77-column schema is larger than the configured predictor subset."
            ),
            (
                "A",
                "Why are aggregate flow statistics not sufficient?",
                "为什么只使用 aggregate flow statistics 不够？",
                "Aggregation can map different packet processes to similar totals. Packet order, direction transitions, bursts, and unequal time gaps may disappear. The flow MLP result supports this concern: on the shared Stage 1 manifest it reached macro-F1 0.7762 and PR-AUC 0.5604, below the position-plus-time Transformer.",
                "先给概念理由，再给论文中的直接对照结果。不要说 flow statistics 没用，因为 Scheme C 表明它们仍然有价值。",
                "Say insufficient by themselves, not useless."
            ),
            (
                "A",
                "Why distinguish position from elapsed time?",
                "为什么要区分 packet position 和 elapsed time？",
                "Position tells the model where a packet appears in the sequence. Elapsed time tells how long the communication has actually taken. Two packets can be adjacent in order but separated by microseconds or seconds, so the two coordinates carry different information.",
                "最重要的例子：相邻位置可以有完全不同的真实时间间隔。实验中 time-only 高于 position-only，二者结合最好。",
                "Do not say time replaces position; the result supports complementarity."
            ),
            (
                "B",
                "How does a learning-based NIDS differ from Suricata?",
                "学习型 NIDS 与 Suricata 有什么不同？",
                "Suricata mainly uses expert-defined rules and signatures. A learning model estimates patterns from data and may capture combinations that are not directly written as one rule. In this thesis, however, Suricata is also used for extraction and weak labels, so the model learns alert-associated behaviour rather than fully independent attack truth.",
                "不要把二者描述成互相取代。你的 pipeline 同时使用 Suricata 做 extraction 和 label generation。",
                "Do not claim the model proves detection of zero-day attacks."
            ),
            (
                "B",
                "Can the model work on encrypted traffic?",
                "模型能用于加密流量吗？",
                "The model does not consume application payload content. It uses packet metadata, directions, lengths, flags, timing, and flow statistics, so encryption does not remove all of its input signal. However, I did not run a dedicated encrypted-versus-unencrypted experiment, so I cannot claim encryption-invariant performance.",
                "可以说 payload-content-independent，但不能说已经证明适用于所有 encrypted traffic。",
                "Payload length is used as a scalar measurement; it is not payload content."
            ),
            (
                "A",
                "Why did you choose a Transformer rather than only an LSTM or GRU?",
                "为什么选择 Transformer，而不是只用 LSTM 或 GRU？",
                "Self-attention lets distant packets or flows interact directly and supports parallel computation. It also makes it natural to use the current flow as a query over history. I still included LSTM, GRU, BiLSTM, and CNN-based baselines because the choice should be supported by controlled results, not only by theory.",
                "理论理由是 long-range interaction 与 parallelism，实验理由是与 recurrent/convolutional baselines 比较后结果更好。",
                "Do not say Transformers are always faster; Stage 1 was the most expensive observed training run."
            ),
            (
                "B",
                "How is your Stage 1 different from GTID?",
                "你的 Stage 1 与 GTID 有什么不同？",
                "GTID is an important time-aware reference and motivates separating packet position from inter-packet time. My Stage 1 applies a related principle to structured packet metadata without payload content, adds a separate bounded flow-statistics conditioning branch, and exports a flow embedding for a second cross-flow stage. I do not claim the sinusoidal idea itself is new.",
                "相同点：time-aware packet modelling。差异：structured metadata、no payload content、bounded statistics conditioning、embedding exported to Stage 2。",
                "Avoid claiming direct numerical superiority because datasets and protocols differ."
            ),
            (
                "B",
                "How is your work different from FlowTransformer or graph-based NIDS?",
                "你的工作与 FlowTransformer 或 graph-based NIDS 有什么不同？",
                "FlowTransformer is mainly a modular flow-record framework, while my Stage 1 reconstructs packet dynamics before cross-flow reasoning. Graph models can represent richer topology, but they require graph construction and maintenance. I chose bounded relation-specific sequences because they are simpler to audit, naturally causal in order, and computationally practical for this thesis.",
                "不要贬低 graph model。你的选择是 design trade-off：表达能力较简单，但 context construction、causality 和资源控制更清楚。",
                "Do not compare scores across unrelated datasets."
            ),
        ],
    ),
    (
        "3. Data provenance, extraction, labels, and class imbalance",
        "数据来源、提取、标签与类别不平衡",
        [
            (
                "A",
                "Where did the main dataset come from?",
                "主要数据集来自哪里？",
                "The main dataset is an approximately 8.62-gigabyte enterprise PCAP supplied by the company. It records traffic visible at one organisational collection point. It was not laboratory traffic generated for this thesis, and it is not presented as a public benchmark.",
                "强调真实企业 capture 带来 environmental realism，但也限制到一个组织、一个采集点和一个时间段。",
                "Do not disclose sensitive endpoints or describe the capture as representative of every enterprise."
            ),
            (
                "A",
                "How large is the dataset?",
                "数据集有多大？",
                "The raw extraction produced 11,750,727 packet rows with 40 columns and 421,898 flow rows with 77 columns. There were 405,254 label-0 flows and 16,644 label-1 flows. The positive prevalence was 3.945 percent, or about a 24.35-to-1 imbalance ratio.",
                "这些数字建议直接背下来：8.62 GB、11.75 million packets、421,898 flows、16,644 positives、3.945%。",
                "The 27.85 packets-per-flow value is only a mean and does not describe the heavy-tailed distribution."
            ),
            (
                "B",
                "Why did you write a custom Suricata output module?",
                "为什么需要自定义 Suricata 输出模块？",
                "The custom C module writes packet and flow CSV records during the same replay. Both views share Suricata's bidirectional flow identifier. This avoids reconstructing packet-to-flow membership later with a separate tool and gives a deterministic alignment interface for Stage 1.",
                "核心价值是 same pass、common flow_id、packet/flow alignment。不是为了修改 Suricata 的检测规则。",
                "Exact extractor reproduction is limited because the Suricata version and module revision were not retained."
            ),
            (
                "A",
                "How do you know packets are aligned with the correct flow?",
                "如何保证 packet 与正确的 flow 对齐？",
                "The packet and flow loggers share the same flow identifier. Preprocessing removes invalid identifiers, keeps one effective flow row per identifier, intersects the packet and flow identifier sets, removes unmatched rows, and stably sorts packets by flow identifier and timestamp. These checks establish a one-to-many flow-to-packet relation before splitting.",
                "回答流程：valid flow_id → deduplicate flow rows → intersect ID sets → remove unmatched → stable time sort。",
                "Do not claim that capture loss or asymmetric routing is eliminated; alignment only covers recorded data."
            ),
            (
                "A",
                "How exactly is the label defined?",
                "标签是如何定义的？",
                "At packet level, the label is 1 when the packet has at least one Suricata alert. At flow level, the label is the maximum packet label in that flow. Therefore, a flow is class 1 if at least one of its packets triggered a configured alert.",
                "可以用公式解释：y_flow = max of packet labels。必须称为 alert-associated flow。",
                "Do not call it expert-verified malicious ground truth."
            ),
            (
                "A",
                "Why do you call the labels weak labels?",
                "为什么这些标签是 weak labels？",
                "They are generated by one configured Suricata ruleset rather than independently verified by security experts. A positive can inherit a rule false alarm, and a negative can contain an attack that the rules did not detect. The model therefore learns behaviour associated with those alerts.",
                "弱标签的两个方向都要说：positive 可能 false alarm，negative 可能 missed attack。",
                "Avoid using attack ground truth as a synonym for the labels."
            ),
            (
                "A",
                "Does label 0 mean benign traffic?",
                "Label 0 是否意味着 benign？",
                "No. Label 0 only means that the configured Suricata rules emitted no alert for that flow. It is more accurate to say no-alert-observed than benign. This limitation affects both training and evaluation.",
                "这是高风险问题，答案必须直接说 No。",
                "Do not soften this limitation; state it clearly and then explain the scope."
            ),
            (
                "A",
                "Could the model simply learn to reproduce Suricata?",
                "模型是否只是学习复现 Suricata？",
                "That is a real possibility because the targets come from Suricata. I removed packet labels, flow labels, rule identifiers, alert text, alert counts, and other alert metadata from the inputs. The model cannot directly copy those fields, but it can learn traffic patterns correlated with the rules. The study therefore evaluates alert-associated behaviour, not independent attack discovery.",
                "先承认标签来源带来的依赖，再说明 direct leakage 已移除。不能把相关性学习说成独立检测真值。",
                "Do not claim this experiment proves zero-day detection."
            ),
            (
                "A",
                "Why is accuracy misleading in your dataset?",
                "为什么 accuracy 会误导？",
                "Because only 3.945 percent of flows are positive. A model that always predicts class 0 would obtain about 96.055 percent accuracy but zero positive recall. That is why I focus on macro-F1, PR-AUC, class-1 performance, FPR, and error counts.",
                "这是最容易解释类别不平衡的数字例子。",
                "Do not omit class-1 recall when discussing an apparently strong accuracy."
            ),
            (
                "B",
                "What external datasets did you use, and why?",
                "使用了哪些外部数据，目的是什么？",
                "I used five non-overlapping 20,000-flow windows from the public CICIDS2017 Wednesday capture and one 38,513-flow company capture from a different server. The first tests public cross-dataset transfer, and the second tests cross-server transfer within the organisational setting. Neither external dataset was used for training or model selection.",
                "区分 public cross-dataset 与 company cross-server。它们不是 source training partition 的扩展。",
                "Two external sources provide preliminary transfer evidence, not universal generalisation."
            ),
        ],
    ),
    (
        "4. Preprocessing, leakage control, sequence construction, and privacy",
        "预处理、数据泄漏控制、序列构建与隐私",
        [
            (
                "A",
                "Why did you use a chronological split instead of a random split?",
                "为什么使用 chronological split，而不是 random split？",
                "A chronological split better represents deployment on later traffic and reduces leakage from distributing temporally adjacent behaviour across partitions. Flows are stably ordered by start time. The earliest 70 percent are training, the next 10 percent validation, and the latest 20 percent test.",
                "强调 deployment realism 和 leakage control。所有 packet 跟随 parent flow，不会跨 partition。",
                "Chronological splitting reduces leakage but does not remove every form of temporal dependence or concept drift."
            ),
            (
                "B",
                "What are the exact split sizes?",
                "三个 partition 的准确大小是多少？",
                "The principal split contains 295,329 training flows, 42,189 validation flows, and 84,380 test flows, for 421,898 flows in total. The split is not stratified, so the boundaries are not moved to equalise class proportions.",
                "建议背下来 295,329 / 42,189 / 84,380。说明 stratify=false。",
                "Do not confuse the full 84,380-flow test set with the smaller Stage 1 comparison manifest."
            ),
            (
                "A",
                "Why are there two different test manifests?",
                "为什么实验中有两个不同的 test manifest？",
                "Stage 1 architecture and sensitivity comparisons use a 35,410-flow test manifest with 1,407 positives. The complete Scheme C and all Stage 2 experiments use the full 84,380-flow test partition with 3,520 positives. I compare effect sizes only within models that share a manifest.",
                "这是教授很可能追问的可比性问题。不同 manifest 的数值不能直接作 paired effect comparison。",
                "Do not compare rounded scores from different manifests as if they were on identical flows."
            ),
            (
                "A",
                "How did you prevent preprocessing leakage?",
                "如何避免 preprocessing data leakage？",
                "All data-dependent transformations are fitted only on the training partition. This includes category vocabularies, clipping quantiles, robust-scaling statistics, and sampler weights. The fitted state is then applied unchanged to validation, test, and external data.",
                "回答时列出至少三个 train-fitted state。绝对不能说在 full dataset 上先 scale 再 split。",
                "External data also uses the source-training preprocessing state."
            ),
            (
                "B",
                "Why did you use log transformation and quantile clipping?",
                "为什么使用 log transform 和 quantile clipping？",
                "Many traffic counts, lengths, rates, and time values are nonnegative and heavy-tailed. Log one plus x compresses very large values. Training quantiles at 0.001 and 0.999 limit extreme outliers without using test statistics.",
                "log1p 处理 heavy tail，winsorisation 使用 training quantiles。",
                "Clipping improves robustness but can also remove extreme signals; this trade-off was not separately ablated."
            ),
            (
                "B",
                "Why did you use robust scaling?",
                "为什么使用 robust scaling？",
                "Robust scaling subtracts the training median and divides by the training interquartile range. It is less sensitive to extreme traffic bursts than mean and standard-deviation scaling. The same fitted medians and ranges are reused outside training.",
                "使用 median/IQR，适合 heavy-tailed network features。",
                "Do not claim it is universally optimal; it is the principal preprocessing choice."
            ),
            (
                "B",
                "How are missing values and unseen categories handled?",
                "缺失值和测试时未见类别如何处理？",
                "Missing categorical values receive an explicit MISSING token. The one-hot vocabulary is learned from training categories, and unseen validation or test categories are ignored so the input dimension does not change. Numerical non-finite values are marked missing and then zero-imputed after the configured transformation.",
                "要区分 categorical MISSING token 与 numerical zero imputation。",
                "Ignoring unseen categories can lose information; it prevents dimension drift but is not a learned unknown embedding."
            ),
            (
                "A",
                "Which fields did you exclude, and why?",
                "哪些字段被排除，为什么？",
                "Stage 1 excludes IP addresses, ports, identifiers, absolute timestamps, TCP sequence and acknowledgement numbers, the combined flag integer, application protocol identity, and all target or alert fields. This reduces memorisation of site-specific identities, capture position, and direct target shortcuts. Relative timing is supplied separately.",
                "排除的是 identity、absolute time、possible shortcuts 和 targets。保留 23 个 principal packet predictors。",
                "Some excluded features may be useful operationally; exclusion was a leakage and transfer design choice."
            ),
            (
                "A",
                "If IP addresses are excluded, how can Stage 2 use source-host context?",
                "如果 IP 地址被排除，Stage 2 如何构建 source-host context？",
                "Host identifiers are kept only as metadata for equality-based context construction. They are not one-hot encoded or passed as semantic model features. Stage 2 receives flow embeddings; the identifiers only decide which earlier embeddings are eligible.",
                "这是 predictor 与 context identifier 的关键区别。模型知道“是否相同”，但不学习某个具体 IP 的 embedding。",
                "Historical membership can still reflect host behaviour, so external interpretation must consider NAT and shared hosts."
            ),
            (
                "A",
                "How do you protect privacy if Stage 2 needs host identifiers?",
                "Stage 2 需要 host identifier，如何保护隐私？",
                "The raw identifiers remain inside the authorised environment and are not reported in the thesis. For release, stable pseudonymisation is preferable because equality relations must be preserved. The pseudonym map itself does not need to be shared, and unnecessary endpoint fields should be removed.",
                "不能简单删除所有 host identity，否则 relation membership 会改变。正确做法是 stable pseudonymisation。",
                "The current CSV files are not claimed to be already pseudonymised."
            ),
        ],
    ),
    (
        "5. Stage 1: intra-flow time-aware representation",
        "Stage 1：Flow 内部的时间感知表示",
        [
            (
                "A",
                "What are the inputs and outputs of Stage 1?",
                "Stage 1 的输入和输出是什么？",
                "For each flow, Stage 1 receives a selected packet-feature matrix, a log inter-arrival tensor, a validity mask, and a separate statistical flow vector. It outputs two-class logits and a 128-dimensional fused flow embedding. The embedding is exported before the final classifier for Stage 2.",
                "输入四部分：packet matrix、time tensor、mask、flow statistics。输出 classifier logits 与 128-d embedding。",
                "Stage 2 does not receive raw packets or raw flow statistics."
            ),
            (
                "A",
                "Why did you keep at most 128 packets?",
                "为什么最多保留 128 个 packet？",
                "The packet count is heavy-tailed, so a fixed upper bound is needed for memory and batching. In the sensitivity study, macro-F1 and the error balance were best at 128. Performance did not improve monotonically at 256, so 128 is the best observed budget for this capture, not a universal value.",
                "先说工程需要 fixed budget，再说 empirical sweep 8–256。避免把 128 说成理论最优。",
                "Long flows are truncated, so later evidence can be lost."
            ),
            (
                "B",
                "Why did you use the head packet-selection policy?",
                "为什么采用 head selection？",
                "Head selection keeps the first 128 packets in chronological order. It is deterministic, achieved the best observed macro-F1 and class-1 recall, and can use an early contiguous prefix. However, its advantage over head-tail was only 0.16 percentage points and is not treated as statistically robust.",
                "head 的优势是 deterministic、contiguous prefix、observed result；局限是可能丢失 flow 末尾的证据。",
                "Do not claim head is universally better than head-tail."
            ),
            (
                "A",
                "How does the cumulative time-aware encoding work?",
                "cumulative time-aware encoding 如何工作？",
                "The stored log inter-arrival value is converted back to a nonnegative microsecond gap and masked. These gaps are accumulated to obtain elapsed time from the start of the flow. A smoothed log of cumulative time is added to the ordinal packet position, and the combined coordinate enters a sinusoidal encoding.",
                "简单顺序：recover gap → mask → cumulative sum → log smoothing → add ordinal position → sinusoidal encoding。",
                "Padding positions receive zero encoding."
            ),
            (
                "A",
                "Why not just append inter-arrival time as another feature?",
                "为什么不把 IAT 简单地作为一个普通 feature？",
                "Appending an interval tells the token about one local gap. Using elapsed time as a sequence coordinate changes how the model represents the location of every packet. The ablation separates this design from LSTM-plus-time or CNN-plus-time variants that simply receive time as an appended feature.",
                "普通 feature 与 sequence coordinate 的作用不同。你的实验也显示 timing 的表示方式会影响 PR-AUC 与 operating point。",
                "Do not say appended time can never work; LSTM also benefited from explicit time."
            ),
            (
                "B",
                "Why use cumulative elapsed time rather than only the current gap?",
                "为什么使用 cumulative elapsed time，而不是只看当前时间间隔？",
                "Cumulative time gives each packet a coordinate relative to the start of the flow. It preserves the irregular spacing of the whole sequence instead of treating each gap only as an isolated value. The logarithm prevents very long flows from producing excessively large coordinates.",
                "cumulative coordinate 表示 packet 相对 flow 开始的真实进程；log 用于压缩尺度。",
                "This design was motivated and ablated by coordinate type, but alternative time encodings were not exhaustively tested."
            ),
            (
                "A",
                "Is Stage 1 causal?",
                "Stage 1 是否 causal？",
                "No, not in the packet-by-packet streaming sense. Stage 1 uses self-attention over the completed or truncated flow, so every retained packet can attend to every other retained packet. It is a flow-level classifier, not an early packet-level detector.",
                "必须明确说 No。Stage 1 没有 causal mask，只是 padding mask。",
                "Do not describe the complete system as real-time packet detection."
            ),
            (
                "A",
                "Why use a separate flow-statistics branch?",
                "为什么使用独立的 flow-statistics branch？",
                "Packet sequences describe how the flow develops, while aggregate statistics provide a compact global summary. Encoding the statistics once avoids copying the same vector into every packet position. The separate branch then conditions packet tokens without replacing them.",
                "强调两种 evidence complementary，独立编码避免 tiled vector 干扰 token difference。",
                "The gain belongs to the integrated Scheme C design, not only to one FiLM operation."
            ),
            (
                "B",
                "How is the statistical correction bounded?",
                "统计特征的 correction 如何被限制？",
                "A learned sigmoid scale starts from approximately 0.0067 because its initial logit is minus 5. The residual is also limited to at most 0.13 times the packet-token norm. These controls make training begin near the packet-only pathway and prevent the statistical branch from immediately dominating.",
                "需要记住两个数字：sigma(-5) approximately 0.0067，maximum residual ratio rho=0.13。",
                "The value 0.13 is a configured choice, not a theoretically derived constant."
            ),
            (
                "B",
                "Why zero-initialise the modulation output?",
                "为什么对 modulation output 使用 zero initialization？",
                "Zero initialisation makes the initial gamma and beta corrections zero. Together with the small scale, the model starts as a stable packet-only encoder and learns statistical corrections gradually. This reduces the risk of a noisy aggregate branch controlling the representation at the start of training.",
                "初始阶段 e_F approximately packet token，随后才学习 bounded correction。",
                "This is a stability design; the thesis does not include a separate zero-initialisation ablation."
            ),
            (
                "B",
                "Why combine attention, mean, and max pooling?",
                "为什么同时使用 attention、mean 和 max pooling？",
                "They summarise different aspects of the packet states. Attention pooling can focus on a few informative packets, mean pooling describes overall behaviour, and max pooling keeps strong local activations. Their concatenation is projected back to one 128-dimensional embedding.",
                "三个 pooling 的角色分别是 selective、global average、strong local evidence。所有 pooling 都使用 mask。",
                "The thesis evaluates the complete hybrid pooling path, not an exhaustive pooling-only ablation."
            ),
        ],
    ),
    (
        "6. Stage 2: causal inter-flow context modelling",
        "Stage 2：Flow 之间的因果上下文建模",
        [
            (
                "A",
                "What are the inputs and outputs of Stage 2?",
                "Stage 2 的输入和输出是什么？",
                "Stage 2 receives frozen 128-dimensional Stage 1 flow embeddings and metadata used only to select eligible history. For each target flow, it builds a bounded sequence of earlier related embeddings and outputs a binary probability for that target. It does not load raw packets, raw flow statistics, or historical labels.",
                "输入内容是 frozen flow embeddings，metadata 只用于 relation selection。输出仍然是 target flow 的 binary probability。",
                "Do not say Stage 2 sees the attack labels of earlier flows."
            ),
            (
                "A",
                "What context relations did you test?",
                "你测试了哪些 context relation？",
                "I tested four relations. Time-only uses recent flows globally. Source-host uses earlier flows initiated by the same source. Destination-host uses earlier flows sent to the same destination. Endpoint context accepts an earlier flow that shares either endpoint in either role.",
                "四个 relation 的 eligibility condition 要准确：global time、same source、same destination、any shared endpoint。",
                "These relations define eligibility, not learned semantic host embeddings."
            ),
            (
                "B",
                "Why is the target included in a window of 128?",
                "为什么 W=128 的 context window 还包含 target？",
                "The current flow occupies the final valid position, so a window of 128 contains at most 127 historical flows plus one target. Keeping the target at a fixed final position simplifies masking, recurrent baselines, and the target-query interface.",
                "W 是 total tokens，不是 128 个 history 再加 target。最多 127 个历史 flow。",
                "Shorter histories are left-padded and padding is masked."
            ),
            (
                "B",
                "What does the online context policy mean?",
                "online context policy 是什么意思？",
                "The detector state persists in chronological order. A validation target may use earlier training flows. A test target may use earlier training, validation, and test flows. The current flow is added to history only after its own context has been constructed, and no future flow or historical label is used.",
                "这模拟 continuous deployment state，而不是把每个 split 的 history 都清空。context 只从时间上更早的 flow 取。",
                "The policy is start-ordered, not strict completion-time availability."
            ),
            (
                "A",
                "Is Stage 2 truly causal?",
                "Stage 2 是否真正 causal？",
                "It is causal in stable flow-start order: every context index is earlier than the target index, and future embeddings and historical labels are excluded. It is not strict streaming causality because an earlier long flow may not have completed when the target starts. A production system should use flow-end or embedding-availability time.",
                "先肯定 start-order causal，再主动说明不满足 end-time availability。这个边界必须讲清楚。",
                "Never describe the current implementation as fully timed real-time streaming."
            ),
            (
                "A",
                "How did you prevent context leakage?",
                "如何避免 Stage 2 的 context leakage？",
                "Flows are stably sorted by start time and flow identifier. Eligible indices are retrieved before the target is inserted into history. Split membership is retained, future indices are impossible, labels are never retrieved as context, and the same stored context indices and masks are reused across compared architectures.",
                "四点：stable order、retrieve before update、no labels/future、reuse identical contexts。",
                "This prevents designed leakage, but it does not solve the completion-time availability limitation."
            ),
            (
                "A",
                "How does target-query attention differ from a vanilla Transformer?",
                "Target-query attention 与 vanilla Transformer 有什么区别？",
                "A vanilla Transformer updates all tokens through full self-attention and classifies the last valid state. My model keeps the current flow as one query and uses historical embeddings only as keys and values. The retrieved history is treated as a correction to the current-flow representation.",
                "核心 inductive bias：current flow 是唯一 query，history 是 evidence source，不需要更新所有 historical tokens。",
                "The architecture comparison does not isolate only this difference because parameter counts and other choices also vary."
            ),
            (
                "B",
                "What is the purpose of the learned gate?",
                "learned gate 的作用是什么？",
                "The gate controls, dimension by dimension, how much the historical summary should modify the current flow. If the history is irrelevant, the model can keep a strong direct path from the target embedding. This is safer than forcing all context to have the same influence.",
                "gate 让 context 成为 optional correction，而不是强制替换 current representation。",
                "Attention or gate weights are not automatically causal explanations of an attack."
            ),
            (
                "B",
                "What happens when there is no eligible history?",
                "如果没有 eligible history，会发生什么？",
                "The historical summary is explicitly set to zero. The gate and context update are also zero, so the current-flow representation passes through unchanged. This avoids allowing an artificial padding token to act as context.",
                "empty context 应该退化为 current-flow path，而不是让 padding 产生虚假 evidence。",
                "This behaviour is defined by implementation, not learned from a fake history token."
            ),
            (
                "B",
                "Does Stage 2 encode real inter-flow time gaps?",
                "Stage 2 是否编码真实的 inter-flow time gap？",
                "No. Stage 2 uses a relative context-age index: age zero is the target, age one is the most recent history, and larger ages are older retained flows. The time-only relation means global recency, not continuous microsecond gaps. Explicit microsecond timing is used only inside Stage 1.",
                "这是重要边界：Stage 2 使用 ordinal age，不是 elapsed-time coordinate。",
                "Do not describe time-only Stage 2 context as a continuous-time model."
            ),
        ],
    ),
    (
        "7. Training, class imbalance, model selection, and metrics",
        "训练、类别不平衡、模型选择与评估指标",
        [
            (
                "A",
                "Why did you use focal loss?",
                "为什么使用 focal loss？",
                "The positive class is rare and many negative examples are easy. Focal loss reduces the influence of already well-classified examples and keeps more attention on difficult cases. I use it together with controlled sampling, while validation and test distributions remain unchanged.",
                "focal loss 作用在 loss weighting，downweight easy examples。不能与 sampler 的作用混淆。",
                "The thesis does not prove focal loss is optimal because a loss-function ablation was not reported."
            ),
            (
                "B",
                "Why use both focal loss and weighted sampling?",
                "为什么同时使用 focal loss 和 weighted sampling？",
                "They act at different points. The sampler changes the expected class exposure in training mini-batches, while focal loss changes each example's contribution to the objective. Sampling is limited to training, and the empirical validation and test distributions are preserved.",
                "sampler 改 batch composition，focal loss 改 loss contribution。二者都只影响 training。",
                "Using both can affect calibration, which is one reason the threshold is selected on validation data."
            ),
            (
                "B",
                "What were the imbalance settings?",
                "类别不平衡相关的具体参数是什么？",
                "Stage 1 uses focal gamma 1, class weights 1 and 1.25, and a sampler target of 10 percent class 1. Stage 2 uses gamma 1, weights 1 and 1.01, and a 5 percent class-1 sampler target. Both use no label smoothing.",
                "需要记住：S1 sampler 10%、alpha [1,1.25]；S2 sampler 5%、alpha [1,1.01]；gamma=1。",
                "These are configured values, not universally optimal class-imbalance settings."
            ),
            (
                "B",
                "Why is the weighted sampler not used for embedding export?",
                "为什么 embedding export 不能使用 weighted sampler？",
                "The sampler draws with replacement. If used for export, it would duplicate some flows and omit others. Therefore, training batches use the sampler, while validation, test, and embedding export use deterministic loaders with exactly one row per flow.",
                "这是非常具体的数据完整性问题。Stage 2 必须得到每个 flow 恰好一个 embedding。",
                "Do not reuse a sampled training loader for deterministic export."
            ),
            (
                "B",
                "What were the main optimisation settings?",
                "主要训练参数是什么？",
                "Both stages use AdamW and batch size 128. Stage 1 uses learning rate 1e-4, weight decay 1e-4, and early-stopping patience 25. Stage 2 uses learning rate 3e-5, weight decay 3e-4, five warm-up epochs, cosine annealing, and patience 15. Both allow up to 500 epochs.",
                "不需要主动背出所有参数，但教授问时要能区分 S1 与 S2。",
                "Maximum epochs are not the same as actual epochs because early stopping is used."
            ),
            (
                "A",
                "How did you select the checkpoint?",
                "如何选择 checkpoint？",
                "For every experiment, I restore the checkpoint with the highest validation PR-AUC. The test set is not used for this choice. This keeps model selection focused on positive-class ranking under imbalance.",
                "checkpoint criterion 是 validation PR-AUC，所有 comparison block 使用一致规则。",
                "Checkpoint selection and threshold selection are separate operations."
            ),
            (
                "A",
                "How did you select the classification threshold?",
                "classification threshold 如何选择？",
                "After restoring the best validation-PR-AUC checkpoint, I choose the threshold that maximises validation class-1 F1. Stage 1 uses thresholds induced by the validation precision-recall curve. Stage 2 searches 199 values from 0.01 to 0.99. The chosen threshold is then fixed for test and external evaluation.",
                "顺序必须准确：先 checkpoint，再 validation predictions，再 threshold，最后 untouched test。",
                "Do not retune the threshold on any test or external label."
            ),
            (
                "A",
                "Why is macro-F1 important?",
                "为什么使用 macro-F1？",
                "Macro-F1 computes the F1 score for each class and gives the two classes equal weight. This prevents the large negative class from dominating the summary. I also report class-1 F1 and the confusion matrix because macro averaging can still hide the operational error balance.",
                "macro-F1 解决 class weighting，但不能替代 class-1 metrics 和 FP/FN counts。",
                "Do not describe macro-F1 as threshold independent; it depends on the chosen threshold."
            ),
            (
                "A",
                "You call the metric PR-AUC, but the code uses average_precision_score. Why?",
                "你称它为 PR-AUC，但代码使用 average_precision_score，为什么？",
                "The thesis states this explicitly. The reported value is scikit-learn average_precision_score, which uses the step-wise precision-recall formulation, not trapezoidal interpolation. I use the label PR-AUC consistently in the presentation, but average precision is the precise implementation name. Every compared model uses the same function.",
                "诚实说明 terminology。PPT 统一写 PR-AUC，但论文已明确 implementation 是 average_precision_score。",
                "Do not claim it is trapezoidal area under an interpolated PR curve."
            ),
            (
                "A",
                "Why not rely on ROC-AUC?",
                "为什么不能只看 ROC-AUC？",
                "ROC-AUC can remain high when the negative population is very large. PR-AUC focuses more directly on retrieving the rare positive class. I therefore report both, but give more interpretive weight to PR-AUC, class-1 performance, FPR, and error counts.",
                "ROC-AUC 不是错误指标，只是在 strong imbalance 下可能显得乐观。",
                "Ranking metrics do not describe one deployed operating threshold, so confusion counts are also required."
            ),
        ],
    ),
    (
        "8. Experimental results, ablations, and statistical interpretation",
        "实验结果、消融与统计解释",
        [
            (
                "A",
                "What is the main Stage 1 result?",
                "Stage 1 最重要的结果是什么？",
                "On the common Stage 1 model manifest, the position-plus-time Transformer achieved macro-F1 0.8882 and PR-AUC 0.8564. It outperformed the flow MLP and the recurrent, convolutional, and no-encoding controls in overall balance. Time-only was stronger than position-only, and combining both was strongest.",
                "回答 RQ1 时先给 0.8882 / 0.8564，再说 position 与 time complementary。",
                "The complete Scheme C fusion result is a different comparison and reaches 0.8954 / 0.8763."
            ),
            (
                "A",
                "What does the flow-MLP comparison prove?",
                "与 flow MLP 的比较说明什么？",
                "It is the direct aggregation-only control. On the same test composition, it reached macro-F1 0.7762 and PR-AUC 0.5604, with 853 false positives and 501 false negatives. The position-plus-time model reduced those counts to 296 and 306, supporting the added value of packet order and time.",
                "这个对照直接回答 RQ1，但应使用“supports”而不是“proves all packet models are superior”。",
                "Flow statistics remain useful in Scheme C, so aggregation is not dismissed completely."
            ),
            (
                "B",
                "Why did the GRU have the lowest false-positive rate but not the best result?",
                "为什么 GRU 的 FPR 最低，却不是最佳模型？",
                "The GRU's FPR was 0.8293 percent, but it missed 437 positive flows. The position-plus-time Transformer introduced only 14 more false positives and removed 131 false negatives. A low FPR alone can describe an overly conservative operating point.",
                "说明 FP/FN trade-off。不能只看单个指标。",
                "The preferred model depends on operational costs, so another deployment could choose a different threshold."
            ),
            (
                "A",
                "What did the fusion ablation show?",
                "fusion ablation 得出了什么结论？",
                "Scheme A repeats flow statistics at every packet, Scheme B uses packets only, and Scheme C uses a separate bounded conditioning branch. Scheme B improved over A, suggesting tiled statistics can interfere with token differences. Scheme C was best, with macro-F1 0.8954, PR-AUC 0.8763, 267 false positives, and 293 false negatives.",
                "需要把 A、B、C 的机制和结论都讲清。",
                "Say Scheme C supports separate conditioning; do not attribute the full gain only to FiLM."
            ),
            (
                "A",
                "Can you isolate the benefit of token-FiLM from model capacity?",
                "能否把 token-FiLM 的收益与额外模型容量分开？",
                "Not completely. Scheme C changes both the fusion mechanism and the capacity of the statistical pathway. The current evidence supports the integrated Scheme C design. A stronger causal ablation would compare parameter-matched alternatives, such as a separate concatenation branch with similar capacity.",
                "这是论文主动承认的 confound。正确回答是不能完全 isolate，并提出 parameter-matched control。",
                "Do not claim every improvement comes from the modulation operation alone."
            ),
            (
                "B",
                "Why did performance not keep improving with longer packet sequences?",
                "为什么 packet sequence 越长，效果没有持续提高？",
                "Performance peaked at 128 and declined at 256. Extra packets can add irrelevant or redundant evidence, but the aggregate sweep cannot identify the exact mechanism. Flow-length composition, truncation, dilution, and optimisation may all contribute. A length-stratified analysis would be needed.",
                "可以提出解释，但必须说 mechanism 没有被直接验证。",
                "Do not call 128 a universal temporal horizon."
            ),
            (
                "A",
                "Was head selection significantly better than head-tail?",
                "head 是否显著优于 head-tail？",
                "No statistical significance is established. The macro-F1 difference was only 0.16 percentage points, and the experiments used one seed. I treat that margin as practically inconclusive and keep head as the principal policy because it is deterministic and preserves an early contiguous prefix.",
                "必须明确说没有 significance。选择 head 是 practical design decision。",
                "Do not turn a small one-seed difference into a strong ranking claim."
            ),
            (
                "A",
                "How much did Stage 2 improve over independent classification?",
                "Stage 2 比 independent classification 提升多少？",
                "On the common full-data manifest, macro-F1 increased from 0.8954 to 0.9275 and PR-AUC from 0.8763 to 0.9345. Stage 2 removed 228 false positives and 205 false negatives, and reduced FPR by about 34.2 percent relative. Both error types improved.",
                "这是回答 RQ2 的核心数字：+3.21 macro-F1 points、+5.82 PR-AUC points、FP -228、FN -205。",
                "The no-context architecture is not capacity matched, so history alone is not perfectly isolated."
            ),
            (
                "A",
                "Why was source-host context the best relation?",
                "为什么 source-host context 最好？",
                "It gave the best observed balance: macro-F1 0.9275, PR-AUC 0.9345, and FPR 0.5429 percent. Repeated outbound activity from one initiator may preserve scanning, retries, or coordinated connections. This is a mechanism hypothesis, not a proven explanation, because the labels do not identify attack families.",
                "先给 observed result，再给 possible mechanism，最后说明 binary labels 不能验证机制。",
                "Do not claim source-host is best for every network or every attack."
            ),
            (
                "A",
                "How strong is the Stage 2 architecture comparison?",
                "Stage 2 architecture comparison 的证据有多强？",
                "The target-query model achieved macro-F1 0.9275, class-1 F1 0.8610, and PR-AUC 0.9345. Against the vanilla Transformer, it removed 177 false positives and 117 false negatives on the same flows. The result supports the integrated design, but it does not isolate every component because architectures and parameter counts are not perfectly matched.",
                "同时讲 performance evidence 和 attribution limitation。No-context head 只有 514 parameters，而 proposed Stage 2 有 283,522。",
                "Do not call the no-context head a capacity-matched causal estimate of history."
            ),
        ],
    ),
    (
        "9. External evaluation, transfer, and generalisation",
        "外部评估、迁移与泛化",
        [
            (
                "A",
                "Did you retrain or tune the model on the external datasets?",
                "是否在外部数据上重新训练或调参？",
                "No. External evaluation uses the source-trained checkpoints, source-training preprocessing state, source context settings, and the threshold selected on source validation data. External labels are used only to calculate metrics.",
                "必须明确 no external training、no checkpoint selection、no threshold tuning。",
                "Feature compatibility still requires preparing external artefacts in the same schema."
            ),
            (
                "A",
                "Why did you select only five CICIDS2017 windows? Is that selection bias?",
                "为什么只选择五个 CICIDS2017 windows？是否存在 selection bias？",
                "The windows were screened before performance comparison using one class-composition rule: label 1 had to remain the minority, matching the qualitative imbalance direction of the source task. This produced w01, w02, w06, w07, and w08. The rule did not use model scores, but it limits the claim to prevalence-compatible windows rather than the full Wednesday capture.",
                "承认 scope restriction。selection criterion 是 prevalence，不是 predictive performance，因此不是按好分数挑窗口，但仍不是完整 Wednesday。",
                "Do not present the five windows as an unbiased estimate of the complete CICIDS2017 day."
            ),
            (
                "B",
                "What were the positive proportions in the CICIDS2017 windows?",
                "CICIDS2017 各窗口的 positive proportion 是多少？",
                "The positive proportions were 18.195 percent for w01, 13.515 percent for w02, 16.480 percent for w06, 13.340 percent for w07, and 15.235 percent for w08. Across all 100,000 pooled flows, the positive proportion was 15.353 percent.",
                "这是 PPT 上的表格数字。至少要记住 range 13.340%–18.195% 和 pooled 15.353%。",
                "These prevalences are much higher than the source prevalence of 3.945 percent."
            ),
            (
                "B",
                "What do the CICIDS2017 results show?",
                "CICIDS2017 的结果说明什么？",
                "Window-level macro-F1 ranged from 0.8536 to 0.9226. The pooled macro-F1 was 0.8966, with 2,355 false positives and 2,937 false negatives over 100,000 flows. The model transfers meaningfully, but performance and error balance are not uniform across time windows.",
                "结论应是 meaningful but non-uniform transfer。不能只报一个 pooled score。",
                "Pooled FPR was 2.7821 percent, higher than on the source test."
            ),
            (
                "A",
                "Why is pooled macro-F1 different from the mean window macro-F1?",
                "为什么 pooled macro-F1 与窗口 macro-F1 的平均值不同？",
                "The arithmetic mean of the five window macro-F1 values is 0.8949. The pooled value 0.8966 is recomputed after adding the five confusion matrices. F1 is nonlinear, so calculating after pooling counts is not the same as averaging separate F1 values.",
                "记住 mean=0.8949，pooled=0.8966。区别来自 nonlinear metric aggregation。",
                "Do not average confusion-matrix-derived metrics and call the result pooled without clarification."
            ),
            (
                "A",
                "Why did you not report a pooled CICIDS2017 PR-AUC?",
                "为什么没有报告 pooled CICIDS2017 PR-AUC？",
                "PR-AUC is a ranking measure and cannot be reconstructed from pooled confusion-matrix counts. It requires the flow-level probability scores from all windows to be concatenated and the metric recomputed. Those pooled scores were not retained for this result, so I did not invent or average a pooled PR-AUC.",
                "这是学术严谨性问题。confusion matrix 只对应一个 threshold，无法恢复 ranking curve。",
                "Do not average window PR-AUC values and label that number pooled PR-AUC."
            ),
            (
                "B",
                "Why was w06 weak even though its PR-AUC was high?",
                "为什么 w06 的 PR-AUC 不低，但 macro-F1 较弱？",
                "Window w06 had PR-AUC 0.8906 and ROC-AUC 0.9758, but class-1 recall was only 0.6305 and 1,218 positives were missed. This suggests the ranking signal remained useful while the fixed source threshold was poorly calibrated for that window.",
                "区分 ranking quality 与 selected operating point。可以说 compatible with miscalibration，但不是已证明原因。",
                "Do not retune on w06 test labels merely to improve the reported result."
            ),
            (
                "B",
                "What happened in w07?",
                "w07 出现了什么问题？",
                "Window w07 had the largest false-positive count, 710, and the lowest PR-AUC, 0.8419. This indicates not only an operating-threshold problem but also weaker class separation. It is a useful example of temporal heterogeneity in external transfer.",
                "w06 偏 calibration/recall，w07 同时有较弱 ranking 和高 FP，这两种 failure mode 要区分。",
                "The exact cause cannot be isolated without deeper traffic and attack-family analysis."
            ),
            (
                "A",
                "What were the cross-server results?",
                "另一台 company server 的结果是什么？",
                "The cross-server set contained 38,513 flows with 3.285 percent positive. It achieved macro-F1 0.9026, class-1 F1 0.8112, PR-AUC 0.8771, and FPR 0.4322 percent. There were 161 false positives and 292 false negatives, so degradation appeared mainly as missed positives.",
                "记住 38,513、3.285%、0.9026 macro-F1、0.8771 PR-AUC。",
                "It is a different server inside the organisation, not a fully independent public organisation."
            ),
            (
                "A",
                "Can you claim that the model generalises?",
                "可以声称模型具有泛化能力吗？",
                "I can claim preliminary transfer under the implemented pipeline. Useful discrimination remained on a public dataset and a different company server. I cannot claim universal generalisation because there are only two external sources, their labels and capture conditions differ, and performance varies across windows.",
                "最好使用 preliminary external generalisation 或 transfer evidence，而不是 universal/general-purpose。",
                "Source-host relations may change meaning under NAT, shared services, or different host populations."
            ),
        ],
    ),
    (
        "10. Efficiency, reproducibility, limitations, deployment, and future work",
        "效率、可复现性、局限性、部署与未来工作",
        [
            (
                "A",
                "What does your reported latency include?",
                "论文报告的 latency 包含什么？",
                "It measures only Stage 2 model evaluation after Stage 1 embeddings and context indices have already been prepared. It excludes PCAP reading, Suricata extraction, Stage 1 inference, context indexing, data transfer, and alert generation. Therefore, it is model-level latency, not end-to-end deployment latency.",
                "这是最重要的 efficiency scope。不能把 0.1965 ms 说成 PCAP-to-alert latency。",
                "Tail latency, warm-up, batching, and run-to-run variability were not measured."
            ),
            (
                "B",
                "Is your model faster than the vanilla Transformer?",
                "你的模型比 vanilla Transformer 更快吗？",
                "No meaningful speed claim is supported. The proposed Stage 2 latency was 0.1965 milliseconds per flow and the vanilla Transformer was 0.1967 milliseconds. The 0.0002-millisecond difference is negligible. The valid conclusion is latency parity with a better observed error balance.",
                "必须说 parity，不要说 faster。约 5,089 flows/s 也只对应 measured Stage 2 model path。",
                "Do not overinterpret a single aggregate timing measurement."
            ),
            (
                "B",
                "What is the efficiency advantage over recurrent Stage 2 models?",
                "与 recurrent Stage 2 models 相比有什么效率优势？",
                "The proposed Stage 2 model has 283,522 parameters. Relative to GRU, LSTM, and CNN-LSTM, the parameter count was lower by about 59.0, 69.3, and 76.7 percent, and measured latency was lower by about 42.2, 43.3, and 45.2 percent. These are model-path comparisons on the same GPU.",
                "与 recurrent alternatives 的差异较大，可以说 compactness/latency advantage；与 vanilla 只能说 parity。",
                "Parameter counts were reported rather than artificially matched."
            ),
            (
                "B",
                "How expensive was Stage 1 training?",
                "Stage 1 训练成本如何？",
                "The position-plus-time Transformer had 344,963 trainable parameters and the observed training run took 4,337.6 seconds, or 72.29 minutes, on an NVIDIA L4. It was the most expensive observed Stage 1 run. The timing includes data loading, validation, checkpointing, and early stopping, so it is descriptive rather than architecture-only throughput.",
                "不要因为 Transformer 可以 parallelize 就声称本实验训练最快。实际上 principal S1 run 最贵。",
                "Single-run Colab time is affected by hosted-environment variation."
            ),
            (
                "A",
                "Can another researcher reproduce your results exactly?",
                "其他研究者能否精确复现结果？",
                "The learning pipeline records many important items, including feature order, preprocessing state, split identifiers, checkpoints, and model settings. Exact reproduction is still limited because the exact CPU, software snapshot, Suricata version, ruleset snapshot, custom-module revision, configuration, and PCAP checksum were not retained. I report those missing items rather than reconstructing them after the fact.",
                "要区分 computational pipeline reproducibility 与 exact label/runtime reproduction。后者受到 provenance 缺失限制。",
                "Do not say the study is fully reproducible when the proprietary data and version metadata are unavailable."
            ),
            (
                "A",
                "Why did you use only one random seed?",
                "为什么只使用一个 random seed？",
                "The main reason was the available compute budget and the large experimental matrix. With one seed, training variance is not estimated. Therefore, I treat small differences as descriptive and do not claim statistical significance. Repeated seeds and confidence intervals are a priority for future work.",
                "直接承认 resource limitation，并说明如何限制结论。不要给虚假的 significance。",
                "Large differences are still descriptive evidence, but their uncertainty is not quantified."
            ),
            (
                "A",
                "What are the three most important limitations?",
                "最重要的三个局限性是什么？",
                "First, the binary targets are Suricata-derived weak labels. Second, Stage 2 is causal only in flow-start order, not strict embedding-availability time. Third, model development uses one source capture and one seed, so statistical and external generalisation are limited. I would also mention the non-capacity-matched no-context control and incomplete end-to-end latency measurement.",
                "优先讲 label validity、causality、generalisability/statistical variance。再补 capacity control 和 deployment measurement。",
                "A strong defense answer states limitations before explaining future corrections."
            ),
            (
                "A",
                "Could this system be deployed online today?",
                "这个系统现在可以直接在线部署吗？",
                "Not as a fully validated production detector. A deployment version would need flow-completion or embedding-availability timing, bounded per-relation buffers, out-of-order handling, peak-memory and tail-latency tests, concept-drift monitoring, and evaluation of the full PCAP-to-alert path. The present work is a research prototype with promising model-level results.",
                "必须避免说 production-ready。列出需要补的 streaming components 和 evaluation。",
                "Operational thresholds must also reflect the real cost of false positives and false negatives."
            ),
            (
                "B",
                "What is the most important next experiment?",
                "下一步最重要的实验是什么？",
                "I would first repeat the principal comparisons across several seeds and add a capacity-matched no-history control. I would then enforce completion-time embedding availability. These experiments directly test whether the reported Stage 2 gain is stable and how much comes from historical evidence under realistic causality.",
                "最优先不是继续增加复杂模型，而是补 strongest validity controls：multiple seeds、matched no-context、strict causality。",
                "Attack-family labels would be the next requirement for explaining which relations help which behaviours."
            ),
            (
                "A",
                "If you had to summarise the scientific value and the practical caution, what would you say?",
                "如何同时总结科学价值和实际谨慎性？",
                "The scientific value is evidence that intra-flow timing and selective inter-flow context address different information losses and provide complementary gains in the evaluated setting. The practical caution is that the labels are weak, causality is start-ordered, and external transfer is limited. The results justify further validation, not immediate universal deployment.",
                "最后用 balanced statement：贡献明确，但 evidence boundary 同样明确。这样的回答通常比夸大结果更有说服力。",
                "Do not end with 'the model solves NIDS.' End with a scoped conclusion."
            ),
        ],
    ),
]
