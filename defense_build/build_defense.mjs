import fs from 'node:fs/promises';
import { Presentation, PresentationFile } from '@oai/artifact-tool';

const OUT = 'C:/Users/XiaoyanXiong/Desktop/NIDSTransformer/Defense_Presentation_Xiaoyan_Xiong.pptx';
const RENDER = 'C:/Users/XiaoyanXiong/Desktop/NIDSTransformer/defense_build/rendered';
const IMG = 'C:/Users/XiaoyanXiong/Desktop/NIDSTransformer/doc/figures';
const ARCH = 'C:/Users/XiaoyanXiong/Desktop/NIDSTransformer/rearch_topic/thesis/ArchitecturalOverview.png';
const W=1280,H=720, ink='#111318', muted='#5E6673', blue='#1677B8', cyan='#6DCBF4', green='#149B6D', orange='#E9A000', panel='#F1F3F5', rule='#C8CDD4';
const deck=Presentation.create({slideSize:{width:W,height:H}});

function box(slide,x,y,w,h,fill='none',line='none',radius=false){return slide.shapes.add({geometry:radius?'roundRect':'rect',position:{left:x,top:y,width:w,height:h},fill,line:{style:'solid',fill:line,width:line==='none'?0:1}})}
function txt(slide,text,x,y,w,h,size=26,color=ink,bold=false,align='left'){const s=slide.shapes.add({geometry:'textbox',position:{left:x,top:y,width:w,height:h},fill:'none',line:{style:'solid',fill:'none',width:0}});s.text=text;s.text.style={fontSize:size,typeface:'Arial',color,bold,alignment:align,verticalAlignment:'middle'};return s}
function header(slide,title,n){txt(slide,title,42,30,1170,72,38,ink,true);txt(slide,String(n).padStart(2,'0'),1180,661,55,22,13,muted,false,'right');}
function note(slide,script,source='Thesis: A Hierarchical Time-Aware Transformer for Flow-Level Network Intrusion Detection, Xiaoyan Xiong, 2026.'){slide.speakerNotes.textFrame.setText(script+'\n\n[Sources]\n- '+source);slide.speakerNotes.setVisible(true)}
async function addImg(slide,path,x,y,w,h,fit='contain',alt='Thesis figure'){const b=await fs.readFile(path);slide.images.add({blob:b.buffer.slice(b.byteOffset,b.byteOffset+b.byteLength),contentType:'image/png',alt,fit,position:{left:x,top:y,width:w,height:h}})}
function arrow(slide,x,y,w,color=blue){const a=slide.shapes.add({geometry:'rightArrow',position:{left:x,top:y,width:w,height:34},fill:color,line:{style:'solid',fill:color,width:0}});return a}
function bullets(slide,items,x,y,w,size=25,gap=66){items.forEach((v,i)=>{txt(slide,'•',x,y+i*gap,28,40,size+2,blue,true);txt(slide,v,x+34,y+i*gap,w-34,52,size,ink,false)});}

// 1
{
 const s=deck.slides.add();txt(s,'MSc Thesis Defense',42,42,450,40,24,blue,true);txt(s,'A Hierarchical Time-Aware\nTransformer for Flow-Level\nNetwork Intrusion Detection',42,150,930,270,56,ink,true);txt(s,'Xiaoyan Xiong  |  Supervisor: Le Viet Duc',42,520,780,42,25,muted);txt(s,'25-minute defense',42,582,300,32,20,muted);
 note(s,"Good morning, and thank you for attending my thesis defense. My thesis investigates a hierarchical time-aware Transformer for flow-level network intrusion detection. The central idea is that network traffic has two levels of temporal structure: packets form a flow, and related flows form broader behaviour. I will first introduce the basic networking concepts, then explain the two-stage model, and finally present the evaluation results, limitations, and conclusions.");
}
// 2
{
 const s=deck.slides.add();header(s,'A packet is one transmitted unit; a flow is a conversation',2);
 txt(s,'PACKET',70,135,220,45,28,blue,true);txt(s,'A single unit sent across a network',70,185,430,45,24,ink);['timestamp','source / destination','protocol','length & flags'].forEach((v,i)=>{box(s,70+i*112,265,95,72,i%2? '#DCEFFC':'#EAF6FC',blue,true);txt(s,v,76+i*112,273,83,56,16,ink,true,'center')});
 arrow(s,535,285,100);txt(s,'group by endpoints, protocol\nand direction over time',507,335,160,70,18,muted,false,'center');
 txt(s,'FLOW',730,135,220,45,28,green,true);txt(s,'A bidirectional sequence of related packets',730,185,470,45,24,ink);for(let i=0;i<5;i++){box(s,745+i*86,285,62,44,i===4?'#DDF1E8':'#EAF6FC',i===4?green:blue,true);txt(s,'p'+(i+1),755+i*86,290,42,32,18,ink,true,'center')};txt(s,'Flow statistics summarize the conversation:\nduration, packet count, bytes, rates and timing.',730,380,470,100,23,ink);
 note(s,"Before discussing the model, I will define two basic concepts. A packet is one transmitted unit. It contains metadata such as a timestamp, source and destination, protocol, length, and control flags. A flow groups packets that belong to the same network conversation, normally using endpoint and protocol information. A flow is therefore a sequence, while flow statistics are summaries of that sequence. Traditional flow-based detectors usually keep only the summaries. My thesis asks what is lost when packet order and timing are compressed too early.");
}
// 3
{
 const s=deck.slides.add();header(s,'Network intrusion detection must balance detail and scalability',3);txt(s,'Raw packets',60,180,220,45,27,blue,true);txt(s,'Fine-grained order and timing\nHigh volume and variable length',60,235,260,90,22,ink);arrow(s,330,250,90);txt(s,'Aggregate flows',455,180,260,45,27,orange,true);txt(s,'Compact and efficient\nBut packet dynamics disappear',455,235,285,90,22,ink);arrow(s,760,250,90);txt(s,'Independent decisions',885,180,300,45,27,green,true);txt(s,'Easy to deploy\nBut related behaviour is ignored',885,235,300,90,22,ink);
 box(s,60,430,1120,110,panel,'none',true);txt(s,'Research gap',90,449,190,34,23,blue,true);txt(s,'Most systems model packet dynamics or cross-flow context — rarely both in one controlled hierarchy.',90,487,1040,42,27,ink,true);
 note(s,"A network intrusion detection system monitors traffic and predicts whether activity is benign or suspicious. Signature engines such as Suricata are effective for known patterns, while learning-based systems can model broader regularities. However, there is a trade-off. Raw packets preserve detail but are expensive. Aggregate flows are scalable but remove ordering. Finally, classifying every flow independently ignores behaviours such as scanning, repeated login attempts, or coordinated connections. This motivates the research gap: packet-level structure and cross-flow context are usually studied separately.");
}
// 4
{
 const s=deck.slides.add();header(s,'Two research questions define the hierarchy',4);box(s,42,145,560,170,'#EAF6FC','none',true);txt(s,'RQ1',70,170,100,36,27,blue,true);txt(s,'Does time-aware packet-sequence modelling produce better flow representations than aggregation alone?',70,215,490,80,25,ink,true);box(s,638,145,600,170,'#E8F5EE','none',true);txt(s,'RQ2',668,170,100,36,27,green,true);txt(s,'Does cross-flow context improve detection beyond independent flow classification?',668,215,520,80,25,ink,true);
 txt(s,'Contributions',42,370,300,45,30,ink,true);bullets(s,['Aligned PCAP-to-learning pipeline with leakage controls','Time-aware intra-flow Transformer with statistical conditioning','Target-query attention over relation-specific historical flows','Chronological internal and external evaluation'],42,425,1140,22,49);
 note(s,"The work is organised around two questions. RQ1 asks whether a time-aware packet sequence produces a more informative representation than aggregate flow statistics alone. RQ2 asks whether historical context from related flows improves the final classification. The thesis contributes an aligned extraction pipeline, a time-aware Stage 1 encoder, a relation-aware Stage 2 model, and chronological evaluation on internal and external traffic.");
}
//5
{
 const s=deck.slides.add();header(s,'The dataset is large, chronological, and strongly imbalanced',5);txt(s,'8.62 GB',60,145,250,70,48,blue,true);txt(s,'enterprise PCAP',60,215,250,35,22,muted);txt(s,'11.75 M',385,145,250,70,48,blue,true);txt(s,'packet records',385,215,250,35,22,muted);txt(s,'421,898',710,145,250,70,48,green,true);txt(s,'flow records',710,215,250,35,22,muted);txt(s,'3.945%',1000,145,220,70,48,orange,true);txt(s,'class-1 prevalence',1000,215,220,35,22,muted);
 const labs=['PCAP','Suricata +\ncustom logger','aligned packet\n& flow CSVs','70 / 10 / 20%\nchronological split','model tensors'];labs.forEach((v,i)=>{box(s,45+i*240,370,180,92,i===4?'#E8F5EE':panel,i===4?green:rule,true);txt(s,v,55+i*240,380,160,70,21,ink,true,'center');if(i<4)arrow(s,225+i*240,397,55)});txt(s,'Weak labels: class 1 means at least one Suricata alert in the flow.',60,560,1120,46,25,ink,true,'center');
 note(s,"The source capture contains about 8.62 gigabytes of enterprise traffic. A custom Suricata output module extracts aligned packet and flow records in one pass. The result contains approximately 11.75 million packet rows and 421,898 flows. Only 3.945 percent of the flows have an alert-derived positive label, so accuracy alone would be misleading. Flows are sorted by start time and split chronologically into 70 percent training, 10 percent validation, and 20 percent testing. Importantly, these are weak labels: class one means that Suricata generated at least one alert in the flow; it is not independently verified attack ground truth.");
}
//6
{
 const s=deck.slides.add();header(s,'The model preserves two levels of traffic structure',6);await addImg(s,IMG+'/overall_framework.png',55,150,1170,430,'contain','Two-stage thesis framework');txt(s,'Stage 1 learns how a flow develops. Stage 2 asks whether earlier related flows change its meaning.',80,600,1120,44,24,ink,true,'center');
 note(s,"This figure gives the full idea in one view. Stage 1 receives the ordered packet sequence and converts it into one fixed-length embedding per flow. Stage 2 treats these flow embeddings as tokens and adds historical context before producing the final benign-or-alert-associated prediction. The hierarchy therefore respects the natural structure: packets form flows, and related flows form behaviour.","Thesis Figure: overall framework, generated from the author's implementation.");
}
//7
{
 const s=deck.slides.add();header(s,'Stage 1 encodes packet order and elapsed time separately',7);txt(s,'Packet position',65,155,280,40,26,blue,true);txt(s,'Where does the packet occur?',65,200,330,36,22,ink);txt(s,'1     2     3     4     5',65,255,330,42,28,blue,true);txt(s,'Cumulative time',470,155,300,40,26,orange,true);txt(s,'How much time has elapsed?',470,200,340,36,22,ink);txt(s,'0 μs   8 μs   13 μs        2.1 s',470,255,390,42,27,orange,true);arrow(s,880,255,85);box(s,990,195,225,140,'#EAF6FC',blue,true);txt(s,'Time-aware\nsinusoidal\nencoding',1010,210,185,108,25,ink,true,'center');box(s,65,400,1150,140,panel,'none',true);txt(s,'Why both?',95,425,190,38,25,blue,true);txt(s,'Identical packet order can represent a burst or a slow interaction. Position and elapsed time provide complementary evidence.',95,470,1060,55,26,ink,true);
 note(s,"Stage 1 first projects every selected packet record into the model dimension. Standard positional encoding identifies ordinal order, but it cannot distinguish microseconds from seconds. I therefore reconstruct the inter-arrival intervals, accumulate elapsed time, apply logarithmic smoothing, and combine this coordinate with packet position in a sinusoidal encoding. The ablations later compare position only, time only, both, and no encoding.");
}
//8
{
 const s=deck.slides.add();header(s,'Stage 1 combines packet evidence with aggregate statistics',8);await addImg(s,ARCH,45,120,1190,505,'contain','Detailed two-stage model architecture');txt(s,'Key design: flow statistics condition packet tokens through a bounded residual, rather than being copied to every packet.',70,630,1140,40,22,ink,true,'center');
 note(s,"This detailed architecture shows the complete pipeline. Stage 1 uses selected packet features, time-aware encoding, a separately encoded flow-statistics branch, a masked Transformer, and hybrid attention, mean, and max pooling. Flow statistics modulate packet tokens through a bounded residual. This prevents the aggregate vector from dominating the packet pathway. Stage 2 then builds a historical context, uses the target flow as the query, gates the retrieved summary, and predicts the final probability.","Thesis Figure: detailed architectural overview, created for the thesis.");
}
//9
{
 const s=deck.slides.add();header(s,'Stage 2 retrieves only earlier related flow embeddings',9);const xs=[80,255,430,605];['z(t-4)','z(t-3)','z(t-2)','z(t-1)'].forEach((v,i)=>{box(s,xs[i],220,125,72,'#E8F5EE',green,true);txt(s,v,xs[i]+10,232,105,48,23,ink,true,'center')});arrow(s,755,238,85,green);box(s,875,195,210,120,'#FFF1D6',orange,true);txt(s,'target z(t)\n= query',900,215,160,78,26,ink,true,'center');txt(s,'Possible relations',70,390,270,38,27,ink,true);bullets(s,['same source host','same destination host','shared endpoint','global time-only recency'],70,440,500,22,46);box(s,650,405,520,135,panel,'none',true);txt(s,'Important scope',680,425,200,34,23,orange,true);txt(s,'“Causal” means start-time ordered.\nStrict streaming availability is not yet enforced.',680,466,450,62,24,ink,true);
 note(s,"For each target flow, Stage 2 retrieves earlier embeddings according to a relation. The tested relations are same source host, same destination host, shared endpoint, and global chronological recency. The target embedding provides one query over the historical keys and values. A learned gate controls how much context updates the current representation. I use only earlier items in stable start-time order and never retrieve historical labels. However, this is not yet strict streaming causality, because a long flow that started earlier may not have completed when the target begins.");
}
//10
{
 const s=deck.slides.add();header(s,'The evaluation emphasises minority-class and operational errors',10);const metrics=[['Macro-F1','equal weight to both classes'],['Average Precision','ranking quality for class 1'],['Class-1 F1','positive precision–recall balance'],['FPR + FP / FN','operational error burden']];metrics.forEach((m,i)=>{box(s,55+(i%2)*600,145+Math.floor(i/2)*190,560,145,i===0?'#EAF6FC':panel,'none',true);txt(s,m[0],80+(i%2)*600,165+Math.floor(i/2)*190,250,38,26,i===0?blue:ink,true);txt(s,m[1],80+(i%2)*600,210+Math.floor(i/2)*190,500,60,22,ink)});txt(s,'Checkpoint: best validation AP   |   Threshold: best validation class-1 F1   |   Test remains untouched',70,560,1140,55,23,ink,true,'center');
 note(s,"Because positive flows are rare, accuracy is not the primary metric. I emphasise macro-F1, class-one F1, average precision, false-positive rate, and absolute false-positive and false-negative counts. Checkpoints are selected using validation average precision, while the operating threshold is selected using validation class-one F1. The final test partition is used only after these choices are fixed. In the thesis logs this ranking metric is named PR-AUC, but technically it is scikit-learn average precision.");
}
//11
{
 const s=deck.slides.add();header(s,'Stage 1 benefits from explicit position and time',11);await addImg(s,IMG+'/stage1_model_comparison.png',45,115,1190,510,'contain','Stage 1 architecture comparison');txt(s,'Position + time: macro-F1 0.8882 and AP 0.8564 — above aggregation, recurrent, convolutional and encoding controls.',55,632,1170,38,21,ink,true,'center');
 note(s,"The Stage 1 comparison uses a common controlled test manifest. The position-plus-time Transformer achieves the strongest overall balance, with macro-F1 of 0.8882 and average precision of 0.8564. It exceeds the flow-statistics MLP, recurrent models, convolutional models, and the encoding ablations. Time-only is stronger than position-only, while using both is strongest, supporting the claim that ordinal location and elapsed time are complementary.","Thesis results: Stage 1 model comparison on the common controlled test manifest.");
}
//12
{
 const s=deck.slides.add();header(s,'Separate statistical conditioning gives the best Stage 1 fusion',12);await addImg(s,IMG+'/stage1_fusion_comparison.png',45,120,1190,485,'contain','Stage 1 fusion comparison');txt(s,'Scheme C: macro-F1 0.8954 | AP 0.8763 | 267 FP | 293 FN',120,620,1040,48,29,green,true,'center');
 note(s,"This ablation compares three ways to combine packet and flow information. Scheme A repeats the aggregate vector at every packet. Scheme B uses packets only. Scheme C encodes the aggregate vector once and uses bounded token modulation. Scheme C reaches macro-F1 of 0.8954 and average precision of 0.8763, while also reducing both false positives and false negatives. The careful conclusion is that a separate statistical pathway is useful; because parameter counts also change, the entire gain cannot be assigned to FiLM alone.","Thesis results: Stage 1 packet and flow-statistics integration ablation.");
}
//13
{
 const s=deck.slides.add();header(s,'Source-host history gives the strongest contextual balance',13);await addImg(s,IMG+'/stage2_context_analysis.png',45,120,1190,500,'contain','Stage 2 relation comparison');txt(s,'Source host: macro-F1 0.9275 | AP 0.9345 | FPR 0.5429%',110,625,1060,42,28,green,true,'center');
 note(s,"For Stage 2, the source-host relation gives the strongest combined result. It has the highest macro-F1 and average precision and the lowest false-positive rate among the tested relations. Endpoint context recovers more positive flows, but it also generates more false positives. Global time-only context is less selective. This does not prove that source history is optimal for every attack family; binary weak labels cannot identify which attack mechanisms benefit.","Thesis results: Stage 2 context-relation ablation at window size 128.");
}
//14
{
 const s=deck.slides.add();header(s,'Target-query attention improves ranking and both error types',14);await addImg(s,IMG+'/stage2_architecture_comparison.png',45,120,1190,500,'contain','Stage 2 architecture comparison');txt(s,'Proposed: macro-F1 0.9275 | class-1 F1 0.8610 | AP 0.9345',110,625,1060,42,28,blue,true,'center');
 note(s,"The architecture comparison fixes the Stage 1 embeddings, context relation, window, objective, and evaluation protocol. The proposed target-query model obtains macro-F1 of 0.9275, class-one F1 of 0.8610, and average precision of 0.9345. It reduces both error types compared with the vanilla Transformer. Nevertheless, the no-context head is much smaller, and other baselines are not parameter matched. The result supports the integrated architecture, but it is not a pure causal estimate of historical context alone.","Thesis results: Stage 2 architecture comparison on the common full-data test manifest.");
}
//15
{
 const s=deck.slides.add();header(s,'Useful discrimination transfers, but not uniformly',15);txt(s,'CICIDS2017',55,155,320,42,26,blue,true);txt(s,'Five × 20,000-flow windows',55,205,390,34,22,muted);txt(s,'0.8966',55,260,320,75,54,blue,true);txt(s,'pooled macro-F1',55,335,320,34,22,muted);txt(s,'0.8536–0.9226',55,405,360,55,35,ink,true);txt(s,'window-level macro-F1 range',55,462,380,34,20,muted);
 box(s,500,135,2,430,rule,'none');txt(s,'Company cross-server',580,155,430,42,26,green,true);txt(s,'38,513 flows | 3.285% positive',580,205,500,34,22,muted);txt(s,'0.9026',580,260,320,75,54,green,true);txt(s,'macro-F1',580,335,320,34,22,muted);txt(s,'0.8771',940,260,250,75,54,orange,true);txt(s,'average precision',940,335,250,34,22,muted);txt(s,'Conclusion: transfer is promising, but temporal and environmental shift changes the error balance.',580,430,600,90,25,ink,true);
 note(s,"External evaluation applies the source-trained model without retraining or threshold adjustment. Across five non-overlapping CICIDS2017 Wednesday windows, pooled macro-F1 is 0.8966, while individual windows range from 0.8536 to 0.9226. A separate company server capture reaches macro-F1 of 0.9026 and average precision of 0.8771. These results demonstrate useful discrimination outside the source capture, but the variation shows that transfer is not invariant across time or environment.","Thesis external-evaluation tables: five CICIDS2017 windows and one company cross-server capture.");
}
//16
{
 const s=deck.slides.add();header(s,'The conclusions are strong within a deliberately limited scope',16);txt(s,'What the evidence supports',55,145,500,42,27,green,true);bullets(s,['Packet order and elapsed time are complementary','Aggregate statistics still add information','Selective source-host context improves the tested detector','External performance remains useful but variable'],55,205,550,22,60);txt(s,'What it does not yet establish',680,145,500,42,27,orange,true);bullets(s,['Independently verified attack ground truth','Strict streaming availability at target start','Statistical significance across random seeds','Universal transfer across networks and attack families'],680,205,550,22,60);
 note(s,"The evidence supports four conclusions. Packet order and elapsed time contribute complementary information. Aggregate statistics remain useful when they condition rather than replace packet evidence. Selective source-host history improves the evaluated detector. Finally, discrimination transfers to the two external sources, although unevenly. The thesis does not establish independently verified attack ground truth, strict online availability, statistical significance across seeds, or universal cross-network generalisation. These limitations define the next experiments rather than invalidating the observed results.");
}
//17
{
 const s=deck.slides.add();txt(s,'Takeaway',42,42,300,40,24,blue,true);txt(s,'Network traffic is hierarchical.\nThe detector should be too.',42,165,1050,160,58,ink,true);txt(s,'Stage 1: how the flow develops',42,430,500,40,27,blue,true);txt(s,'Stage 2: how earlier related flows change its meaning',42,482,760,40,27,green,true);txt(s,'Questions?',42,595,300,50,34,ink,true);
 note(s,"To conclude, network traffic is naturally hierarchical, and the detector benefits from respecting that hierarchy. Stage 1 models how a flow develops internally through packet order and elapsed time. Stage 2 asks how earlier related flows change the interpretation of the target. Together, these two levels improve alert-associated flow classification in the evaluated settings. Thank you for your attention. I welcome your questions.");
}

await fs.mkdir(RENDER,{recursive:true});
for (const [i,s] of deck.slides.items.entries()) {const png=await deck.export({slide:s,format:'png',scale:1});await fs.writeFile(`${RENDER}/slide-${String(i+1).padStart(2,'0')}.png`,new Uint8Array(await png.arrayBuffer()));}
const pptx=await PresentationFile.exportPptx(deck);await pptx.save(OUT);
console.log(OUT);
