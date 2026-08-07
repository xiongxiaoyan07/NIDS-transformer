import fs from 'node:fs/promises';
import { Presentation, PresentationFile } from '@oai/artifact-tool';

const OUT = 'C:/Users/XiaoyanXiong/Desktop/NIDSTransformer/Defense_Presentation_Xiaoyan_Xiong_Revised.pptx';
const RENDER = 'C:/Users/XiaoyanXiong/Desktop/NIDSTransformer/defense_build/rendered_revised';
const IMG = 'C:/Users/XiaoyanXiong/Desktop/NIDSTransformer/doc/figures';
const ARCH = 'C:/Users/XiaoyanXiong/Desktop/NIDSTransformer/rearch_topic/thesis/ArchitecturalOverview.png';
const PACKET_FLOW = 'C:/Users/XiaoyanXiong/Desktop/NIDSTransformer/defense_build/packet_flow_infographic.png';
const TALK = JSON.parse(await fs.readFile('C:/Users/XiaoyanXiong/Desktop/NIDSTransformer/defense_build/notes_v2.json','utf8'));
const W=1280,H=720, ink='#111318', muted='#5E6673', blue='#1677B8', cyan='#6DCBF4', green='#149B6D', orange='#E9A000', panel='#F1F3F5', rule='#C8CDD4';
const deck=Presentation.create({slideSize:{width:W,height:H}});

function box(slide,x,y,w,h,fill='none',line='none',radius=false){return slide.shapes.add({geometry:radius?'roundRect':'rect',position:{left:x,top:y,width:w,height:h},fill,line:{style:'solid',fill:line,width:line==='none'?0:1}})}
function txt(slide,text,x,y,w,h,size=26,color=ink,bold=false,align='left'){const s=slide.shapes.add({geometry:'textbox',position:{left:x,top:y,width:w,height:h},fill:'none',line:{style:'solid',fill:'none',width:0}});s.text=text;s.text.style={fontSize:size,typeface:'Arial',color,bold,alignment:align,verticalAlignment:'middle'};return s}
function header(slide,title,n){txt(slide,title,42,30,1170,72,38,ink,true);txt(slide,String(n).padStart(2,'0'),1180,661,55,22,13,muted,false,'right');}
let noteIndex=0;
function note(slide,script,source='Thesis: A Hierarchical Time-Aware Transformer for Flow-Level Network Intrusion Detection, Xiaoyan Xiong, 2026.'){const finalScript=TALK[noteIndex++]?.script||script;slide.speakerNotes.textFrame.setText(finalScript+'\n\n[Sources]\n- '+source);slide.speakerNotes.setVisible(true)}
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
 const s=deck.slides.add();header(s,'Packets are units; flows are network conversations',2);
 await addImg(s,PACKET_FLOW,42,115,1195,350,'contain','Educational illustration showing one packet, an ordered packet sequence, and a bidirectional flow');
 txt(s,'PACKET',55,480,220,36,25,blue,true);txt(s,'One transmitted unit with header metadata and a data length.',55,520,340,64,20,ink);
 txt(s,'PACKET SEQUENCE',455,480,270,36,25,orange,true);txt(s,'Ordered packets reveal direction, bursts, and unequal time gaps.',455,520,350,64,20,ink);
 txt(s,'FLOW',875,480,220,36,25,green,true);txt(s,'Related packets grouped by endpoints and protocol; statistics summarize the whole conversation.',875,520,335,82,20,ink);
 box(s,55,615,1155,44,panel,'none',true);txt(s,'This thesis keeps both views: packet dynamics inside each flow, and statistical/context information around the flow.',70,619,1125,34,20,ink,true,'center');
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
 const s=deck.slides.add();header(s,'The thesis asks two questions at two traffic levels',4);box(s,42,130,565,190,'#EAF6FC','none',true);txt(s,'RQ1 — INTRA-FLOW',70,150,260,36,25,blue,true);txt(s,'Does time-aware packet-sequence modelling produce more informative flow representations than flow-level aggregation alone?',70,195,500,105,23,ink,true);box(s,632,130,606,190,'#E8F5EE','none',true);txt(s,'RQ2 — CROSS-FLOW',662,150,280,36,25,green,true);txt(s,'Does incorporating cross-flow context improve flow-level intrusion detection compared with independent per-flow classification?',662,195,530,105,23,ink,true);
 txt(s,'Research logic',42,365,250,40,29,ink,true);txt(s,'Packets → Stage 1 flow representation → Stage 2 historical context → final flow-level prediction',42,410,1160,40,24,blue,true);
 txt(s,'Four contributions',42,480,300,38,26,ink,true);bullets(s,['Aligned extraction and leakage-controlled preprocessing','Position + cumulative-time packet encoding','Bounded statistical conditioning and target-query attention','Chronological internal and external evaluation'],42,525,1160,20,39);
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
 const s=deck.slides.add();header(s,'Stage 1 — Intra-flow embedding learning',8);await addImg(s,ARCH,45,115,1190,490,'contain','Detailed two-stage model architecture');txt(s,'Stage 1 output',55,616,190,34,22,blue,true);txt(s,'One fixed 128-dimensional embedding per flow',245,616,390,34,22,ink,true);txt(s,'Key design',680,616,150,34,22,orange,true);txt(s,'Flow statistics condition packet tokens through a bounded residual.',830,616,390,34,21,ink,true);
 note(s,"This detailed architecture shows the complete pipeline. Stage 1 uses selected packet features, time-aware encoding, a separately encoded flow-statistics branch, a masked Transformer, and hybrid attention, mean, and max pooling. Flow statistics modulate packet tokens through a bounded residual. This prevents the aggregate vector from dominating the packet pathway. Stage 2 then builds a historical context, uses the target flow as the query, gates the retrieved summary, and predicts the final probability.","Thesis Figure: detailed architectural overview, created for the thesis.");
}
//9
{
 const s=deck.slides.add();header(s,'Stage 2 — Causal inter-flow context modelling',9);const xs=[65,225,385,545];['z(t-4)','z(t-3)','z(t-2)','z(t-1)'].forEach((v,i)=>{box(s,xs[i],190,115,66,'#E8F5EE',green,true);txt(s,v,xs[i]+8,200,99,44,22,ink,true,'center')});arrow(s,680,205,80,green);box(s,790,165,200,110,'#FFF1D6',orange,true);txt(s,'target z(t)\n= query',815,180,150,76,24,ink,true,'center');arrow(s,1010,205,80,blue);box(s,1110,175,115,90,'#EAF6FC',blue,true);txt(s,'final\nlogits',1125,188,85,60,21,ink,true,'center');
 txt(s,'Relation policies',65,335,250,36,25,green,true);const rel=[['Source host','same initiator'],['Destination host','same receiver'],['Endpoint','either endpoint'],['Time only','global recency']];rel.forEach((v,i)=>{txt(s,v[0],65+(i%2)*285,385+Math.floor(i/2)*70,155,30,21,ink,true);txt(s,v[1],65+(i%2)*285,416+Math.floor(i/2)*70,190,26,18,muted)});
 box(s,680,345,540,150,panel,'none',true);txt(s,'Eligibility and masking',710,365,260,34,23,blue,true);txt(s,'• earlier in stable start-time order\n• no future embeddings or historical labels\n• left padding excluded by the context mask',710,407,470,78,20,ink);
 box(s,680,520,540,88,'#FFF6E5','none',true);txt(s,'Scope: start-time ordered, but not yet strict flow-completion-time streaming.',705,537,490,54,21,ink,true,'center');
 note(s,"For each target flow, Stage 2 retrieves earlier embeddings according to a relation. The tested relations are same source host, same destination host, shared endpoint, and global chronological recency. The target embedding provides one query over the historical keys and values. A learned gate controls how much context updates the current representation. I use only earlier items in stable start-time order and never retrieve historical labels. However, this is not yet strict streaming causality, because a long flow that started earlier may not have completed when the target begins.");
}
//10
{
 const s=deck.slides.add();header(s,'Evaluation focuses on class balance and operational errors',10);const metrics=[['Macro-F1','Equal weight to class 0 and class 1'],['PR-AUC','Positive-class ranking across thresholds'],['Class-1 F1','Precision–recall balance at one threshold'],['FPR + FP / FN','False-alert rate and absolute errors']];metrics.forEach((m,i)=>{box(s,55+(i%2)*600,135+Math.floor(i/2)*175,560,130,i===0?'#EAF6FC':panel,'none',true);txt(s,m[0],80+(i%2)*600,152+Math.floor(i/2)*175,250,34,25,i===0?blue:ink,true);txt(s,m[1],80+(i%2)*600,194+Math.floor(i/2)*175,500,46,20,ink)});txt(s,'Model-selection protocol',55,505,300,36,25,blue,true);txt(s,'1  Select checkpoint by validation PR-AUC',55,550,350,32,21,ink);txt(s,'2  Select threshold by validation class-1 F1',450,550,390,32,21,ink);txt(s,'3  Evaluate the untouched chronological test set',860,550,365,32,21,ink);box(s,55,610,1170,44,panel,'none',true);txt(s,'Accuracy is secondary: an always-negative rule would score about 96.1% on the raw data and detect no positive flow.',70,614,1140,34,20,ink,true,'center');
 note(s,"Because positive flows are rare, accuracy is not the primary metric. I emphasise macro-F1, class-one F1, average precision, false-positive rate, and absolute false-positive and false-negative counts. Checkpoints are selected using validation average precision, while the operating threshold is selected using validation class-one F1. The final test partition is used only after these choices are fixed. In the thesis logs this ranking metric is named PR-AUC, but technically it is scikit-learn average precision.");
}
//11
{
 const s=deck.slides.add();header(s,'Stage 1 results — Position + time works best',11);await addImg(s,IMG+'/stage1_model_comparison.png',45,110,1190,500,'contain','Stage 1 architecture comparison');txt(s,'Position + time',55,620,220,34,22,blue,true);txt(s,'Macro-F1 0.8882  |  PR-AUC 0.8564  |  296 FP  |  306 FN',275,620,650,34,21,ink,true);txt(s,'vs. flow MLP',955,620,150,34,20,orange,true);txt(s,'+0.112 macro-F1',1090,620,145,34,19,ink,true);
 note(s,"The Stage 1 comparison uses a common controlled test manifest. The position-plus-time Transformer achieves the strongest overall balance, with macro-F1 of 0.8882 and average precision of 0.8564. It exceeds the flow-statistics MLP, recurrent models, convolutional models, and the encoding ablations. Time-only is stronger than position-only, while using both is strongest, supporting the claim that ordinal location and elapsed time are complementary.","Thesis results: Stage 1 model comparison on the common controlled test manifest.");
}
//12
{
 const s=deck.slides.add();header(s,'Stage 1 results — Separate statistical conditioning works best',12);await addImg(s,IMG+'/stage1_fusion_comparison.png',45,115,1190,485,'contain','Stage 1 fusion comparison');txt(s,'Scheme A',65,615,120,32,20,muted,true);txt(s,'repeat flow statistics',175,615,210,32,19,ink);txt(s,'Scheme B',420,615,120,32,20,muted,true);txt(s,'packet only',530,615,145,32,19,ink);txt(s,'Scheme C',720,615,120,32,21,green,true);txt(s,'macro-F1 0.8954 | PR-AUC 0.8763 | 267 FP | 293 FN',830,615,395,36,18,ink,true);
 note(s,"This ablation compares three ways to combine packet and flow information. Scheme A repeats the aggregate vector at every packet. Scheme B uses packets only. Scheme C encodes the aggregate vector once and uses bounded token modulation. Scheme C reaches macro-F1 of 0.8954 and average precision of 0.8763, while also reducing both false positives and false negatives. The careful conclusion is that a separate statistical pathway is useful; because parameter counts also change, the entire gain cannot be assigned to FiLM alone.","Thesis results: Stage 1 packet and flow-statistics integration ablation.");
}
//13
{
 const s=deck.slides.add();header(s,'Stage 2 results — Source-host history gives the best balance',13);await addImg(s,IMG+'/stage2_context_analysis.png',45,110,1190,500,'contain','Stage 2 relation comparison');txt(s,'Source host',55,620,180,34,22,green,true);txt(s,'macro-F1 0.9275  |  PR-AUC 0.9345  |  class-1 recall 0.8503  |  FPR 0.5429%',235,620,980,34,20,ink,true);
 note(s,"For Stage 2, the source-host relation gives the strongest combined result. It has the highest macro-F1 and average precision and the lowest false-positive rate among the tested relations. Endpoint context recovers more positive flows, but it also generates more false positives. Global time-only context is less selective. This does not prove that source history is optimal for every attack family; binary weak labels cannot identify which attack mechanisms benefit.","Thesis results: Stage 2 context-relation ablation at window size 128.");
}
//14
{
 const s=deck.slides.add();header(s,'Stage 2 results — Target-query attention reduces errors',14);await addImg(s,IMG+'/stage2_architecture_comparison.png',45,110,1190,500,'contain','Stage 2 architecture comparison');txt(s,'Proposed model',55,620,210,34,22,blue,true);txt(s,'macro-F1 0.9275  |  class-1 F1 0.8610  |  PR-AUC 0.9345',265,620,650,34,20,ink,true);txt(s,'vs. vanilla Transformer',930,620,210,34,19,orange,true);txt(s,'−177 FP, −117 FN',1130,620,110,34,17,ink,true);
 note(s,"The architecture comparison fixes the Stage 1 embeddings, context relation, window, objective, and evaluation protocol. The proposed target-query model obtains macro-F1 of 0.9275, class-one F1 of 0.8610, and average precision of 0.9345. It reduces both error types compared with the vanilla Transformer. Nevertheless, the no-context head is much smaller, and other baselines are not parameter matched. The result supports the integrated architecture, but it is not a pure causal estimate of historical context alone.","Thesis results: Stage 2 architecture comparison on the common full-data test manifest.");
}
//15
{
 const s=deck.slides.add();header(s,'External evaluation — Transfer varies across windows',15);txt(s,'CICIDS2017 Wednesday: five non-overlapping 20,000-flow windows',55,120,650,34,23,blue,true);
 const rows=[['Window','Positive','Macro-F1','PR-AUC'],['w01','18.195%','0.9125','0.9088'],['w02','13.515%','0.9226','0.9123'],['w06','16.480%','0.8536','0.8906'],['w07','13.340%','0.8640','0.8419'],['w08','15.235%','0.9220','0.9397'],['Pooled','15.353%','0.8966','—']];
 txt(s,'Company cross-server',760,140,380,38,25,green,true);txt(s,'38,513 flows',760,200,250,32,21,muted);txt(s,'3.285%',760,250,210,68,48,orange,true);txt(s,'positive prevalence',970,266,230,34,20,muted);txt(s,'0.9026',760,345,230,68,48,green,true);txt(s,'macro-F1',970,361,190,34,20,muted);txt(s,'0.8771',760,440,230,68,48,blue,true);txt(s,'PR-AUC',970,456,190,34,20,muted);box(s,735,535,480,100,panel,'none',true);txt(s,'Interpretation',760,548,160,30,21,blue,true);txt(s,'Transfer is promising, but temporal and environmental shift changes the error balance.',760,579,420,46,20,ink,true);
 const t=s.tables.add({left:55,top:170,width:650,height:390,rows:rows.length,columns:4,values:rows,columnWidths:[145,165,165,175]});t.styleOptions={headerRow:true,bandedRows:true};t.cells.block({row:0,column:0,rowCount:1,columnCount:4}).assign({fill:'#EAF6FC',textStyle:{bold:true,fontSize:18,color:ink}});t.cells.block({row:1,column:0,rowCount:6,columnCount:4}).assign({textStyle:{fontSize:17,color:ink},borders:{style:'solid',fill:rule,width:1}});
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
