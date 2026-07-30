# Thesis Editorial and Redundancy Audit

## 1. Executive assessment

The thesis has a coherent research story and enough experimental breadth for an
MSc thesis, but the same story is currently explained too many times. The main
problem is not repeated terminology, which is unavoidable, but repeated
*argumentative work*: several chapters define the same two stages, explain the
same causal context, state the same training controls, or answer the same
research questions.

The active thesis contains approximately 20,600 English word tokens. The most
important structural warning is that **Problem Formulation and System Overview**
contains 26 displayed equation/align environments, while **Proposed
Methodology** contains another 30. A problem-formulation chapter should define
the task, information available at prediction time, notation, causal constraint,
and output. It should not pre-implement almost the entire architecture.

Recommended target: remove roughly 2,500--3,500 words from repeated explanation
and use part of the recovered space for deeper discussion. The final thesis
would become shorter but intellectually stronger.

## 2. Chapter ownership model

Each claim should have one primary home.

| Chapter | Content it should own | Content it should not repeat |
|---|---|---|
| Introduction | problem motivation, research gap, RQs, contributions, one-paragraph approach, roadmap | full pipeline, layer details, complete metric justification |
| Related Work | critical comparison with prior methods and final research gap | repeated descriptions of the proposed implementation |
| Problem Formulation and System Overview | task, scope, information units, notation, causal constraints, abstract two-stage mapping | pooling equations, focal loss, exact context implementation, hyperparameters |
| Dataset Construction and Preprocessing | provenance, extraction, labels, schemas, predictors, transformations, alignment, split | model architecture and experimental baseline definitions |
| Proposed Methodology | the proposed architecture and only the equations needed to reproduce it | baseline catalogues, exact optimizer schedules, metric definitions |
| Experimental Methodology | baselines, ablations, fixed controls, hyperparameters, checkpoint/threshold rules, metrics, hardware/timing protocol | raw schemas, model equations, repeated context definitions |
| Results and Discussion | empirical observations, mechanisms, alternatives, operational implications, validity of each inference | training procedure and metric definitions |
| Conclusion and Future Work | concise answers, overall boundary, two or three future directions | another full result table in prose or a second contribution list |

## 3. Problem Formulation versus Proposed Methodology

### Diagnosis

There is substantial overlap.

- `main.tex:499--590` already defines packet projection, positional/time
  encoding, Transformer processing, attention pooling, flow encoding, late
  fusion, and the Stage 1 classifier.
- `main.tex:1636--1965` defines the same Stage 1 process in implementation-level
  detail.
- `main.tex:591--717` defines all four context relations, recent-history
  selection, sequence construction, masking, the Stage 2 Transformer, and the
  no-context control.
- `main.tex:1966--2209` defines those relations and the Stage 2 architecture
  again.
- `main.tex:718--794` gives focal loss, sequential optimization, and validation
  threshold selection; focal loss is repeated at `main.tex:2210--2250`, while
  thresholding and model selection belong to the experimental protocol.
- The RQ mapping table at `main.tex:795--827` overlaps with the more useful
  experimental matrix at `main.tex:2531--2567`.

### Recommended correction

Keep the chapter, but make it genuinely abstract.

1. Retain `Detection Setting and Design Requirements` and reduce the six
   requirements to four compact constraints: hierarchical input, explicit time,
   causal/payload-independent information, and leakage-aware evaluation.
2. Retain one high-level system figure. It is justified because it covers PCAP
   extraction through prediction; the methodology figure should cover only
   model internals.
3. Retain the notation table.
4. Replace the detailed Stage 1 section with one mapping:

   `z_i^(1) = F_theta1(X_i, tau_i, m_i, s_i)`.

   State only that Stage 1 maps packet dynamics and flow statistics to one flow
   embedding. Do not define attention pooling or feature modulation here.
5. Replace the detailed Stage 2 section with one causal set constraint and one
   mapping:

   `p_i = G_theta2(z_i^(1), C_i)`, where every `j in C_i` precedes target `i`.

   Move the four relation definitions and recent-window operation exclusively
   to Proposed Methodology.
6. Delete focal-loss details and threshold search from this chapter. A generic
   supervised objective is enough. Keep focal loss in Proposed Methodology if it
   is considered part of the method; put its exact parameters and threshold rule
   only in Experimental Methodology.
7. Delete the RQ mapping table. Replace it with two sentences referring forward
   to the experimental matrix.
8. Reduce the chapter summary to a two-sentence transition, or remove it after
   the above simplification.

This should reduce the chapter from about 2,340 words to approximately
1,300--1,600 words and from 26 displayed equation environments to roughly
5--8.

## 4. Repetition across the remaining chapters

### Introduction

The Introduction is broadly sound, but the two approach paragraphs explain the
complete Stage 1 and Stage 2 pipeline too precisely. Keep one high-level
paragraph stating the two-stage idea and remove implementation terms such as
the complete list of context methods and detailed late-fusion behavior. Those
belong later.

The final metric paragraph also repeats Experimental Methodology and the Results
opening. Replace it with one sentence: because the task is imbalanced, the
evaluation prioritizes macro-F1, PR-AUC, class-1 performance, and FPR over
accuracy alone.

The roadmap at `main.tex:95` says "next section" even though the document uses
chapters. Rewrite it with chapter references and the actual chapter order.

### Related Work

The final `Summary and Research Gap` is necessary and should remain. However,
several preceding sections end with a paragraph beginning "The present work..."
or explaining how the thesis differs. Keep local criticism of prior work, but
move most proposed-method positioning into the final gap section. This will
make Related Work read as a critical synthesis instead of repeatedly announcing
the method.

The comparison table is useful, but `\begin{table}[t][H]` at `main.tex:154`
contains two placement arguments. Use one, preferably `[H]` only if strict
placement is essential; otherwise use `[tbp]`.

### Dataset Construction and Preprocessing

This chapter has a clear role and contains less harmful repetition than the
others. Retain the provenance, label limitations, schema, leakage control, and
Stage 1 artifact interface. The Stage 1 artifact section is justified because
it defines a data contract, not a second model description.

The final validity section is also appropriate here because the limitations
arise from label and capture construction. In the Conclusion, refer back to it
and summarize only the two most important boundaries instead of repeating all
four.

### Proposed Methodology

This chapter should become the single source of truth for all model equations.
Keep the cumulative-time encoding, bounded token-FiLM, masked Transformer,
hybrid pooling, relation-specific causal context, context-age encoding, and
target-query gate.

Move `Stage 2 Control Models` (`main.tex:2179--2208`) to Experimental
Methodology. Baselines are experimental controls, not parts of the proposed
model.

Keep the focal-loss equation if imbalance-aware training is claimed as part of
the method, but remove exact learning rates, schedules, patience, batch size,
and clipping from `main.tex:2252--2273`. Those values already appear in the
Experimental Methodology table.

Split `Principal Configuration`: retain architecture rows here and retain
optimization rows only in Experimental Methodology. At present both tables
repeat batch size, learning rate, weight decay, focal gamma, sampler fraction,
checkpoint criterion, and seed.

The Methodological Summary can remain as one short transition paragraph. It
should state the two novel mechanisms and point to the evaluation chapter; it
should not repeat the whole information path.

### Experimental Methodology

This chapter opens by saying that it avoids duplication, but then repeats:

- chronological splitting and train-fitted transforms from the Dataset chapter;
- Stage 2 context causality from Proposed Methodology;
- Stage 2 baseline descriptions from Proposed Methodology;
- optimizer and sampling values from Proposed Methodology;
- metric motivation already stated at length in the Introduction and Results.

Recommended changes:

1. Compress `Dataset Partition and Leakage Controls` to two paragraphs that
   reference the Dataset chapter and state only the invariants required for fair
   comparison.
2. Keep the Experimental Matrix. It is the clearest organization of RQ1/RQ2
   evidence and should replace the earlier RQ mapping table.
3. Define all Stage 1 and Stage 2 baselines only here.
4. Keep exact optimizer, sampler, early-stopping, checkpoint, threshold, and
   random-seed settings only here.
5. Keep metric definitions only here. Reduce metric discussion in the
   Introduction and Results opening to cross-references.
6. Add an active threshold-selection subsection. The current title includes
   thresholding, but the active text does not state the complete search rule.
   The thesis must identify whether each experiment maximizes class-1 F1,
   macro-F1, or another validation metric.
7. End the chapter with one transition sentence to Results. The existing
   chapter summary is commented out, so the active chapter currently ends at the
   environment table.

## 5. Chapter summaries

Not every chapter needs a formal `Chapter Summary` section.

- **Related Work: keep.** `Summary and Research Gap` performs essential
  synthesis and motivates the method.
- **Problem Formulation: remove or reduce.** The system figure and experimental
  mapping already summarize it.
- **Dataset: no extra summary required.** The validity paragraph already closes
  the chapter.
- **Proposed Methodology: keep one short paragraph.** It can bridge architecture
  to evaluation.
- **Experimental Methodology: no full summary required.** Add one or two
  transition sentences after the reproducibility/efficiency protocol.
- **Results and Discussion: remove the current Chapter Summary.** It repeats the
  integrated findings and is immediately followed by Conclusion.
- **Conclusion: Closing Remarks may remain**, but it should be short and should
  not function as a fourth repetition of the results.

## 6. Contributions in the Conclusion

A full contribution list is not necessary in both Introduction and Conclusion.
The Introduction is the correct place for the definitive contribution list.
The current Conclusion repeats the same pipeline, Stage 1, Stage 2, and
evaluation contributions.

Recommended option: delete the standalone `Principal Contributions` section at
`main.tex:3679--3715` and replace it with one synthesis paragraph after the RQ
answers. That paragraph should explain the combined intellectual contribution:
the thesis demonstrates a hierarchical representation in which explicit
intra-flow time and causal inter-flow context contribute separable evidence.

If the university rubric explicitly asks for contributions in the final
chapter, keep the heading but replace the four-item list with a three- or
four-sentence evidence-based synthesis. Do not reproduce the Introduction list.

## 7. Results and Discussion

### Current balance

The chapter is empirically thorough, but numerically dense. Many paragraphs
repeat six to ten values already printed in figures. The discussion is not
absent: the FPR/recall trade-off, non-monotonic sequence length, context noise,
and target-query inductive bias are useful interpretations. The problem is that
these interpretations are often only the final sentence after a long metric
inventory.

### Editing rule

For each experiment, use this four-part structure:

1. **Observation:** identify the winner or trade-off using one primary metric
   and one operational error measure.
2. **Mechanism:** explain the most plausible model/data reason.
3. **Implication:** state what this means for NIDS operation or the RQ.
4. **Boundary:** name the strongest alternative explanation or missing control.

Exact secondary metrics should remain in the figure/table rather than being
repeated in prose.

### Specific improvements

- At `main.tex:3179--3196`, reduce the metric inventory. Retain the macro-F1 and
  PR-AUC gain over the strongest baseline plus the FP/FN trade-off. Move the
  other exact values to a compact appendix table if needed.
- Deepen the time-encoding discussion by distinguishing ordering from physical
  time and by noting that the CNN result shows threshold performance and score
  ranking can move in opposite directions.
- For Scheme C, acknowledge that token-FiLM changes both conditioning and model
  capacity. Add parameter-matched late-concatenation and FiLM-without-late-fusion
  controls before claiming that the modulation mechanism itself causes the
  gain.
- For sequence length, report performance by original flow length. Otherwise it
  is unclear whether `L=256` hurts because of stale packets, different truncation
  rates, optimization, or random variation.
- For Stage 2 relations, the current source-host interpretation is plausible but
  not attack-specific evidence. Explain the hypothesis: source histories can
  preserve repeated outbound behavior, destination histories can mix unrelated
  clients of a popular service, and endpoint context can increase recall while
  importing heterogeneous flows. State that binary labels cannot verify these
  attack-family mechanisms.
- For window size, report the observed context-length distribution. A nominal
  `W=256` may add old flows, or it may consist mostly of padding. These imply
  different mechanisms.
- Keep the no-context result. Do not remove it. Label it as a lower bound, and
  add a capacity-matched masked-history ablation. Until then, the Stage 2 gain is
  evidence for the complete context model, not a perfectly isolated causal
  estimate of context alone.
- Keep the efficiency figure. If all Stage 2 profiles were obtained on L4,
  report that directly. Do not claim that 0.1965 ms is faster than 0.1967 ms;
  claim latency parity with vanilla Transformer and meaningful reductions only
  against the recurrent models. Repeat timed inference with warm-up and report
  mean/dispersion.
- Add a cross-stage error-transition analysis using aligned predictions:
  Stage-1 FN corrected by Stage 2, Stage-1 FP corrected, and new errors
  introduced. This explains *how* context changes decisions better than another
  list of aggregate metrics.
- Add performance conditioned on context availability or context length. This
  tests whether the gain is actually concentrated among flows with useful
  history.
- Restore a concise visible reliability paragraph. The current single-seed
  limitation is commented out in Results but appears later in Conclusion.

### Recommended Results ending

1. Keep Stage 1 and Stage 2 experiment blocks.
2. Remove `Stage 1 Answer to RQ1` because RQ1 is answered again later.
3. Rename `Integrated Findings and Research Questions` to `Cross-Stage
   Discussion` and expand it around mechanisms, operational implications, and
   evidence boundaries.
4. Remove `Chapter Summary`.
5. In Conclusion, answer each RQ in one compact paragraph with only the decisive
   effect size.

## 8. Paragraph and chapter transitions

The local paragraph transitions are generally readable, but chapter transitions
need tightening.

- Introduction roadmap: use `Chapter~\ref{...}` rather than "next section".
- Related Work to Problem Formulation: the research-gap section already provides
  a good bridge; no extra summary is needed.
- Problem Formulation to Dataset: use a two-sentence transition stating that the
  next chapter instantiates the defined variables from PCAP.
- Dataset to Methodology: the current final paragraph is acceptable, but its
  final sentence can point specifically to the model inputs defined next.
- Proposed Methodology to Experimental Methodology: end the methodological
  summary with "The next chapter evaluates these components through controlled
  baselines and ablations."
- Experimental Methodology to Results: add an active final sentence; currently
  the source jumps from the environment table to Results because the remainder
  is commented.
- Results to Conclusion: remove the Results chapter summary so that the
  cross-stage discussion leads directly into the final synthesis.

At paragraph level, avoid repeated openers such as "This chapter reports..."
followed by another paragraph restating scope. A useful transition should state
the logical dependency between sections, not merely announce the next heading.

## 9. Submission-critical technical issues

These issues are more urgent than stylistic redundancy.

1. **Abstract is unfinished.** `main.tex:39--42` contains template instructions
   rather than an abstract. The keyword list at line 47 contains the typo
   `Machine leanring` and should use research concepts, not baseline model names.
2. **Visible corrupted text.** `main.tex:2331` contains un-commented mojibake
   after "context relation" and will appear in the document.
3. **Malformed equation text.** `main.tex:1574` contains `j_k<i,quad`; it should
   be `j_k<i,\quad`.
4. **Invalid table placement.** `main.tex:154` uses `[t][H]`; retain only one
   placement option.
5. **Stage 2 main-results table is inconsistent.** At `main.tex:3347--3356`, the
   column specification contains a confusion-matrix column, the header omits its
   title, the Stage 1 row contains a confusion matrix, and the Stage 2 row does
   not. Add `TN/FP/FN/TP` to the header and `80,421/439/527/2,993` to the Stage 2
   row, or remove that column from both rows.
6. **Environment macros are incorrectly duplicated.** `\ExpGPU` and related
   macros are first defined as placeholders at `main.tex:2421--2425` and then
   defined again with `\providecommand` at `main.tex:2438--2444`.
   `\providecommand` does not overwrite an existing definition, so the first
   placeholder wins. Keep one definition or use `\renewcommand` after the first.
7. **Seven active citations are unresolved in `sources.bib`:**
   `Fawcett2006ROC`, `GoogleColabFAQ2026`, `NVIDIAL4ProductBrief`,
   `NVIDIAT4Datasheet`, `Pedregosa2011ScikitLearn`,
   `Pineau2021Reproducibility`, and `Dong2025PretrainingEncrypted`. The first six
   exist in `experimental_methodology_additions.bib` but were not merged into
   `sources.bib`. The last appears to be a key mismatch with the existing
   `Dong2025EncryptedPretrain` entry.
8. **Source cleanup.** The file contains 440 fully commented lines, including
   abandoned sections and mojibake notes. They do not create PDF repetition but
   make the final source harder to review. Move useful notes to a separate
   development file and delete obsolete blocks from the submission source.
9. **Minor wording.** `Its Metrics are` at `main.tex:3196` should be lower-case
   `its metrics are`, but the sentence should preferably be shortened as part of
   the Results edit.

## 10. Recommended revised structure

1. Introduction
   - Motivation and problem
   - Research gap
   - RQ1 and RQ2
   - Contributions
   - Thesis organization
2. Related Work
   - Existing method families
   - Comparative table
   - Synthesis and gap
3. Problem Formulation and System Overview
   - Task and scope
   - Design constraints
   - High-level system
   - Notation and abstract two-stage mapping
4. Dataset Construction and Preprocessing
5. Proposed Methodology
   - Stage 1 model
   - Stage 2 model
   - Sequential interface and loss
   - Short method summary
6. Experimental Methodology
   - Data/split invariants by reference
   - Experimental matrix and baselines
   - Training and threshold protocol
   - Metrics
   - Hardware and timing
7. Results and Discussion
   - Stage 1 evidence and discussion
   - Stage 2 evidence and discussion
   - Cross-stage discussion and validity
8. Conclusion and Future Work
   - Compact RQ answers
   - Overall scope
   - Pre-training
   - Relation-aware context
   - Validation/deployment

## 11. Recommended editing order

1. Fix submission-critical errors and the abstract.
2. Reduce Problem Formulation and establish the chapter-ownership boundaries.
3. Remove baseline and optimizer duplication from Proposed/Experimental
   Methodology.
4. Rewrite the Results prose using observation--mechanism--implication--boundary.
5. Remove repeated RQ summaries and the duplicate contribution list.
6. Perform a final transition, citation, label, and compilation audit.

Do not begin by polishing individual sentences. Structural ownership should be
fixed first; otherwise the same paragraph will be polished and later deleted.
