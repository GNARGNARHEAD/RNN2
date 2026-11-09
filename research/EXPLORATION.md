# Exploratory Research Connections

Reviewed 4 November 2026. This pass follows connections beyond the initial architecture, tutoring, and interpretability survey. The aim is to find useful questions and simple principles. None of the mechanisms below is an adopted project requirement.

The student will start without pretrained weights. Where these papers rely on substantial prior training, that dependence remains part of their evidence limits; it cannot be treated as an ability already available to our student.

## Detect a developmental change before deciding what it means

Developmental interpretability studies follow networks across training checkpoints. Hoogland and colleagues estimate the local learning coefficient, a measure of loss-landscape geometry, in small language and regression transformers. Changes in that measure aligned with independently investigated changes in computation, including n-grams, induction, and regression specialization. Some later development reduced robustness. A detected transition therefore needs interpretation; it is not automatically an improvement. The language analysis across checkpoints cost about 200 A100 GPU-hours, so its published measurement procedure is substantial. [Hoogland et al., 2024–2025](https://arxiv.org/html/2402.02364v3).

A follow-up examines individual attention heads' developmental trajectories. It also cautions that true local learning coefficients are unknown for these models and that estimation assumptions are not generally satisfied during training. [Wang et al., October 2024](https://arxiv.org/html/2410.02984).

**Question for RNN2:** can we notice an interesting change without having named the skill in advance? Saved checkpoints, repeated observations, and focused follow-up experiments support that question. The local learning coefficient is an optional research instrument rather than a ready-made measure of cognitive development.

## A lighter way to notice representation changes

Centered Kernel Alignment compares representations on the same examples. Linear CKA tolerates neuron permutations, orthogonal rotations, and uniform scaling, making it more useful than comparing raw coordinates when a representation changes basis. It does not cover every function-preserving transformation. [Kornblith et al., ICML 2019](https://proceedings.mlr.press/v97/kornblith19a.html).

Counterexamples show that outliers and transformations preserving useful information can substantially change CKA; model behaviour can also remain similar while the metric changes. [Davari et al., 2022](https://arxiv.org/html/2210.16156v2).

**Possible use:** collect a small sample of activations for fixed inputs at selected checkpoints. A change could tell us where to investigate predictions and intervention effects. This detects differences in representation geometry, not semantic progress, and is not a substitute estimator for the local learning coefficient. Keep input histories, tokenization, evaluation settings, and measurement locations comparable.

## Let experience reveal what is learnable now

Intelligent Adaptive Curiosity selects experiences where prediction is improving. Robot experiments produced changing activities while avoiding both mastered situations and unlearnable noise. Progress was estimated within comparable situations; pooling unrelated difficulty levels could mistake a switch to easier activity for learning. The underlying learners were local predictors rather than language transformers. [Oudeyer, Kaplan and Hafner, 2007](https://www.pyoudeyer.com/ims.pdf).

Enhanced POET jointly evolves environments and agents and transfers solutions between them. Its controlled comparisons found that direct optimization and a smooth curriculum missed some solutions reachable through more varied paths. This concerns specialized locomotion policies and very large CPU budgets, not one general learner. [Enhanced POET, ICML 2020](https://proceedings.mlr.press/v119/wang20l/wang20l.pdf).

**Question for RNN2:** which experiences are becoming useful, and which unexpected successes might become stepping stones? A small record of surprising successes and occasional transfer checks could preserve such opportunities without implementing an automated environment-evolution system.

## Teach reusable building blocks and then withdraw assistance

DreamCoder alternates solving tasks, consolidating recurring program fragments into a reusable library, and training a neural search guide. Expertise resides jointly in explicit functions and neural parameters. [DreamCoder, 2020](https://arxiv.org/html/2006.08381v1).

LAPS adds language guidance. Its no-language test is particularly relevant: learned libraries and their search priors improved held-out program synthesis even when test-time language and neural search were removed. A complete neural search system also benefited, but that comparison does not isolate changes in neural representations from the better library. These results concern bounded string, graphics, and scene tasks and multi-core search. [LAPS, ICML 2021, Table 1](https://arxiv.org/html/2106.11053v3).

**Question for RNN2:** can teaching a shared operation behind several mistakes create something the student later reuses independently? We could compare varied examples and near misses with ordinary answer demonstrations, then withdraw the explanation and test new combinations. An explicit program library is one possible future mechanism; these papers do not establish that the same process occurs inside language-model weights.

## Distinguish contextual adaptation from lasting learning

Test-Time Training layers use a small model's weights as recurrent state. Incoming tokens update that state, while outer training learns the initialization and update-related machinery. Separate sequences start from a shared initialization. [TTT, July 2024](https://arxiv.org/html/2407.04620v1).

Titans also has adaptive contextual memory, but the component it calls persistent memory is input-independent and fixed at test time. The terminology alone does not establish that an experience produces a lasting new skill. [Titans, 2025](https://arxiv.org/html/2501.00663v1).

**Question for RNN2:** where does an acquired ability reside, how long does it persist, and what restores or clears it? A useful system could operate at several timescales, while our records distinguish live state, reusable stored experience, and durable parameter changes.

## The learning objective matters as much as the storage mechanism

Cartridges trains reusable KV memory while holding model weights fixed. Its experiments show that strong corpus memorization can coexist with poor answers to unfamiliar queries; varied conversation data and context distillation improve transfer. The reported recipe uses substantial accelerator resources. [Cartridges, June 2025](https://arxiv.org/html/2506.06266).

Nested Learning studies components with different update frequencies. Some of its continual-learning experiments adapt existing language models with 15 billion tokens of further pretraining. It raises relevant questions about consolidation but does not provide a lightweight, ready-made developmental engine. [Nested Learning, 2025](https://arxiv.org/html/2512.24695).

**Implication:** test whether the student can use experience in new situations, regardless of whether it was stored in weights, a context state, or an external artifact. Those locations should remain visible when attributing capability to the student.

## Small parameter counts and favourable scaling can hide costs

TNT investigates the poor hardware utilization and update-granularity tradeoffs of TTT and Titans. Its optimizations substantially improve their measured training efficiency, but an optimized Transformer remains faster in its reported time-to-target-loss comparison. The experiment uses 150M models on a TPU pod. This is a concrete reason to measure the whole operation on our hardware. [TNT, ICLR 2026](https://arxiv.org/html/2511.07343).

Tiny Recursive Message Passing illustrates another cost: wiring affects how many recurrent steps gradients must traverse to reach relevant decisions. Its puzzle-specific gains also involve substantial inference procedures. Small networks do not by themselves establish cheap learning or general capability. [October 2026 preprint](https://arxiv.org/html/2610.18649v1).

**Implication:** compare useful progress per time and memory, including context length, repeated inference, update frequency, and tutor effort. The best parameter count remains an empirical question.

## Vary the surface while preserving the relationship

Lake and Baroni train transformers across instruction-learning episodes with changing symbol meanings and compositions. The resulting models infer new mappings from examples, but their few-shot tests follow extensive meta-training and use fixed weights at test time. The authors distinguish this from a model of developmental acquisition. [Nature, October 2023](https://www.nature.com/articles/s41586-023-06668-3).

TinyStories shows that carefully generated, restricted-language stories can support coherent text in very small models. Its deliberate variation matters, while its teacher-generated data, external evaluation, and accelerator training costs limit conclusions about independent local development. [TinyStories, May 2023](https://arxiv.org/html/2305.07759).

**Question for RNN2:** can the tutor make a relationship easier to discover by varying names, presentation, and examples while retaining the underlying rule? This suggests richer practice and transfer tests without requiring a prescribed human developmental ladder.

## What is worth carrying forward

The strongest new direction is to study a developmental trajectory: preserve interesting moments, detect changes, and investigate their meaning. Curiosity and abstraction-learning research add questions about which experiences are useful and what becomes reusable after assistance is withdrawn.

The mechanisms reviewed here remain options. The first local system can support these questions through modest observation, saved checkpoints, varied teaching, and reversible interventions. More elaborate memory, interpretability, or curriculum machinery should answer a particular question that the running learner exposes.

## Depth, width, and Ripple — 5 November 2026

Ricursive Intelligence's Ripple-65B Preview uses 80 distinct layers and residual width 5,120. The latter is the dimension of each token's hidden state, not a parameter count. Its internal attention width is 10,240 (80 query heads of dimension 128), with 8 KV heads; its gated FFN intermediate width is 40,960. It is therefore narrow in its residual stream while doing substantially wider computation inside each block. The weights are not shared across decoder layers. [Official configuration](https://huggingface.co/ricursiveAI/Ripple-65B-Preview/resolve/main/config.json), [implementation](https://huggingface.co/ricursiveAI/Ripple-65B-Preview/resolve/main/modeling_ricursive.py).

The launch report combines architecture, normalization, information-flow, data, and post-training changes. Its headline efficiency comparison does not isolate depth as the explanation. Particularly useful is its willingness to discard mechanisms that failed to help at the intended scale: two-stream mHC helped smaller experiments but not its 4B test, while extra normalization became useful at that scale. [Ricursive's report](https://www.ricursive.com/ripple).

Controlled depth experiments at 41M, 134M, and 374M parameters found initial gains followed by diminishing returns or deterioration. Crucially, they held residual width and attention dimensions fixed and traded depth against FFN width. Their results do not prescribe our residual-width/depth ratio; their substantial pretraining also differs from our developmental setting. [Petty et al., NAACL 2024](https://arxiv.org/html/2310.19956v2).

One relevant lead from Ripple is Value Residual Learning: later attention layers receive a mixture of their own value vectors and first-layer values. Its separate paper includes 82M–468M experiments, making preservation of early information through depth a concrete question at our scale. The tests concern language-model loss under extensive training, not cumulative tutored development. [Primary paper](https://arxiv.org/pdf/2410.17897).

**Implications:** distinguish residual width, attention width, FFN width, and computational depth. More layers supply successive transformations, not a guaranteed count of reasoning steps. Residual width is neither a concept count nor the total knowledge capacity of a model: weights and multiple token states also matter. There is no established numerical minimum for a general world representation without specifying tasks, precision, context, accuracy, and computational resources.

For our existing full-attention, 4x GELU-FFN probe, body parameters are approximately `12 * layers * width^2`. This formula does not describe Ripple's wider gated blocks. Candidate shapes near 100M total parameters, using our current vocabulary of 256 and sequence length 128, are:

| Residual width | Layers | Calculated parameters |
|---:|---:|---:|
| 1,024 | 8 | 101,402,880 |
| 768 | 14 | 99,691,264 |
| 512 | 32 | 101,156,096 |
| 384 | 56 | 99,552,256 |

These are arithmetic candidates, not measured training results or a selected student. More layers introduce sequential execution and can increase saved activation memory even at similar parameter counts. A useful comparison would keep the block design, vocabulary, and educational conditions comparable, use initialization suitable for the depth, and measure progress per example and per elapsed time, retention, transfer, memory, and thermals. Residual width 512 or 768 is a practical hypothesis to investigate, not a theoretical lower bound.

The broader lesson for RNN2 is to combine appropriate information flow and stable learning with experiences that respond to the student's development. Each mechanism should earn its place through a controlled comparison at the intended student scale.

## Delayed memory bus across depths — 5 November 2026

The proposed architecture samples internal hidden representations at several depths, perhaps five sections, and exchanges compressed information through a persistent bus. Each section reads an earlier snapshot and writes information for a later tick. The aim is continuity across an ongoing stream of experience, including access by early processing to prior deep processing. Improved stability and useful contextual organization are hypotheses to test.

Feedback Transformer is a close precedent: it combines representations across layers into memory for earlier time steps, accessible from every layer. Its memory retains a history of vectors rather than five compact slots. The authors explicitly discuss the resulting sequential computation and loss of training parallelism. [Primary paper, section 3.2](https://arxiv.org/pdf/2002.09402).

Recurrent Memory Transformer separates previous-segment read memory from current-segment writes and trains through multiple segments. Its experiments show that longer gradient windows can be useful but consume more GPU memory. It supplies a useful training/causality reference, although its memory tokens sit at segment boundaries rather than at several internal depths. [Primary paper, section 3](https://arxiv.org/pdf/2207.06881).

Transformer-XL carries prior hidden states while stopping their gradients. This distinguishes the lifetime of numerical context from the lifetime of the training graph. Its ordinary hidden-state producers also participate in current losses; that does not show that a dedicated bus writer can learn when every write is immediately detached. [Primary paper, section 3.2](https://arxiv.org/pdf/1901.02860).

**Possible minimal design:** give each tap its own small state slot, learned write projection, and access to learned reads from the previous bus. Freeze the read buffer for a tick, collect writes into a separate next-state buffer, and publish that state on the next tick. Gates and gradual updates make feedback strength and persistence controllable. Separate writer slots retain source identity and avoid blindly averaging representations from different depths. Five slots of 64 FP32 values would occupy 1.25 KiB per stream for one state buffer; projection parameters and training activations are additional costs. These numbers are illustrative, not selected dimensions.

**Timing:** a tick must be explicit. Per-token updates maximize responsiveness but introduce sequential dependencies across training tokens. Per-chunk updates preserve parallelism within a chunk and are a practical first hypothesis for this GPU. A whole-chunk summary may influence only later chunks, never earlier predictions in the same chunk. The delay removes a same-tick circular dependency; it does not parallelize the main stack or make recurrence free.

**Training:** if a write affects only a later loss, backpropagation must connect that later use to the write. An initial experiment should unroll at least two ticks and compare longer bounded windows as memory permits. Detach the graph at window boundaries while carrying the numerical bus onward. The bus can persist across teaching blocks and tutor reviews. Use coherent streams and separate state per stream; preserve state with checkpoints. Weight updates and direct edits can change how stored representations are interpreted, so continuity through those changes needs examination.

**Evaluation:** compare no bus, last-layer feedback, and distributed feedback under comparable memory and teaching budgets. Test use of history, resistance to distraction, correction when context changes, finite/recoverable dynamics, and durable learning after a diagnostic state reset. Shuffled or frozen bus controls can help establish whether useful communication is occurring. Keep contextual memory and persistent parameter learning distinct. This remains an architecture candidate, not a requirement or a demonstrated stability mechanism.
