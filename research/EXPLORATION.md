# Exploratory Research Connections

Reviewed 4 November 2026. This pass follows connections beyond the initial architecture, tutoring, and interpretability survey. The aim is to find useful questions and simple principles. None of the mechanisms below is an adopted project requirement.

Applicability updated 14 November 2026: the active target is approximately 20B parameters on one B200, with persistent cloud checkpoints. The local probe and its proposed follow-up experiments are retired. Research mechanisms below remain optional; follow the [B200 implementation plan](../experiments/2026-11-14-b200-plan.md) for current work. Small-model findings do not establish results at the new target scale.

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

TinyStories shows that carefully generated, restricted-language stories can support coherent text in very small models. Its deliberate variation matters, while its teacher-generated data, external evaluation, and accelerator training costs limit conclusions about independent cumulative development. [TinyStories, May 2023](https://arxiv.org/html/2305.07759).

**Question for RNN2:** can the tutor make a relationship easier to discover by varying names, presentation, and examples while retaining the underlying rule? This suggests richer practice and transfer tests without requiring a prescribed human developmental ladder.

## What is worth carrying forward

The strongest new direction is to study a developmental trajectory: preserve interesting moments, detect changes, and investigate their meaning. Curiosity and abstraction-learning research add questions about which experiences are useful and what becomes reusable after assistance is withdrawn.

The mechanisms reviewed here remain options. The B200 student can support these questions through modest observation, saved checkpoints, varied teaching, and reversible interventions. More elaborate memory, interpretability, or curriculum machinery should answer a particular question that the running learner exposes.

## Depth, width, and Ripple — 5 November 2026

Ricursive Intelligence's Ripple-65B Preview uses 80 distinct layers and residual width 5,120. The latter is the dimension of each token's hidden state, not a parameter count. Its internal attention width is 10,240 (80 query heads of dimension 128), with 8 KV heads; its gated FFN intermediate width is 40,960. It is therefore narrow in its residual stream while doing substantially wider computation inside each block. The weights are not shared across decoder layers. [Official configuration](https://huggingface.co/ricursiveAI/Ripple-65B-Preview/resolve/main/config.json), [implementation](https://huggingface.co/ricursiveAI/Ripple-65B-Preview/resolve/main/modeling_ricursive.py).

The launch report combines architecture, normalization, information-flow, data, and post-training changes. Its headline efficiency comparison does not isolate depth as the explanation. Particularly useful is its willingness to discard mechanisms that failed to help at the intended scale: two-stream mHC helped smaller experiments but not its 4B test, while extra normalization became useful at that scale. [Ricursive's report](https://www.ricursive.com/ripple).

Controlled depth experiments at 41M, 134M, and 374M parameters found initial gains followed by diminishing returns or deterioration. Crucially, they held residual width and attention dimensions fixed and traded depth against FFN width. Their results do not prescribe our residual-width/depth ratio; their substantial pretraining also differs from our developmental setting. [Petty et al., NAACL 2024](https://arxiv.org/html/2310.19956v2).

One relevant lead from Ripple is Value Residual Learning: later attention layers receive a mixture of their own value vectors and first-layer values. Its separate paper includes 82M–468M experiments, making preservation of early information through depth a concrete research question. The tests concern language-model loss under extensive training, not cumulative tutored development or validation at our approximately 20B target. [Primary paper](https://arxiv.org/pdf/2410.17897).

**Implications:** distinguish residual width, attention width, FFN width, and computational depth. More layers supply successive transformations, not a guaranteed count of reasoning steps. Residual width is neither a concept count nor the total knowledge capacity of a model: weights and multiple token states also matter. There is no established numerical minimum for a general world representation without specifying tasks, precision, context, accuracy, and computational resources.

The former local probe's approximately 100M shape comparison was unimplemented and is retired. For the B200 target, exact depth, residual width, attention width, FFN dimensions, vocabulary, and context remain to be established together. More layers introduce sequential execution and can increase saved activation memory even at similar parameter counts. Compare proposed shapes with consistent educational conditions and suitable initialization, measuring progress per example and time, retention, transfer, and complete memory use. No narrow-width lower bound or layer count is selected by this research.

The broader lesson for RNN2 is to combine appropriate information flow and stable learning with experiences that respond to the student's development. Each mechanism should earn its place through a controlled comparison at the intended student scale.

## Delayed memory bus across depths — 5 November 2026

The proposed architecture samples internal hidden representations at several depths, perhaps five sections, and exchanges compressed information through a persistent bus. Each section reads an earlier snapshot and writes information for a later tick. The aim is continuity across an ongoing stream of experience, including access by early processing to prior deep processing. Improved stability and useful contextual organization are hypotheses to test.

Feedback Transformer is a close precedent: it combines representations across layers into memory for earlier time steps, accessible from every layer. Its memory retains a history of vectors rather than five compact slots. The authors explicitly discuss the resulting sequential computation and loss of training parallelism. [Primary paper, section 3.2](https://arxiv.org/pdf/2002.09402).

Recurrent Memory Transformer separates previous-segment read memory from current-segment writes and trains through multiple segments. Its experiments show that longer gradient windows can be useful but consume more GPU memory. It supplies a useful training/causality reference, although its memory tokens sit at segment boundaries rather than at several internal depths. [Primary paper, section 3](https://arxiv.org/pdf/2207.06881).

Transformer-XL carries prior hidden states while stopping their gradients. This distinguishes the lifetime of numerical context from the lifetime of the training graph. Its ordinary hidden-state producers also participate in current losses; that does not show that a dedicated bus writer can learn when every write is immediately detached. [Primary paper, section 3.2](https://arxiv.org/pdf/1901.02860).

**Possible minimal design:** give each tap its own small state slot, learned write projection, and access to learned reads from the previous bus. Freeze the read buffer for a tick, collect writes into a separate next-state buffer, and publish that state on the next tick. Gates and gradual updates make feedback strength and persistence controllable. Separate writer slots retain source identity and avoid blindly averaging representations from different depths. Five slots of 64 FP32 values would occupy 1.25 KiB per stream for one state buffer; projection parameters and training activations are additional costs. These numbers are illustrative, not selected dimensions.

**Timing:** a tick must be explicit. Per-token updates maximize responsiveness but introduce sequential dependencies across training tokens. Per-chunk updates preserve parallelism within a chunk and are a hypothesis to measure if this design is revisited. A whole-chunk summary may influence only later chunks, never earlier predictions in the same chunk. The delay removes a same-tick circular dependency; it does not parallelize the main stack or make recurrence free.

**Training:** if a write affects only a later loss, backpropagation must connect that later use to the write. An initial experiment should unroll at least two ticks and compare longer bounded windows as memory permits. Detach the graph at window boundaries while carrying the numerical bus onward. The bus can persist across teaching blocks and tutor reviews. Use coherent streams and separate state per stream; preserve state with checkpoints. Weight updates and direct edits can change how stored representations are interpreted, so continuity through those changes needs examination.

**Evaluation:** compare no bus, last-layer feedback, and distributed feedback under comparable memory and teaching budgets. Test use of history, resistance to distraction, correction when context changes, finite/recoverable dynamics, and durable learning after a diagnostic state reset. Shuffled or frozen bus controls can help establish whether useful communication is occurring. Keep contextual memory and persistent parameter learning distinct. This remains an architecture candidate, not a requirement or a demonstrated stability mechanism.

## Predictive coding as a learning-rule comparison — 12 November 2026

**Status:** parked research, not part of the B200 implementation sequence or an adopted learning mechanism. The local comparison was never implemented. The question is whether adjusting temporary hidden states before updating weights improves the randomly initialized student's learning enough to justify its measured cost. Predicting the next event is a task; predictive coding (PC) is a family of inference and learning procedures. Neither automatically supplies continual-learning retention, action selection, or the full uncertainty model discussed in the Friston/Goertzel podcast.

### Evidence and its limits

PC can alternate two operations: hold weights fixed while adjusting internal activities to reduce inconsistencies, then hold those activities fixed while changing the weights. Gradients still appear in both operations. [Learning on Arbitrary Graph Topologies via Predictive Coding](https://pmc.ncbi.nlm.nih.gov/articles/PMC7614467/).

There is direct, small-transformer evidence. *Predictive Coding beyond Gaussian Distributions* trains a causal one-block, width-128 transformer and finds that categorical/KL treatment at softmax boundaries substantially improves its PC result. This does not establish an advantage at our scale or on our GPU. [Primary paper](https://arxiv.org/html/2211.03481v1).

FabricPC supplies open-source transformer examples and a PC/backprop switch on the same graph. It is Python/JAX, with documented CUDA GPU setup, not a validated replacement for the planned C++ runtime. A contributor's matched five-seed experiment on a 413,249-parameter, two-block character model reports test perplexity 7.12 ± 0.05 for four PC settling steps versus 6.00 ± 0.05 for backprop, taking about 1.46 times as long. More settling improved early-layer gradient alignment but worsened test performance in those runs. This is an author's research report, not an independently reproduced benchmark or a timing forecast for our B200 student. [Repository](https://github.com/trueagi-io/FabricPC), [experiment report](https://bezawit.dev/projects/fabricpc/).

Solver choice matters. FabricPC distinguishes state-based PC, where feedback travels approximately one graph boundary per simultaneous settling step, from error-parameterized PC using a global reverse pass and backprop-scale memory. The PCX benchmark also finds depth and settling costs in actual GPU execution; theoretically parallel local operations do not automatically run in parallel. [FabricPC guide](https://github.com/trueagi-io/FabricPC/blob/main/docs/user_guides/03_how_predictive_coding_works.md), [PCX benchmark](https://arxiv.org/html/2407.01163).

### Concept retained for a possible future comparison

If revisited, keep forward modules identical across the comparison, including embeddings, causal attention, residual paths, normalization, and output head. One possible grouping places embeddings and the first transformer block in the first module, keeping embeddings trainable. Introduce a temporary state `z_l` at each block output. Input token IDs remain fixed; final normalization and projection form the output head `g`. This grouping is a research proposal, not the selected 20B architecture.

For observed targets `y`, use the proposed energy

```text
e_l = z_l - f_l(z_(l-1); theta_l)
E = sum_l 0.5 * ||e_l||^2 + beta * cross_entropy(g(z_L), y)
```

Cross entropy accepts logits; logits are not clamped to one-hot targets. Sum hidden squared errors over features, then average both consistency error and cross entropy over the same valid token positions. Record the relative weight `beta` explicitly, initially 1. State and weight learning rates are separate choices.

For each chunk:

1. Initialize states with an ordinary forward pass.
2. Freeze weights. Take a bounded number `K` of simultaneous state-gradient steps, recomputing predictions as states change. Every state update uses the same old snapshot.
3. Detach settled states. Calculate each module's parameter gradient from its local error, and the head's gradient from cross entropy; make one AdamW weight update.
4. Discard temporary states. Evaluate with a fresh ordinary causal forward pass, without targets being available to hidden-state inference.

LibTorch can compute the local derivatives through each module. Do not retain or differentiate through the whole settling trajectory. This is local communication between modules, with ordinary differentiation inside each module; it is not neuron-local learning or a demonstrated asynchronous GPU execution scheme. A whole transformer block remains the unit of local computation.

This variant recomputes predictions as states move. It is not the fixed-prediction construction that can reproduce backprop gradients. Check its state and parameter gradients against finite differences of its own energy, rather than requiring equality to backprop. [Backprop relationship and assumptions](https://arxiv.org/html/2006.04182v5).

### Conditions for reconsidering this mechanism

The old laptop-sized comparison is retired. A future proposal should use small configurations of the same B200 runtime for numerical checks, followed by a controlled learning comparison. Verify state/weight derivatives, finite values, causality, and recovery; confirm that supervisory influence reaches the earliest module. Match initialization, data, precision, optimizer family, evaluation, and tuning effort, and account for all relaxation work and additional state memory.

Useful evidence would concern independent prediction, retention after new learning, and transfer at acceptable measured cost. Synthetic repeated symbols can check machinery but do not establish those properties. No PC implementation or performance claim at 20B follows from the local experiment or these papers.

Temporary settling is not persistent contextual memory. Resetting these states isolates learning stored in weights. A delayed bus, persistent states, adaptive tutor choices, and action learning remain separate experimental changes. Reduced internal energy alone is not evidence of better independent prediction, retention, or transfer.

Finally, this fixed-precision, point-state candidate does not include a posterior covariance or entropy over states or weights. The fuller variational account requires explicit priors and uncertainty approximations; adding a scalar error weight does not implement it. [Variational free energy and the Laplace approximation](https://www.fil.ion.ucl.ac.uk/~karl/Variational%20free%20energy%20and%20the%20Laplace%20approximation.pdf). A PC feasibility result would therefore not confirm the podcast's broader efficiency or developmental claims.
