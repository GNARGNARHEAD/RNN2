# Interpretability for a Developing Student

Reviewed 4 November 2026. The useful question is how to connect changes in the student's behaviour with changes in its computation. These studies provide instruments for that investigation; they do not supply a complete interpretation of an arbitrary model.

## Watching mechanisms develop

*Progress measures for grokking via mechanistic interpretability* follows small transformers learning modular addition. The authors reconstructed a Fourier-based computation and tested it through interventions. Measures derived from that mechanism showed useful structure forming before the conspicuous rise in test accuracy. The work motivates preserving checkpoints during apparently flat performance, while its hand-derived measures remain specific to the task. [Nanda et al., ICLR 2023](https://arxiv.org/html/2301.05217v3).

*In-context Learning and Induction Heads* connects the formation of particular attention circuits with improved use of preceding context. Its strongest causal evidence concerns small attention-only models. [Olsson et al., 2022](https://transformer-circuits.pub/2022/in-context-learning-and-induction-heads/).

*What needs to go right for an induction head?* investigates interacting subcircuits during training. Redundant heads can hide a component's contribution when it is removed individually. Equivalent computations can also occupy different internal coordinates, making direct transplantation between models misleading. [Singh et al., ICML 2024](https://proceedings.mlr.press/v235/singh24c.html).

**Implication:** compare repeated behavioural and internal observations across checkpoints, including probabilities or losses where a binary score hides changes. An apparent plateau may warrant investigation; it is not itself evidence of hidden progress. For our student, preserve the initial state and early checkpoints so the formation of foundational mechanisms can be investigated.

## Testing an internal role

Activation patching replaces a selected intermediate activation in one run with a value from another run and measures the effect. Causal tracing uses this idea to investigate recovery of information after a controlled disruption. [Meng et al., NeurIPS 2022](https://arxiv.org/html/2202.05262v5#S2.SS1).

*Towards Best Practices of Activation Patching* shows that the input alteration, output metric, and collection of patched components can substantially change the apparent localization. Plausible contrasting inputs and a clearly defined output distinction make the intervention easier to interpret. A joint intervention does not establish each component's individual importance. [Zhang and Nanda, ICLR 2024](https://arxiv.org/html/2309.16042v2#S6).

Gradient-based approximations can rank many candidate sites, but can also miss effects. *AtP\** uses approximate ranking with exact patching verification; a small layer-level investigation may not need that acceleration. [Kramár et al., 2024](https://arxiv.org/html/2403.00745v1).

**Implication:** begin with a small, explicit intervention. Information being readable from an activation does not establish that the model uses it. A successful patch demonstrates an effect under the tested conditions, rather than a unique location for a concept or an automatically appropriate weight edit.

## Finding distributed features

*Towards Monosemanticity* learns sparse feature representations from a small transformer's activations. It demonstrates why individual neurons can be poor units of explanation: meaningful patterns may span several neurons, with individual neurons participating in multiple patterns. The learned dictionary is an interpretation tool rather than a unique inventory of concepts. [Anthropic, October 2023](https://www.anthropic.com/research/towards-monosemanticity-decomposing-language-models-with-dictionary-learning).

*Circuit Tracing* constructs attribution graphs using a learned replacement for parts of the network and tests hypotheses with interventions. The replacement is approximate, omits some mechanisms, and needs reconstruction-error terms. Matching the original output on one prompt does not establish that the same mechanism produced it. The published large-scale pipeline requires substantial additional training. [Ameisen et al., March 2025](https://transformer-circuits.pub/2025/attribution-graphs/methods.html).

**Implication:** consider a sparse autoencoder or related tool when mixed activations obstruct a specific investigation. Compare intervention effects and behaviour across checkpoints; neither a neuron number nor a learned feature ID necessarily keeps the same meaning as the student or the measuring model changes.

## A possible first instrument

The C++ student could expose selected named intermediate tensors for recording and temporary replacement during a forward run. A small diagnostic would compare two aligned inputs that differ in one relevant fact, establish their baseline predictions, and replace a selected activation in one with the corresponding activation from the other. Repeat in the reverse direction and on fresh variations. A same-run replacement provides a simple check that the instrument itself leaves the result intact.

This is an engineering proposal, not a measured result. It needs forward execution and a few saved tensors rather than a second large model. Preserve the original checkpoint and compare intervention effects within each checkpoint before interpreting changes across training. More detailed instruments can follow a concrete discovery or question.
