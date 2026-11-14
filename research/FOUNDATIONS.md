# Research Findings for RNN2

Reviewed 4 November 2026. This first reading pass asks which findings improve our observations or decisions while leaving the implementation small. The sources describe results in different settings; applying their lessons to this student remains an experiment.

Applicability updated 14 November 2026: the project targets approximately 20B parameters on one B200 with persistent cloud checkpoints. These research findings inform that work; they do not prescribe architecture or replace the [active implementation plan](../experiments/2026-11-14-b200-plan.md). The local GPU experiment is retired.

Our student will begin without pretrained weights. Results obtained with already capable students can suggest methods to investigate, but do not establish how foundational language and learning abilities will arise in this project.

## Liquid networks and efficient language models

The original MIT liquid-network work concerns temporal dynamics. Liquid Time-Constant networks vary their effective time constants with input and state. Closed-form Continuous-time networks approximate those dynamics without the numerical differential-equation solver. Their reported speed advantages concern comparisons with differential-equation models, not modern language transformers. In the authors' implementation, a forward call updates hidden state; it does not update learned weights. Neural Circuit Policies provide structured sparse wiring for such cells. [CfC paper, 2022](https://arxiv.org/abs/2106.13898), [author implementation](https://github.com/mlech26l/ncps).

Liquid AI's LFM3 release of 27 October 2026 is a different design: gated short convolutions combined with sliding-window and global attention. It is an example of allocating attention selectively in an efficient language model. The documentation reviewed covers native llama.cpp inference and Transformers/TRL fine-tuning. Its AMD edge measurements concern a different workload and hardware; neither those inference results nor the control-task results above establish from-scratch C++ training of our proposed B200 student. [LFM3 announcement](https://www.liquid.ai/blog/lfm3), [model card](https://huggingface.co/LiquidAI/LFM3-1.7B).

**Implication:** keep hidden-state dynamics, persistent parameter learning, and execution efficiency distinguishable. Liquid mechanisms and hybrid language models are candidates to investigate if measurements justify them. The name alone does not imply continual learning.

## Retention and the ability to keep learning

*Loss of plasticity in deep continual learning* shows that learning ability can deteriorate across repeated tasks. Its vision and control experiments investigate this separately from forgetting. Selectively replacing low-utility units helped in the tested settings; ordinary regularization also helped in some cases. These results motivate monitoring new learning, but do not establish neuron replacement as a remedy for a language learner. [Dohare et al., Nature, August 2024](https://www.nature.com/articles/s41586-024-07711-7).

*On Tiny Episodic Memories in Continual Learning* provides a useful simple baseline: combine current examples with a small sample of previous examples in ordinary training updates. It reduced forgetting in supervised image benchmarks, including evaluation on held-out examples. Those experiments used task-specific heads and known task identity; their results do not demonstrate open-ended language retention. [Chaudhry et al., 2019](https://arxiv.org/html/1902.10486).

A later vision-transformer preprint found only marginal improvements from its adaptations of continual backpropagation. This reinforces the need to test whether a remedy transfers to our architecture and setting. [Sun et al., March 2026 preprint](https://arxiv.org/html/2603.07787v1).

**Implication:** observe both whether old skills remain usable and whether new skills are becoming harder to acquire. A small, varied rehearsal set is a useful comparison before adding machinery. Keep its evaluation examples separate. Changing task difficulty can itself change learning speed, so comparisons need an appropriate reference.

## Teaching from the student's behaviour

*Teacher–Student Curriculum Learning* selects practice using evidence of learning progress and deterioration. Its experiments use defined task families; adaptive selection helped, but uniform sampling was competitive in one setting. This supports responding to observed change without assuming that a complex curriculum controller is necessary. [Matiisen et al., 2017 preprint](https://arxiv.org/html/1707.00183v2).

*On-policy Distillation of Language Models* trains already capable students using teacher feedback on sequences the students generate themselves. This addresses the mismatch between polished teacher demonstrations and the situations the student actually reaches. [Generalized Knowledge Distillation, ICLR 2024](https://arxiv.org/html/2306.13649v3).

*Speculative Knowledge Distillation* finds that poor student-generated contexts can also make teacher guidance unhelpful, and studies inserting teacher corrections into those sequences. Both distillation methods rely on teacher token probabilities and compatible vocabularies. Prose feedback from an external tutor would be an adaptation of the principle, not an implementation of those algorithms. [Speculative Knowledge Distillation, ICLR 2025](https://arxiv.org/html/2410.11325v3).

**Implication:** observe actual attempts, vary assistance, and let evidence of improvement or regression influence subsequent practice. These papers do not establish development from random initialization or lifelong retention.

## Interpretability

[Interpretability for a developing student](INTERPRETABILITY.md) covers mechanisms forming across training checkpoints, activation patching, and sparse feature analysis. The immediate lesson is to preserve observations across learning and support small causal interventions. A readable feature, a behavioural effect, and a complete explanation are different levels of evidence.

## What this changes for the project

Keep a few distinctions visible in our records:

- State changes during an interaction and persistent changes to parameters.
- Retaining earlier abilities and acquiring new ones.
- Independent performance and performance with assistance.
- A tutor's interpretation and the evidence that supports it.

The first system can observe these through interactions, saved checkpoints, small probes, and ordinary updates. We can study unexpected internal behaviour more closely when it appears. The earlier weight-space programming review adds direct intervention as another experimental tool; these findings do not establish when it should replace teaching.

The next infrastructure work is the CUDA runtime and recovery path in the [B200 plan](../experiments/2026-11-14-b200-plan.md). Consider execution, inspection, and updates together at the approximately 20B target. This reading pass adds questions to observe, without selecting a liquid architecture, a new optimizer, or a curriculum algorithm. Return to the literature when a result gives us a more specific question.
