# RNN2

Develop a continuously learning student from random weights through tutorship, observation, and inspection of its internal computation. The engineering target is **approximately 20 billion parameters on one rented NVIDIA B200**, implemented in **C++**. Begin with simple text/symbol mimicry and prediction, then investigate how acquired foundations support further learning.

The ambition is a learner whose abilities accumulate: it retains useful experience, recognizes relationships in unfamiliar settings, combines skills, and becomes better able to learn. Study what develops along the way. Unexpected behaviour or internal organization can change the questions we ask.

**Current status — 14 November 2026:** the B200, approximately 20B target, and persistent cloud storage with separate downloads are approved project direction. The repository contains the implementation plan and research; the retired laptop probe and its setup have been removed. The B200 runtime has not been implemented or validated, no cloud resources have been provisioned, and no educational student has been trained. Twenty billion is a target to establish experimentally, not a demonstrated capacity result.

## Settled direction

- **One B200, approximately 20B parameters.** Use low-precision computation and compressed training state where they preserve learning. FP4 is the preferred compute direction to investigate, with other precisions retained where needed. Exact dimensions, precision assignments, and optimizer implementation depend on validation.
- **One C++ student runtime** serves inference, inspection, full-parameter learning, and direct editing. The planned numerical foundation is CUDA LibTorch with validated specialized kernels where useful. Python may support packaging or orchestration; student computation belongs in C++.
- **Start without pretrained student weights.** Ordinary random initialization is the baseline. A pretrained external GPT tutor is appropriate.
- **Begin with RNN1-style mimicry and prediction over simple text or symbols.** The K through 12 analogy expresses cumulative development, not a fixed syllabus. The student has no image capability: no image inputs, vision encoder, or visual learning objective.
- **Learning continues across sessions.** There is no planned terminal trained state followed by permanently frozen deployment. Checkpoints record stages of the same developing learner. Pauses, bounded update blocks, and evaluation with learning disabled are compatible with this lifecycle.
- **Persistent cloud storage holds the full learning state.** Save and verify it before releasing compute; resume from that storage next session. Download full backups separately while the GPU is off. The laptop is for development, control, and review, rather than large-model training.
- **Keep the investigation open.** Recurrence, a delayed memory bus, authored weight edits, particular depth/width choices, and a predetermined developmental sequence are not requirements. A future 128 GB machine is not a current compatibility constraint.
- **Human involvement grounds material design choices.** Discuss substantial changes to architecture, educational objectives, rewards, or intervention mechanisms with GNARGNARHEAD before adoption. Routine work within an approved scope does not require repeated approval.

The [B200 implementation and checkpoint plan](experiments/2026-11-14-b200-plan.md) is the detailed operational reference. It separates approved direction, proposed numerical mechanisms, supporting evidence, and work still to validate.

## How the student develops

A student encounters something and responds. The tutor observes, forms a tentative explanation, and chooses a teaching experience or intervention. The student changes. We examine what changed, whether it lasts, and what it enables.

The tutor's assessment concerns the student's grasp of the situation and appropriate direction. GNARGNARHEAD's use of “sentiment” refers to this contextual understanding, rather than emotional tone or positive/negative sentiment classification.

- **Equivalent correct answers receive full credit.** For two plus two, `4`, `four`, and four text tally marks can all express the answer. A specific representation matters only when the task requires it.
- **A useful approach with an execution error can receive partial credit.** Three tally marks might indicate recognition that a quantity is requested, while leaving uncertainty about counting, combining quantities, or expressing the result. Nearby examples should distinguish these hypotheses before the tutor assumes an arithmetic deficit.

Appraisal should both recognize useful understanding and reveal what needs development. Preserve useful patterns while correcting errors, then check fresh cases and earlier learning. A score alone does not capture the tutor's diagnostic role, and a plausible-looking response does not establish understanding.

Token generation and meaning-based reward are compatible. The concrete rule that turns the tutor's judgment into a learning update remains to be agreed. In particular, it must explain how to strengthen a useful aspect without indiscriminately reinforcing an accompanying mistake. RNN1 already grades contextual partial credit, but uses that score to guide supervised correction whose updates use answer-token cross-entropy. Carry forward the grading principle without confusing it with direct reward optimization. [RNN1 grading protocol](https://github.com/GNARGNARHEAD/RNN/blob/3b240e1230b64343ab6cc4f23a8cce9a443f8d22/TUTOR.md#L131-L157), [score-guided correction](https://github.com/GNARGNARHEAD/RNN/blob/3b240e1230b64343ab6cc4f23a8cce9a443f8d22/scripts/tutor_stepper.py#L708-L735), [weight update](https://github.com/GNARGNARHEAD/RNN/blob/3b240e1230b64343ab6cc4f23a8cce9a443f8d22/rnn/session.py#L387-L412).

Initially, the tutor teaches through examples, targets, and interactions. Natural-language comprehension must itself develop; written criticism cannot be assumed to be an understood lesson. Keep the first educational implementation paused until the smallest foundational loop is agreed: what the student receives, what it reproduces or predicts, and how the tutor's response changes learning. Infrastructure preparation can proceed within the approved B200 direction.

Games and advanced tasks are not the starting curriculum. The earlier game-first preview was withdrawn. Any later educational game needs a concrete human-reviewed design and independent checks for coherence, achievable goals, bounded sessions or termination, and obvious reward exploits. Suspend teaching with a suspected invalid version and preserve replayable evidence. This does not authorize image capability.

## Observation and inspection

Track acquisition, retention, transfer to unfamiliar examples, composition of abilities, and subsequent ease of learning. Distinguish performance with tutor assistance from independent performance, temporary context from durable parameter changes, and an interpretation from a validated effect.

Collect inexpensive numerical and behavioural summaries during learning. Run deeper inspections when a question warrants them. Perform tensor inspection beside the student on the B200 and return selected results to the laptop. Full weight downloads are unnecessary for ordinary tutor reviews.

Teach, observe, and inspect at rates supported by measured learning and tutor latency. Earlier laptop-era step-count suggestions are historical starting ideas; the B200 cadence will be measured using generation, API turnaround, updates, small evaluations, and checkpoint cost. Prepare examples ahead only within the approved scope, and identify the student and lesson versions associated with delayed feedback.

Preserve checkpoints and comparable observations around interesting changes. Temporary activation replacement and direct parameter editing are possible instruments, not assumed improvements. Record deliberately supplied structure so it can be distinguished from abilities acquired through experience. Compare interventions against continued development from the same starting checkpoint, including retention and subsequent learning.

## Compute, storage, and recovery

The planned arrangement has three parts:

| Component | Responsibility |
|---|---|
| Single B200 instance | One C++ student, training, generation, and bounded inspection |
| Persistent cloud volume | Complete learning checkpoints, observations, and replayable history across rental sessions |
| Laptop and independent backup destination | Development, control, small reports, and occasional full backups |

A checkpoint must preserve the authoritative weights, optimizer state, numerical/random state, and learning position. An inference-only FP4 export is not a substitute for resuming the learner. Save at a consistent update boundary, write immutable chunks with hashes, and publish the completion manifest last. Keep a previous verified generation and test recovery on a fresh compute instance.

The provider candidate is Runpod Secure Cloud with a standard network volume whose S3 interface supports downloads without a running Pod. Provider/datacenter selection remains to be made. The planning allocation is 500 GB persistent storage, expandable as measured usage grows. Copies on that volume are not independent backups; retain occasional complete copies elsewhere. See the [operational plan](experiments/2026-11-14-b200-plan.md) for source-backed constraints, size estimates, transfer times, and dated cost references.

## Next implementation work

1. Prepare the CUDA C++ build and bounded checkpoint reader/writer for the B200 runtime. Carry forward the numerical checks from the retired experiment, without restoring a local training setup.
2. Validate actual forward/backward/update operations on a small disposable configuration, then prove save, compute shutdown, fresh-instance recovery, and interrupted-download recovery.
3. Validate compressed optimizer storage and FP8/NVFP4 operations separately and together. Scale toward the approved approximately 20B target, measuring full training, inspection, and checkpoint peaks. Smaller configurations are validation cases; any proposed change to the target should follow the evidence and be discussed.
4. Agree the smallest mimicry/prediction loop and its feedback-to-update mechanism before student education begins. Evaluate learning, retention, and transfer separately from hardware feasibility.

The approximately 20B training system is not yet demonstrated. Numerical stability, actual peak memory, useful tutor-cycle latency, and recovery overhead are the immediate engineering questions. Paid provisioning, exact model dimensions, and adoption of particular learning mechanisms are separate from this repository realignment.

## Project reference

- [AGENTS.md](AGENTS.md): instructions for contributing and preserving the approved scope.
- [B200 implementation and checkpoint plan](experiments/2026-11-14-b200-plan.md): next work, capacity evidence, persistence, and validation gates.
- [Research foundations](research/FOUNDATIONS.md): continual learning, teaching, and efficient models.
- [Interpretability](research/INTERPRETABILITY.md): observing development and testing internal roles.
- [Exploratory research](research/EXPLORATION.md): optional mechanisms and their evidence limits.
- [Retired local feasibility result](experiments/archive/2026-11-04-local-gpu-probe.md): a short historical record; it is not a current setup guide or B200 benchmark.

There is no current runnable training target. Add implementation files as the B200 work is built and validated. Keep one student runtime across inference, inspection, learning, and edits. Large weights, raw traces, and generated build artifacts belong in the ignored `models/`, `runs/`, and `build/` directories; their existence does not make them part of the active implementation.

## Decision record

- **4–5 November 2026:** selected C++, random student initialization, and ongoing tutorship with observation and inspection. The local numerical experiment established limited training feasibility.
- **8 November 2026:** required human review of material educational designs, including independently validated games if introduced later.
- **13 November 2026:** clarified meaning-based tutor appraisal and continuing learning; restored text/symbol mimicry and prediction as the starting scope, excluded image capability, and withdrew the game-first preview.
- **14 November 2026:** selected one B200, then approved approximately 20B as the engineering target and persistent cloud storage with separate downloads. Realigned the repository, retired the local probe and setup, and retained concise historical evidence. The exact architecture, numerical recipe, and foundational educational loop remain to be validated and agreed.
