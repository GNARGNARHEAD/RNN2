# RNN2

Build a student without pretrained weights and explore how much capability we can develop on this laptop through tutorship, observation, and inspection of its internal computation. Begin with mimicry and prediction, and investigate how foundations can support increasingly demanding learning.

The ambition is a learner whose abilities accumulate: it retains useful experience, recognizes relationships in unfamiliar settings, combines skills, and becomes better able to learn. We also want to study what develops along the way. Unexpected behaviour or internal organization can change the questions we ask.

This is an open engineering exploration. Study broadly, distill the fundamentals, and add complexity when an observed need gives it a clear purpose.

## Current decisions

- **C++ is the project language.** This was settled on 4 November 2026.
- **The student starts without pretrained weights.** The first educational baseline will use ordinary random initialization. Developing its foundational abilities is part of the experiment. The external tutor can be a pretrained frontier model.
- **The laptop is the student's execution environment**, with its integrated GPU as the intended numerical accelerator. A more capable external model can act as tutor.
- **Aim for the strongest student we can train and develop locally from scratch.** Measure capability, execution cost, inspection cost, and update speed when choosing its size.
- **Keep the developmental method open.** Recurrence, a particular transformer architecture, and a fixed curriculum are exploratory options.
- **Human confirmation is required for design adoption.** The tutor must discuss new games and material design changes with GNARGNARHEAD and obtain explicit confirmation before using them in the student's education. Routine teaching can proceed within an already approved scope.

Native LibTorch with ROCm is the first backend validated on this laptop and is the current experimental baseline. Architecture, student size, precision, and update methods remain open. The initial probe uses FP32 and full-parameter AdamW; those are tested settings, not permanent constraints.

## The basic method

A student encounters something and responds. The tutor observes, forms a tentative explanation, and chooses an experience or intervention. The student can change. We examine what changed, whether it lasts, and what it enables.

Teaching, inspection, direct weight editing, and constructing a specified neural computation are possible interventions. The tutor can also pause intervention and observe. An unusual pattern deserves investigation before we decide whether it is useful or faulty.

Education and deliberate design should inform one another. The student's development informs the tutor's teaching choices and hypotheses about the learner's overall organization. Those hypotheses can motivate authored initializations or later structural interventions; their observed effects then revise the tutor's understanding. Evaluate them through subsequent learning, retention, and transfer, with preserved checkpoints and suitable comparisons. Keep broad organization open to investigation as well as individual computations.

Record any deliberately supplied computation or structure so we can distinguish it from abilities acquired through experience.

Keep observations, explanations, and demonstrated effects distinguishable. A proposed mechanism becomes more convincing when a controlled change produces the predicted effect on fresh cases. Save enough history to revisit surprises and revise an explanation.

Useful signs of development include retention after a fresh session, transfer to unfamiliar examples, composition of abilities, and easier acquisition of later skills. These guide inquiry; discoveries can also motivate new questions and measurements. Distinguish temporary context from persistent learning, and the student's independent performance from performance with tutor assistance.

## Human involvement in design

The tutor's design loop includes a conversation with GNARGNARHEAD. New games and material changes to mechanics, observations, actions, rewards, objectives, or termination conditions require explicit human confirmation before adoption for student education. Apply the same principle to substantial changes in the student's architecture, educational objectives, or learning/intervention mechanisms. The tutor may prepare and validate a candidate beforehand; provide a concrete proposal, its educational purpose, a visual demonstration where applicable, and the evidence and unresolved questions needed to review it.

For games, check coherent dynamics and observation/action interfaces, achievable goals, and reliable episode endings or explicitly bounded sessions for continuing tasks. Demonstrate success with a scripted or human controller using the student's allowed observation/action interface in representative cases; check for stuck states, impossible generated situations, and obvious ways to obtain reward without the intended achievement. Preserve replayable examples. These checks provide evidence, not a guarantee that every exploit has been excluded. Do not use the developing student's inability to succeed as the sole test of whether an environment works.

Record confirmation with the environment/design version and any approved range of variations. Ordinary examples, seeds, and difficulty adjustments within that agreed scope need not each prompt a new approval. Changes outside the scope must return to the conversation; silence is not confirmation. If a suspected exploit or impossible situation is found later, suspend teaching with the affected version, preserve the evidence, and bring the issue to GNARGNARHEAD before adopting a revised design. The working document records this requirement; runtime enforcement remains to be implemented.

## Initial cadence (provisional)

Use different rates for learning, observation, and intervention. The first session should help establish the student's pace of change and the cost of observing it.

- During ordinary learning, collect inexpensive summaries of loss, numerical health, timing, and resources.
- Start with a small behavioural check after roughly 100 training steps: current learning, earlier learning, and fresh variations or combinations.
- Begin tutor reviews every 3–5 such blocks, with earlier reviews when meaningful changes warrant them. Include concrete responses, trends, and modest, consistent internal summaries across the network.
- Undertake deeper inspection or structural intervention when there is a specific question to investigate. Preserve the checkpoint and compare the candidate with continued education from the unedited state, including subsequent learning and retention.

These numbers are an engineering starting point, not a validated optimum or permanent requirement. Record training exposure and elapsed time; shorten or lengthen observation intervals as evidence warrants. Keep the questions and possible responses open while making individual experiments understandable. Continued observation is a legitimate choice, and unexpected successes deserve investigation as well as failures.

## Education from foundations

Mimicry and prediction provide the initial direction, extending the questions from the earlier RNN work. The analogy to a K through 12 education expresses the ambition for cumulative foundations: basic relationships, language, quantities, and operations supporting more demanding reasoning and practical tasks. The sequence, pace, and eventual reach should follow what the student demonstrates.

Initially, the tutor can teach through the examples, targets, and interactions it supplies. Natural-language comprehension is itself something the student must acquire; a written critique cannot be assumed to be an understood lesson. More conversational instruction can develop as the student's capabilities support it.

Early examples can be simple while the model is sized for sustained development on the laptop. Revisit earlier competencies, vary familiar relationships, and investigate whether they become useful in new combinations. Preserve opportunities to observe unexpected organization as education proceeds.

Use simple, locally authored games as another educational mode for the same student, beginning with a small C++ paddle-and-ball world. The tutor can design and revise environments in response to observed learning, supply demonstrations, and select situations for prediction and action. Mechanics can serve the student's education without reproducing human-oriented arcade conventions; paddle tilt is one possible extension, not a required initial feature. Games complement the broader educational direction.

Support the required human design review with a visual view and replay derived from the actual simulation state. Make the student's observations, actions, and predictions inspectable, distinguishing them from information available only to the simulator or evaluator. Tutor-authored changes should be recorded as environment versions with reproducible episodes and checked for coherent rules, observations, and outcomes. Prefer parameter changes within an approved range when they answer the question; new mechanics remain possible after review and confirmation. Preserve some unchanged evaluation situations so improving scores can be distinguished from easier lessons or changed rules. These are design intentions; the game and tutor integration are not implemented yet.

## GPU probe and next experiment

The [C++ GPU probe](experiments/2026-11-04-gpu-probe.md) now runs on the integrated GPU. It verifies intermediate values, gradients, AdamW updates, a controlled parameter edit, and model/optimizer checkpoint recovery against the same implementation on the CPU. Separate runs measure training speed and memory at increasing sizes. All probe models start from random weights.

Build and configure an already installed backend with:

```bash
bash scripts/build_probe.sh
source scripts/backend_env.sh
build/native/rnn2_probe --help
```

The experiment record contains package installation, complete verification/benchmark commands, results, and limits. GPU execution requires access to the host's GPU devices; the command sandbox does not expose them. Python is used for installation and optional process supervision. The executable performs inference, inspection, differentiation, optimization, direct editing, and checkpoint operations in C++.

The next educational experiment should use this working backend with a small paddle-and-ball environment to investigate cumulative mimicry, prediction, and action: practice a relationship, vary its presentation, introduce a related demand, then revisit the earlier ability and probe unfamiliar combinations. Establish coherent simulation, a visual inspection/replay view, and a clear student observation/action interface. Validate the candidate game and obtain GNARGNARHEAD's confirmation of its concrete design and allowed variations before student training uses it. Choose the initial student's size from measured memory and lesson-turnaround costs. The probe supplies engineering evidence; it does not establish which size or architecture learns best.

## Current evidence

The host reports an AMD Ryzen 5 6600H and integrated GPU target `gfx1035`, with about 13.87 GiB usable system RAM. HIP exposes 6.93 GiB of shared device memory, which draws on the same physical RAM. Available memory varies with other applications.

On 4 November 2026, native C++ training passed correctness checks at 27,872 and 3,320,576 parameters using AMD's stable ROCm 10.1.0 / PyTorch 2.14.0 packages. A 151,775,488-parameter model completed three measured minutes of FP32 training with full AdamW updates at about 719 synthetic tokens/s, retaining about 2 GiB available host memory. A shorter 12.9M run measured 6,734 tokens/s. These figures use batch 4, sequence 128, and vocabulary 256; they describe the disposable workload, not language ability or a universal size ceiling. The sustained run showed modest timing drift and high temperatures; see the experiment record for measurements and limits.

No educational student has been selected. Longer context, vocabulary, precision, memory overhead, and acceptable update time all affect the useful size. Foundational learning and cumulative development remain untested.

Weight-space programming is a reason to investigate direct edits. The reported examples support particular repairs and bounded constructions; their value for cumulative development on this laptop is an experimental question.

## Project structure

This README is the working reference for purpose, current decisions, open questions, and the next experiment. [AGENTS.md](AGENTS.md) gives short instructions for coding sessions.

[Research findings](research/FOUNDATIONS.md) distill the initial reading on liquid networks, continual learning, and adaptive teaching into practical implications and evidence limits.

[Exploratory connections](research/EXPLORATION.md) follow developmental interpretability, curiosity, reusable abstractions, and memory across timescales. They record questions and possibilities without selecting an architecture.

`CMakeLists.txt` and `src/probe.cpp` provide the disposable native probe; its `verify` mode contains the numerical checks. `scripts/` contains backend setup and bounded process supervision. Keep inference, inspection, learning, and editing connected to the same implementation as the student develops. Add separate tests when they answer a new correctness question.

`experiments/` records the question, relevant software and model versions, starting checkpoint, intervention, observations, interpretation, and unresolved questions. Include commands and seeds when needed to reproduce a finding. Records can be brief.

Use `models/` for local model files and checkpoints, and `runs/` for generated outputs and detailed traces. These directories and `build/` are excluded from Git by `.gitignore`. Preserve small experiment records and references to the relevant artifacts with the source.

## Keeping the project coherent

Update the current decisions and next experiment here as they change. Keep proposals labelled until adopted. Put detailed evidence in experiment records and link the findings that change our direction. Introduce new modules and documents when they serve a concrete need.

**4 November 2026:** Established this working reference and selected C++. Set the initial next step to the C++ GPU feasibility probe, leaving model choice and architecture open.

**4 November 2026, probe implementation:** Native LibTorch/ROCm passed actual GPU training checks, controlled editing, and checkpoint continuation. The runnable probe and measurements are recorded under `experiments/`. This completes the initial backend feasibility step; student selection remains experimental.

**5 November 2026, education and design:** Begin the educational baseline with random weights. Use dynamic feedback between the student's development and the tutor's teaching, inspection, and design interventions. Intentionally authored organization remains an experimental option informed by this loop; a complete initial design is not a prerequisite for starting education.

**5 November 2026, initial cadence:** Agreed to a provisional separation between continuous learning/monitoring, small checks around 100 training steps, tutor reviews around every 3–5 blocks, and question-driven deeper investigations. Adapt the cadence from observed development and inspection cost.

**8 November 2026, educational games:** Begin with a simple paddle-and-ball world. Extend the tutor's possible interventions to authoring and adapting educational environments, including mechanics designed for the student. Provide visual feedback for human oversight, preserve environment versions and comparison cases, and keep game learning connected to the same student's broader development. Detailed mechanics and the game-learning method remain open.

**8 November 2026, human design confirmation:** Explicitly require conversation and GNARGNARHEAD's confirmation before adopting new games or material design changes. Prepare a reviewable candidate with visual evidence and checks for achievable goals, termination, and obvious exploits. Record the approved scope so routine teaching can continue within it; return for review when that scope changes or an environment defect is suspected.

The student will begin without pretrained weights, starting with mimicry and prediction and exploring cumulative education from those foundations. This replaces the earlier proposal to begin with a pretrained student.

The first bounded research pass reinforces tracking retention and the ability to learn separately, distinguishing hidden-state adaptation from parameter learning, and choosing practice from the student's observed behaviour.

## References

- [Earlier RNN tutor project](https://github.com/GNARGNARHEAD/RNN) — background on tutoring, rehearsal, and interference in a deliberately tiny learner.
- [PyTorch C++ frontend](https://docs.pytorch.org/tutorials/advanced/cpp_frontend.html) and [native installation](https://docs.pytorch.org/cppdocs/installing.html) — candidate numerical and training backend.
- [AMD TheRock release documentation](https://github.com/ROCm/TheRock/blob/21593717e7ddb57acf26a8a336fdf2e7187534e9/RELEASES.md) — the GPU package route identified during feasibility research.
- [Weight-space programming report](https://rikiyat.github.io/blog/weight-space-programming/) and [experimental records](https://github.com/RikiyaT/lm_understand-logs) — evidence to examine when considering direct parameter interventions.
