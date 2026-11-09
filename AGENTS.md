# Working on RNN2

Read `README.md` before making project changes. It is the working reference for direction, decisions, open questions, and the next experiment. The user's current instructions take precedence.

- Use `GNARGNARHEAD` as the project identity and as both Git commit author and committer, with the account's GitHub noreply email address. Do not publish personal names, personal email addresses, or personal home-directory paths. Verify the identities and tracked contents of every commit being pushed.
- Implement the student runtime in C++. The numerical backend and initial model remain open until supported by the feasibility work.
- Start the student without pretrained weights. Begin its education with mimicry and prediction; treat the K through 12 analogy as cumulative development with an exploratory sequence. A pretrained external tutor remains appropriate.
- Keep one student implementation serving inference, inspection, learning, and direct editing.
- Preserve the exploratory scope. Recurrence, a fixed model size, and a predetermined developmental sequence are not requirements.
- Seek GNARGNARHEAD's explicit confirmation before adopting new games or material design changes for student education, including rules, observations/actions, rewards, termination, architecture, educational objectives, or learning/intervention mechanisms. Prepare a concrete, validated candidate and visual demonstration where applicable before requesting confirmation. Record the approved version and scope; routine teaching within that scope does not require repeated approval.
- Validate game coherence, achievable goals, episode termination or bounded continuing sessions, and obvious reward exploits independently of the student's performance. Preserve replayable evidence. Suspend teaching with a version suspected of being invalid or exploitable and discuss it with GNARGNARHEAD before adopting a revised design.
- Prefer the smallest understandable change that answers the current question. Reuse established numerical machinery where appropriate.
- Treat observations, hypotheses, and validated effects distinctly. Preserve checkpoints and enough provenance to revisit an interesting result.
- Test actual GPU operations and numerical results. Package availability, device detection, and matrix multiplication alone do not establish working model training.
- Use supervised subagents for external research and review their evidence before adopting a claim or implementation choice.
- Keep `README.md` current when decisions or the next experiment change. Add detailed findings under `experiments/` as work produces them. Keep large weights and raw traces in the ignored artifact directories.
