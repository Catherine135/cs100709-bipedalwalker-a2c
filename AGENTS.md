# Agent handoff

Read README.md first. The current user request is to improve the uploaded baseline
as much as possible toward an original-reward score of 300. The main Windows path
is now PPO; A2C remains a reproducible baseline. The README supersedes the earlier
one-million-step A2C-only handoff and explains the bounded 20-million-step PPO run.

Reuse GWTower's validated Python environment. Add only missing PPO dependencies;
do not reinstall PyTorch. Preserve the exact 1600-step Hardcore environment and
raw rewards. Observation normalization is internal and exported with the Agent.
Never train on the user's Mac; read-only model evaluation and zero-update checks
are allowed. Reading or editing this repository alone is not a request to train.

Follow the README through evaluation and delivery when asked to train. Verify
process identity before resuming; never load an A2C checkpoint as PPO. Preserve
old runs, and do not publish new models or logs unless asked. Report measured
scores honestly; 300 is a target, not a guarantee or a quoted course passing rule.

Keep the single train.py deliverable and teacher Agent interface. Comments are
English, complete modules have two blank lines between them, and standalone
aligned comments introduce each module. Cite third-party algorithms. Run relevant
non-training checks after changes; do not use model fitting as a local test.
