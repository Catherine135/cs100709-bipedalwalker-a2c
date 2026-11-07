# Agent handoff

Read README.md first. The current authorized implementation is the Issue #3
campaign (protocol version 2, directory runs/campaign-issue3-v4): Gaussian PPO
and gSDE PPO from the pinned Issue #3 release, two independent training seeds per
profile, bounded failed-terrain replay, separate complete reward/coverage
checkpoints, then alternating continuation routes within 100M total new steps.
Use --mode campaign on the target Windows computer when instructed to train.
Do not start the old Issue #2 protocol, reuse its directory, or silently increase
the budget. The source hash and protocol must remain unchanged during a run.

Reuse GWTower's validated Python environment. No new dependency is introduced;
install only missing packages, never routinely reinstall PyTorch. Preserve the
original rewards, observations and exact 1600-step Hardcore environment. Training
terrain seeds must not overlap development or final seeds. Keep at least 75%
fresh terrain resets and the bounded retry queue in the default protocol.

Never train on the user's Mac. Only read-only evaluation, --dry-run and zero-update
checks are allowed there. Reading/editing the repository alone is not a request
to train. Do not call a real learner or optimizer.step during local validation;
tests use synthetic outcomes and mocked learners with Adam.step blocked.

On Windows, follow the campaign through evaluation and packaging. Check live
processes before updating/resuming, preserve prior runs, and follow the README
for safe interrupts. Report completed-update steps separately from discarded-step
upper bounds. The final 100 seeds stay unread until final-choice.json is frozen;
a disappointing holdout must not trigger reselection or further training.

Report coverage_goal_reached (>=80% finish), reward_goal_reached (raw mean >=300),
and their conjunction goal_reached separately. These apply independently to both
development groups and final confirmation. Report strict_300_all_finished too.
The five historical example seeds are demonstrations, not a fresh test set.
A completed campaign with unmet goals still exits 0; an interruption exits 130.
Do not publish new trained models or logs without the user's explicit request.

Keep the single train.py deliverable, portable CPU Agent object and stateless
teacher Agent.select_action(state, eval=True) API. Preserve the A2C baseline.
Use English comments, standalone module comments and two blank lines between
complete modules. Cite external algorithms and distinguish literature results
from this project's measurements. Run the non-training checks after source edits.
