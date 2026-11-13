# Agent handoff

Read README.md first. The current implementation is the Issue #4 campaign
(protocol version 3, directory runs/campaign-issue4-v5). It imports the pinned
interrupted-run best model and full optimizer checkpoint. Compare ordinary PPO
and CAPG PPO on two independent training seeds, then continue the better method's
reward/coverage leaders within 60M new interactions. Stop after three unproductive
continuation stages. The program manages the allocation; do not extend it silently.

Keep the interrupted v2 directory and source commit 13fb52b intact. That run did
not complete its final holdout. Never resume it using the new source. The new
protocol requires a new directory; reuse the existing validated Python environment
and install only missing packages. CAPG adds no dependencies. Do not reinstall
PyTorch, add TensorFlow, or repeat setup on GWTower's already configured machine.

Preserve original rewards, observations, Hardcore physics, the 1600-step limit,
and stateless Agent.select_action(state, eval=True). CAPG changes the training
log-probability at clipped action bounds; Gaussian raw-action gradients remain a
valid comparator. Do not change only one side of the rollout/update likelihoods.
Keep CAPG configuration and GAE lambda consistent after checkpoint restoration.

Never train on the user's Mac. Only zero-optimizer-update checks, dry-run and
read-only evaluation are allowed. Use mocked learners for lifecycle checks and
block Adam.step. Formal training occurs only on the target Windows computer.
Do not claim a score improvement from software checks or from incomparable seed
sets. The latest release's 203.47 is a validation mean, not a final-test result.

On Windows, inspect live processes before updating or resuming, preserve old
runs, and follow the README through final packaging. Keep the new final 100
seeds unused until final-choice.json is frozen. Do not use their results to
reselect models or continue training. Report coverage (>=80%), raw mean (>=300),
their conjunction, and strict all-finished/300 results separately. Include failure
reasons, saturated-action fraction and finished-episode returns in diagnostics.

A completed run with unmet goals exits 0; a safe interruption exits 130. Report
completed-update steps separately from crash loss bounds. Do not publish newly
trained models or logs without an explicit user request. Preserve the single
train.py deliverable and legacy A2C/gSDE loading paths, English standalone
comments, and two blank lines between complete modules. Cite algorithm sources.
