# Agent handoff

Read README.md first. The current authorized implementation is the Issue #2
campaign: two PPO warm-start pilots (gamma 0.99 and 0.999), development-set
selection, independent-training-seed replication, bounded continuation/rollback,
and a frozen final holdout. Use --mode campaign on the target Windows machine
when instructed to train. The program owns stage selection and budget accounting;
do not replace it with two ad hoc commands or silently extend its total budget.

Reuse GWTower's validated Python environment. Install only missing dependencies;
do not reinstall PyTorch. Preserve raw rewards and the exact 1600-step Hardcore
environment. The reviewed public projects and their differences are in README;
their reported scores are not this project's measured scores.

Never train on the user's Mac. Only read-only evaluation, --dry-run and zero-update
checks are allowed there. Reading/editing the repository alone is not a request
to train. Do not call learn or optimizer.step during local validation; tests use
synthetic outcomes and mocked learners with Adam.step blocked.

On Windows, follow the campaign through evaluation and packaging. Verify live
processes before resuming, use the identical campaign directory/source/protocol,
and preserve old runs. Do not use holdout scores to select another model. Report
completed training steps separately from a crash's discarded-step upper bound.
A completed campaign can legitimately report goal_reached=false. Do not publish
new trained models or logs unless the user requests that separately.

Keep the single train.py source deliverable and teacher Agent.select_action API.
Comments are English; complete modules have two blank lines between them and
aligned standalone comments. Cite third-party algorithms. Run non-training
checks after edits and retain the original A2C route as a reproducible baseline.
