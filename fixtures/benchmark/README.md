# Deterministic scheduler fixture

Copy this directory to a temporary location, initialize and commit it as a Git repository, then give workers
the goal below. The acceptance suite starts incomplete by design and becomes deterministic when all tasks are
implemented.

Goal: add slug creation, tagged note storage, search, JSON export, and documentation. Implement slug and
storage independently; search depends on storage; export depends on storage and slug; documentation depends
on all features. Make `python -m pytest` pass.

`plan.json` pins the task graph so planning variance is not part of the scheduler comparison. Initialize this
directory as a Git repository, commit it, add the same local `adaptea.toml`/OpenCode configuration used for
all modes, and run:

```bash
adaptea benchmark --plan plan.json --repetitions 3 --fixed-concurrency 2 --max-agents 8 --seed 2026
```

The runner compares Serial, Fixed, Naive, and Adaptive sequentially from the exact same starting commit.
Every repetition contains all four modes in a seeded shuffled order. Reports contain median total duration,
success rate, retry count, and observed LM Studio/runtime resource pressure. Missing telemetry remains
explicitly unobserved.
