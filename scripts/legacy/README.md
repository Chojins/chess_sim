# Historical experiment launchers

These launchers record the earlier residual-RL and capture experiments. They
contain fixed checkpoint paths, seed ranges, output names and, in some cases,
waits for another historical experiment's status file. They are retained for
reference rather than recommended for a new training run.

Each launcher changes to the repository root before invoking the reusable tools,
so its calibration and output paths retain their original meaning. For example:

```bash
bash scripts/legacy/run_capture_supervised.sh
```

This example runs a historical experiment; it is not a dry run. Existing output
guards and status dependencies still apply. Review the script before running it.

The experiment history is in [RL_APPROACH.md](../../RL_APPROACH.md). Current
baseline results and limitations are in [the v29 record](../../docs/baselines/v29.md).
