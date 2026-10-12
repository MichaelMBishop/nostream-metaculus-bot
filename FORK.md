# Fork notes

This repo is a fork of [No-Stream/nostreambot-metaculus-bot](https://github.com/No-Stream/nostreambot-metaculus-bot).
`AGENTS.md` and `docs/` are upstream's and describe upstream's deployment. This file lists what the fork
changes, and what to check when syncing.

## Fork-only changes

| Change | Where | PR |
|---|---|---|
| numpyro and import-linter skipped on Intel macOS; secret scan reads PR commits on a private repo | `pyproject.toml`, `uv.lock`, `ci.yaml` | #1 |
| MiniBench cron commented out | `run_bot_on_minibench.yaml`, `tests/test_workflow_reliability.py` | #2 |
| cron-job.org dispatcher pointed at this fork | `scripts/fork_dispatch_setup.py` | #5 |
| Gap-fill v2 off in the bot workflows | `GAP_FILL_V2_ENABLED: 'false'` in six workflows | #6 |
| GitHub's own cron dropped from the tournament workflow | `run_bot_on_tournament.yaml`, `tests/test_workflow_reliability.py` | #7 |
| Supply-probe test date fix; pyjwt, urllib3 and virtualenv bumps (drop if upstream fixes the same) | `tests/test_supply_probe.py`, `uv.lock` | #8 |
| Blend aggregation in place of the median | see below | #9 |
| `report_status` job dropped from the tournament workflow (saves ~1,400 Actions min/month) | `run_bot_on_tournament.yaml`, `tests/test_forecast_alert_workflows.py` | #14 |

## Blend aggregation

With `FORK_BLEND_AGGREGATION_ENABLED: 'true'` (set in `run_bot_on_tournament.yaml`), the bot publishes
`0.7 * median + 0.3 * mean` of the member forecasts instead of the median:
- binary: per probability;
- multiple choice: per option, then clamped and renormalized;
- numeric, discrete and date: pointwise in CDF space.

With three members, that weights the low, middle and high member 0.1, 0.8 and 0.1. Mantic per-bin
questions keep upstream's mean. Unset or `'false'` restores the plain median.

The code is in `metaculus_bot/fork_aggregation.py` and `tests/test_fork_aggregation.py`. It touches
upstream code in two places:
- `aggregation_strategies.py`: the `BLEND` strategy value, and one branch in each `combine_*` function.
- `aggregation_pipeline.py`: two lines in `_numeric_combine_strategy`. Every median path passes through
  that seam, so this is the only switch.

## Syncing upstream

1. `git fetch upstream`, branch off `main`, then `git merge upstream/main`.
2. Expect conflicts in three places:
   - the `GAP_FILL_V2_ENABLED` lines in the workflows;
   - the tail of `run_bot_on_tournament.yaml`, if upstream edits its `report_status` job. Keep it deleted;
   - occasionally the two blend hooks above. Upstream edits `aggregation_pipeline.py` often, usually
     inside large refactors.
3. Run the free gates. `tests/test_fork_aggregation.py` fails if a merge drops the hook, routes a
   median around `_numeric_combine_strategy`, or loses the workflow flag.
4. After merging, check the first forecasting run's log for the blend:
   - numeric: `NUMERIC_AGGREGATE ... method=blend`;
   - binary: `STACKING base combine: binary blend of [...]`.
