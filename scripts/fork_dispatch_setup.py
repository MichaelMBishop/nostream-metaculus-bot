"""cron-job.org dispatcher for this fork (MichaelMBishop/nostream-metaculus-bot).

Upstream's ``cronjob_dispatch_setup.py`` hardcodes No-Stream's repository and dispatches three
workflows; this fork runs only the tournament workflow, so this wrapper builds that one job with
the fork's dispatch URL and reuses upstream's plan/apply machinery unchanged. Same keys as upstream:
CRONJOB_API_KEY and GH_DISPATCH_TOKEN (fine-grained PAT, Actions read/write on this repo only),
read from .env and never printed.

Usage:
    uv run python -m scripts.fork_dispatch_setup            # dry run: payload (token redacted) + plan
    uv run python -m scripts.fork_dispatch_setup --apply    # PAID: create/update the live job
"""

from __future__ import annotations

import json
import os
import sys
from collections.abc import Sequence
from typing import Any

from metaculus_bot.config import load_environment
from scripts import cronjob_dispatch_setup as upstream

FORK_DISPATCH_URL = (
    "https://api.github.com/repos/MichaelMBishop/nostream-metaculus-bot/actions/workflows/{workflow}/dispatches"
)
# Distinct title so plan_actions never matches a job created by upstream's script.
FORK_TOURNAMENT_JOB = upstream.DispatchJob(
    "nostream fork dispatch: tournament", "run_bot_on_tournament.yaml", (2, 32), enabled=True
)


def build_fork_payload(gh_token: str) -> dict[str, Any]:
    payload = upstream.build_job_payload(FORK_TOURNAMENT_JOB, gh_token)
    payload["url"] = FORK_DISPATCH_URL.format(workflow=FORK_TOURNAMENT_JOB.workflow_file)
    return payload


def main(argv: Sequence[str] | None = None) -> int:
    parser = upstream.build_parser()
    args = parser.parse_args(argv)
    load_environment()
    missing = upstream.missing_secrets()
    if args.apply and missing:
        parser.error(f"--apply needs {' and '.join(missing)} in the environment or .env")
    if not args.apply:
        print(f"--- {FORK_TOURNAMENT_JOB.title} ({FORK_TOURNAMENT_JOB.workflow_file}) ---")
        print(json.dumps(build_fork_payload(upstream.GH_DISPATCH_TOKEN_PLACEHOLDER), indent=2))
    if missing:
        print(f"Skipped reading the cron-job.org account: {' and '.join(missing)} not set, so there is no plan.")
        return 0
    payloads = [build_fork_payload(os.environ[upstream.GH_DISPATCH_TOKEN_ENV])]
    client = upstream.CronJobClient(os.environ[upstream.CRONJOB_API_KEY_ENV])
    try:
        actions = upstream.plan_actions(client, payloads)
        if args.apply:
            actions = upstream.apply_actions(client, actions)
    except upstream.CronJobApiError as exc:
        print(f"cron-job.org API error: HTTP {exc.status_code}: {exc.body}", file=sys.stderr)
        return 1
    for action in actions:
        print(upstream.render_action(action, applied=args.apply))
    return 0


if __name__ == "__main__":
    sys.exit(main())
