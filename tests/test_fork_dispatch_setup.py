"""The fork's dispatcher targets this fork's tournament workflow, not upstream's repository."""

from scripts import cronjob_dispatch_setup as upstream
from scripts.fork_dispatch_setup import build_fork_payload


def test_payload_dispatches_the_forks_tournament_workflow() -> None:
    payload = build_fork_payload("tok")
    assert payload["url"] == (
        "https://api.github.com/repos/MichaelMBishop/nostream-metaculus-bot/actions/workflows/"
        "run_bot_on_tournament.yaml/dispatches"
    )
    assert payload["schedule"]["minutes"] == [2, 32]
    assert payload["extendedData"]["body"] == '{"ref":"main"}'
    assert payload["extendedData"]["headers"]["Authorization"] == "Bearer tok"


def test_building_the_fork_payload_leaves_upstreams_url_alone() -> None:
    build_fork_payload("tok")
    assert "No-Stream/" in upstream.GITHUB_DISPATCH_URL
