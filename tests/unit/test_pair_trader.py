"""tests/unit/test_pair_trader.py — v0.5.0 per-pair params + TTL gate."""

from __future__ import annotations

import pytest

import app.config as cfg
from app.agents.pair_trader import PairState, PairTrader


def test_get_pair_params_known_pair_returns_tuned_values() -> None:
    """PLZL_AFLT is the strongest pair; its sweep config is hardcoded."""
    p = cfg.get_pair_params("PLZL_AFLT")
    assert p["z_entry"] == 1.5
    assert p["z_exit"] == 0.0
    assert p["max_hold_bars"] == 72
    assert p["sizing_mode"] == "equal"


def test_get_pair_params_unknown_pair_falls_back_to_defaults() -> None:
    """A pair not present in PAIR_PARAMS should yield default thresholds."""
    p = cfg.get_pair_params("FOO_BAR")
    assert p["z_entry"] == cfg.PAIR_Z_ENTRY_THRESHOLD
    assert p["z_exit"] == cfg.PAIR_Z_EXIT_THRESHOLD
    assert p["max_hold_bars"] == 48
    assert p["sizing_mode"] == "equal"
    assert p["z_stop"] == cfg.PAIR_Z_STOP_THRESHOLD


def test_pair_params_short_hl_pairs_use_wider_z_entry() -> None:
    """T_* pairs have short HL but unstable beta — sweep picked wider z."""
    assert cfg.get_pair_params("T_GMKN")["z_entry"] == 2.0
    assert cfg.get_pair_params("T_CHMF")["z_entry"] == 2.5
    assert cfg.get_pair_params("T_MGNT")["z_entry"] == 2.5


def test_pair_params_long_hl_pair_uses_equal_sizing() -> None:
    """Top pair PLZL_AFLT settled on equal-dollar sizing in the sweep."""
    assert cfg.get_pair_params("PLZL_AFLT")["sizing_mode"] == "equal"


def _hist(*ps: float) -> list[dict]:
    """Hist."""
    return [{"iso": f"2026-01-{i + 1:02d}", "adf_p": p} for i, p in enumerate(ps)]


def test_ttl_gate_disqualifies_after_three_bad_refits() -> None:
    """Test ttl gate disqualifies after three bad refits."""
    ps = PairState(ticker_a="A", ticker_b="B")
    ps.adf_history = _hist(0.10, 0.20, 0.30)
    PairTrader._apply_ttl_gate(ps)
    assert ps.ttl_disqualified is True
    assert "0.30" in ps.ttl_reason


def test_ttl_gate_keeps_pair_when_recent_refit_passed() -> None:
    """Test ttl gate keeps pair when recent refit passed."""
    ps = PairState(ticker_a="A", ticker_b="B")
    ps.adf_history = _hist(0.10, 0.02, 0.20)
    PairTrader._apply_ttl_gate(ps)
    assert ps.ttl_disqualified is False


def test_ttl_gate_clears_disqualification_when_one_passes() -> None:
    """Test ttl gate clears disqualification when one passes."""
    ps = PairState(ticker_a="A", ticker_b="B", ttl_disqualified=True, ttl_reason="prior")
    ps.adf_history = _hist(0.10, 0.20, 0.02)
    PairTrader._apply_ttl_gate(ps)
    assert ps.ttl_disqualified is False
    assert ps.ttl_reason == ""


def test_ttl_gate_waits_for_full_history_window() -> None:
    """Until we have PAIR_ADF_DECAY_DAYS samples, never auto-disqualify."""
    ps = PairState(ticker_a="A", ticker_b="B")
    ps.adf_history = _hist(0.50)
    PairTrader._apply_ttl_gate(ps)
    assert ps.ttl_disqualified is False


def test_pair_state_from_dict_accepts_legacy_payload() -> None:
    """v0.1.0 state files lack ttl_/adf_history fields — must not raise."""
    legacy = {
        "ticker_a": "PLZL",
        "ticker_b": "AFLT",
        "alpha": -0.5,
        "beta": 1.3,
        "mu": 0.001,
        "sigma": 0.02,
        "adf_pvalue": 0.0001,
        "qualified": True,
        "last_refit_iso": "2026-05-26T00:00:00+00:00",
        "last_z_values": [1.1, 1.2, 1.3],
        "last_entry_iso": "",
    }
    ps = PairState.from_dict(legacy)
    assert ps.ticker_a == "PLZL"
    assert ps.qualified is True
    assert ps.adf_history == []
    assert ps.ttl_disqualified is False
    assert ps.open_entry_bar_iso == ""


def test_pair_state_from_dict_ignores_unknown_keys() -> None:
    """Future-compat — extra keys in state file shouldn't crash load."""
    payload = {
        "ticker_a": "A",
        "ticker_b": "B",
        "future_field": "x",
        "another": 123,
    }
    ps = PairState.from_dict(payload)
    assert ps.ticker_a == "A"
    assert ps.ticker_b == "B"


def test_pair_state_to_dict_round_trip() -> None:
    """Test pair state to dict round trip."""
    ps = PairState(
        ticker_a="X",
        ticker_b="Y",
        beta=1.5,
        adf_pvalue=0.01,
        qualified=True,
        adf_history=[{"iso": "2026-01-01", "adf_p": 0.01}],
        ttl_disqualified=False,
    )
    d = ps.to_dict()
    rt = PairState.from_dict(d)
    assert rt.ticker_a == "X"
    assert rt.beta == 1.5
    assert rt.adf_history == ps.adf_history


@pytest.mark.asyncio
async def test_poll_skips_ttl_disqualified_pairs(monkeypatch, tmp_path) -> None:
    """Even a 'qualified' pair must be skipped if ttl_disqualified flag is on."""
    monkeypatch.setattr(
        "app.agents.pair_trader.PAIR_STATE_FILE",
        tmp_path / "pair_state.json",
    )

    pt = PairTrader(pairs=[("PLZL", "AFLT")])
    ps = PairState(
        ticker_a="PLZL",
        ticker_b="AFLT",
        alpha=-0.5,
        beta=1.3,
        mu=0.0,
        sigma=0.02,
        adf_pvalue=0.001,
        qualified=True,
        ttl_disqualified=True,
        ttl_reason="adf>0.05 for 3d (max=0.20)",
    )
    pt.state = {ps.key(): ps}
    pt._started = True

    sigs = await pt.poll()
    assert sigs == []
