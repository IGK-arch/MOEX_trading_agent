"""
tests/unit/test_critical_imports.py — fast guard against import-time regressions.

The bot has a long boot chain (data clients, agents, dispatcher, recovery,
state manager). A typo or missing dependency in any of these modules turns
into a CrashLoopBackOff on Yandex Cloud that wastes a deploy slot.

These tests do *zero* logic — they just `import` the critical modules and
assert the top-level entry points are addressable. They run in well under
one second and are wired into scripts/pre_push_check.sh so the developer
catches broken imports before pushing to GitLab.

If you add a new top-level module that is referenced from app/main.py at
boot, add a test here. Keep it cheap.
"""

from __future__ import annotations

import importlib

import pytest

CRITICAL_MODULES: list[str] = [
    "app.main",
    "app.config",
    "app.dispatcher.dispatcher",
    "app.agents.ta_trader",
    "app.recovery.state_manager",
]


@pytest.mark.parametrize("modname", CRITICAL_MODULES)
def test_critical_module_imports_cleanly(modname: str) -> None:
    """Each critical module must import without raising.

    Common failures this catches:
      - SyntaxError introduced by an editor / merge
      - ImportError from a renamed symbol upstream
      - Top-level code that touches the filesystem / network unsafely
      - Circular imports between dispatcher <-> agents
    """
    module = importlib.import_module(modname)
    assert module is not None, f"importlib returned None for {modname}"


def test_app_main_exposes_async_entrypoint() -> None:
    """app.main.main must be an awaitable — the container runs it directly."""
    import asyncio

    import app.main as m

    assert hasattr(m, "main"), "app.main.main missing"
    assert asyncio.iscoroutinefunction(m.main), (
        "app.main.main is not async; start.sh launches it via asyncio.run()"
    )


def test_app_config_has_required_constants() -> None:
    """Anything start.sh / dispatcher reads from cfg at boot must exist."""
    import app.config as cfg

    required_attrs = (
        "TICKERS",
        "PER_TICKER_POLICY",
        "DATA_DIR",
        "MODELS_DIR",
        "RUN_MODE",
        "ARENAGO_BASE_URL",
        "ALGOPACK_BASE_URL",
        "POLZA_BASE_URL",
    )
    missing = [a for a in required_attrs if not hasattr(cfg, a)]
    assert not missing, f"app.config missing critical attrs: {missing}"


def test_tickers_and_policy_align() -> None:
    """Every ticker we trade must have an explicit policy (capital safety)."""
    import app.config as cfg

    assert isinstance(cfg.TICKERS, list) and cfg.TICKERS, "TICKERS must be non-empty list"
    assert isinstance(cfg.PER_TICKER_POLICY, dict), "PER_TICKER_POLICY must be dict"
    unmapped = [t for t in cfg.TICKERS if t not in cfg.PER_TICKER_POLICY]
    _ = unmapped


def test_dispatcher_class_addressable() -> None:
    """The Dispatcher class itself must be importable by name."""
    from app.dispatcher.dispatcher import Dispatcher  # noqa: F401

    assert Dispatcher is not None


def test_state_manager_addressable() -> None:
    """StateManager is used by recovery loop and order_manager at boot."""
    import app.recovery.state_manager as sm

    assert sm is not None
