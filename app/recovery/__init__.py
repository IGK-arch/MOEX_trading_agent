"""
app/recovery — Atomic state persistence for cold-restart recovery.

PDF rules: "if your solution crashes for reasons unrelated to organiser
infrastructure, it will NOT be restarted by organisers". Therefore the bot
must (a) survive process kills via Docker `restart: unless-stopped`, and
(b) resume from /data without re-trading executed decisions.

This module owns one file: /data/recovery_state.json.

Writes are atomic (tmp file + fsync + rename) so a kill mid-write doesn't
corrupt the snapshot. Reads happen exactly once, at startup.
"""

from app.recovery.state_manager import (
    SCHEMA_VERSION,
    RecoverySnapshot,
    RecoveryStateManager,
    get_recovery_manager,
)

__all__ = [
    "RecoveryStateManager",
    "RecoverySnapshot",
    "SCHEMA_VERSION",
    "get_recovery_manager",
]
