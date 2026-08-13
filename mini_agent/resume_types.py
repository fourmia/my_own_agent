from __future__ import annotations

from dataclasses import dataclass
from typing import Literal


PatchReconciliationStatus = Literal["not_executed", "executed", "indeterminate"]


@dataclass(frozen=True)
class PatchReconciliation:
    """一次 started-only Patch 的只读状态判断结果。"""

    run_id: str
    call_id: str
    tool: str
    status: PatchReconciliationStatus
    path: str | None
    reason: str
    started_sequence: int
    current_file_sha256: str | None
    expected_before_sha256: str | None
    expected_after_sha256: str | None
    current_workspace_sha256: str | None
    expected_before_workspace_sha256: str | None
    expected_after_workspace_sha256: str | None


__all__ = ["PatchReconciliation", "PatchReconciliationStatus"]
