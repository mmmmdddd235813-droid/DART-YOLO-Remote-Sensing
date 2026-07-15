"""Backward-compatible alias for the paper's CSD module.

Older internal experiments used the file/module name `SPD`. The manuscript and
public model configuration use `CSD`, so new code should import from `CSD.py`.
"""

from CSD import CSD

SPD = CSD
SDZ = CSD

__all__ = ["SPD", "SDZ", "CSD"]
