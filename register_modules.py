"""Register DART-YOLO modules for Ultralytics YAML parsing."""

from __future__ import annotations

import inspect


def register_modules() -> None:
    """Expose custom layers in the namespace used by Ultralytics parse_model."""
    import ultralytics.nn.tasks as tasks

    from CSD import CSD, SDZ, SPD
    from DGAM import DGAM
    from DRFM import DRFM
    from LMSA import LMSA

    custom = {
        "CSD": CSD,
        "SDZ": SDZ,
        "SPD": SPD,
        "DRFM": DRFM,
        "LMSA": LMSA,
        "DGAM": DGAM,
    }
    for name, cls in custom.items():
        setattr(tasks, name, cls)

    _patch_parse_model(tasks)


def _patch_parse_model(tasks) -> None:
    """Extend the local Ultralytics parser without editing site-packages."""
    if getattr(tasks.parse_model, "_dart_yolo_patched", False):
        return

    src = inspect.getsource(tasks.parse_model)
    if "CSD," not in src:
        src = src.replace(
            "            A2C2f,\n        }\n    )",
            "            A2C2f,\n            CSD,\n            DRFM,\n        }\n    )",
        )
    if "            DRFM,\n        }\n    )  # modules with 'repeat' arguments" not in src:
        src = src.replace(
            "            A2C2f,\n        }\n    )  # modules with 'repeat' arguments",
            "            A2C2f,\n            DRFM,\n        }\n    )  # modules with 'repeat' arguments",
        )
    if "elif m in frozenset({LMSA, DGAM}):" not in src:
        src = src.replace(
            "        elif m is AIFI:\n            args = [ch[f], *args]",
            "        elif m in frozenset({LMSA, DGAM}):\n"
            "            c2 = ch[f]\n"
            "            args = [ch[f], *args[1:]]\n"
            "        elif m is AIFI:\n"
            "            args = [ch[f], *args]",
        )

    namespace = tasks.__dict__
    exec(compile(src, inspect.getsourcefile(tasks.parse_model) or "<parse_model>", "exec"), namespace)
    namespace["parse_model"]._dart_yolo_patched = True


__all__ = ["register_modules"]
