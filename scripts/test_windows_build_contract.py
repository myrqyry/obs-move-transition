#!/usr/bin/env python3
"""Guard the Windows-only Move Transition sources against MSVC build regressions.

This is a fast source-level check. Windows GitHub Actions must still compile
and link the plugin with MSVC; this check does not replace that build.
"""

import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def check() -> list[str]:
    main = (ROOT / "move-transition.c").read_text(encoding="utf-8")
    nvidia = (ROOT / "nvidia-move-filter.c").read_text(encoding="utf-8")
    directshow = (ROOT / "move-directshow-filter.cpp").read_text(encoding="utf-8")
    errors: list[str] = []

    # NVIDIA settings are persisted as integer feature IDs; keep Bottom at 6.
    expected = "FEATURE_BOUNDINGBOX_BOTTOM"
    constants = {
        name: int(value)
        for name, value in re.findall(
            r"^\s*#define\s+(FEATURE_BOUNDINGBOX_\w+)\s+(\d+)\b",
            nvidia,
            flags=re.MULTILINE,
        )
    }
    if constants.get(expected) != 6:
        errors.append(f"NVIDIA {expected} must remain serialized feature ID 6")
    used = set(re.findall(r"\bFEATURE_BOUNDINGBOX_\w+\b", nvidia))
    undefined = sorted(used - set(constants))
    if undefined:
        errors.append(f"Undefined NVIDIA bounding-box feature identifiers: {undefined}")

    # The Windows registration functions live in separate C/C++ translation units.
    declarations = {
        "load_nvar": r"extern\s+bool\s+load_nvar\s*\(void\s*\)\s*;",
        "nvidia_move_filter": r"extern\s+struct\s+obs_source_info\s+nvidia_move_filter\s*;",
        "SetMoveDirectShowFilter": (
            r"extern\s+void\s+SetMoveDirectShowFilter\s*\("
            r"\s*struct\s+obs_source_info\s*\*\s*\w+\s*\)\s*;"
        ),
    }
    win_block = re.search(r"#ifdef\s+WIN32\s*\n(.*?)#endif", main, flags=re.DOTALL)
    if win_block is None:
        errors.append("Windows-only registration block missing")
    else:
        for name, declaration in declarations.items():
            if not re.search(declaration, win_block.group(1)):
                errors.append(f"Missing Windows-only declaration: {name}")

    if not re.search(r"\bbool\s+load_nvar\s*\(void\s*\)\s*\{", nvidia):
        errors.append("NVIDIA loader implementation missing")
    if not re.search(r"\bstruct\s+obs_source_info\s+nvidia_move_filter\s*=", nvidia):
        errors.append("NVIDIA source-info definition missing")
    if not re.search(
        r'extern\s+"C"\s*\{\s*void\s+SetMoveDirectShowFilter\s*\(',
        directshow,
    ):
        errors.append("DirectShow registration must retain its C linkage")

    return errors


if __name__ == "__main__":
    problems = check()
    if problems:
        for problem in problems:
            print(f"FAIL: {problem}")
        raise SystemExit(1)
    print("Windows registration and NVIDIA feature-ID checks passed")
