#!/usr/bin/env python3
"""
Wrapper TrackEval z kompatybilnością NumPy ≥ 1.24.

TrackEval używa usuniętych aliasów (np.float / np.int / np.bool).
Ten skrypt aplikuje patch w tym samym procesie, potem uruchamia
oryginalny TrackEval/scripts/run_mot_challenge.py.
"""

from __future__ import annotations

import runpy
import sys
from pathlib import Path


def _patch_numpy_aliases() -> None:
    import numpy as np

    if not hasattr(np, "float"):
        np.float = np.float64  # type: ignore[attr-defined, assignment]
    if not hasattr(np, "int"):
        np.int = np.int_  # type: ignore[attr-defined, assignment]
    if not hasattr(np, "bool"):
        np.bool = bool  # type: ignore[attr-defined, assignment]


def main() -> None:
    _patch_numpy_aliases()

    system_dir = Path(__file__).resolve().parent
    trackeval_script = system_dir / "TrackEval" / "scripts" / "run_mot_challenge.py"
    if not trackeval_script.is_file():
        raise FileNotFoundError(
            f"Brak TrackEval: {trackeval_script}. "
            "Sklonuj: git clone https://github.com/JonathonLuiten/TrackEval.git"
        )

    # Zachowaj argumenty CLI (bez nazwy tego wrappera) dla run_mot_challenge.py
    sys.argv = [str(trackeval_script), *sys.argv[1:]]
    # TrackEval skrypty zakładają, że katalog TrackEval jest na sys.path
    trackeval_root = str(system_dir / "TrackEval")
    if trackeval_root not in sys.path:
        sys.path.insert(0, trackeval_root)

    runpy.run_path(str(trackeval_script), run_name="__main__")


if __name__ == "__main__":
    main()
