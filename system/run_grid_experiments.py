#!/usr/bin/env python3
"""
Grid-search eksperymentów MCMT:
  Model × Tracker SCT × Sekwencja → main.py → TrackEval → results/experiments_summary.csv

Uruchomienie (z katalogu system/, środowisko mtmc_env):
  # SMOKE TEST (zalecane przed pełnym gridem):
  python run_grid_experiments.py --smoke

  python run_grid_experiments.py --dry-run
  python run_grid_experiments.py --models yolov8m.pt yolov26m.pt --trackers BoT-SORT ByteTrack
"""

from __future__ import annotations

import argparse
import csv
import copy
import logging
import re
import shutil
import subprocess
import sys
import time
from itertools import product
from pathlib import Path
from typing import Any

from src.config import TRACKER_BACKENDS, load_config, resolve_tracker_backend, save_config

SYSTEM_DIR = Path(__file__).resolve().parent
CONFIG_PATH = SYSTEM_DIR / "configs" / "experiment.yaml"
CONFIG_BACKUP_PATH = SYSTEM_DIR / "configs" / "experiment.yaml.grid_backup"
SUMMARY_CSV = SYSTEM_DIR / "results" / "experiments_summary.csv"
LOG_PATH = SYSTEM_DIR / "results" / "grid_experiments.log"


DEFAULT_MODELS = ["yolov8m.pt", "yolov10m.pt", "yolov11m.pt", "yolov26m.pt", "rt-detr-l.pt"]
DEFAULT_TRACKERS = [
    "ByteTrack",
    "BoT-SORT",
    "Deep_OC-SORT",
    "Hybrid-SORT",
    "BoostTrack++",
    "TrackTrack",
]
DEFAULT_SEQ_NAMES = ["MOT17-02-FRCNN", "MOT17-04-FRCNN"]


MIN_DETECTION_CONF = 0.1

SUMMARY_FIELDS = [
    "Model",
    "Tracker",
    "Sequence",
    "HOTA",
    "MOTA",
    "IDF1",
    "IDSW",
    "AssA",
    "DetA",
    "Avg_Inference_Time",
    "FPS",
    "Status",
    "Error",
]


def setup_logging() -> None:
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(LOG_PATH, encoding="utf-8"),
        ],
    )


def sanitize_tracker_folder(model: str, tracker: str) -> str:
    """Unikalna nazwa katalogu wyników TrackEval (bez spacji / znaków specjalnych)."""
    model_stem = Path(model).stem.replace(".", "_")
    tracker_stem = re.sub(r"[^A-Za-z0-9_+-]+", "_", tracker)
    return f"{model_stem}__{tracker_stem}"


def count_sequence_frames(seq_name: str, config: dict[str, Any]) -> int:
    ds = config["datasets"]["mot17"]
    pattern = ds["video_pattern"].format(split=ds.get("split", "train"), seq_name=seq_name)
    img_dir = Path(pattern.split("%")[0])
    if not img_dir.is_dir():
        img_dir = SYSTEM_DIR / img_dir
    if not img_dir.is_dir():
        return 0
    return sum(1 for p in img_dir.iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp"})


def apply_experiment_config(
    base_config: dict[str, Any],
    *,
    model: str,
    tracker: str,
    seq_name: str,
    tracker_folder: str,
) -> dict[str, Any]:
    cfg = copy.deepcopy(base_config)
    cfg.setdefault("detection", {})
    cfg.setdefault("sct_tracker", {})
    cfg.setdefault("experiment", {})
    cfg.setdefault("datasets", {}).setdefault("mot17", {})

    cfg["detection"]["model"] = model
    # ZABEZPIECZENIE PROGÓW: słabe detekcje dla drugiej fazy asocjacji (ByteTrack / BoT-SORT)
    cfg["detection"]["conf"] = MIN_DETECTION_CONF

    cfg["sct_tracker"]["name"] = tracker
    cfg["experiment"]["dataset"] = "mot17"
    cfg["experiment"]["seq_name"] = seq_name
    cfg["experiment"]["tracker_name"] = tracker_folder
    cfg["experiment"]["run_all_sequences"] = False

    sequences = list(cfg["datasets"]["mot17"].get("sequences", []))
    if seq_name not in sequences:
        sequences.append(seq_name)
        cfg["datasets"]["mot17"]["sequences"] = sequences

    return cfg


def run_subprocess(cmd: list[str], step_name: str) -> tuple[int, float]:
    logging.info("  → %s: %s", step_name, " ".join(cmd))
    t0 = time.perf_counter()
    try:
        result = subprocess.run(
            cmd,
            cwd=str(SYSTEM_DIR),
            check=False,
            capture_output=False,
        )
        elapsed = time.perf_counter() - t0
        if result.returncode != 0:
            raise RuntimeError(f"{step_name} zakończył się kodem {result.returncode}")
        return result.returncode, elapsed
    except Exception:
        elapsed = time.perf_counter() - t0
        raise


def _to_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        text = str(value).strip()
        if not text or text.lower() in {"nan", "none", "-"}:
            return None
        return float(text)
    except (TypeError, ValueError):
        return None


def parse_summary_txt(path: Path) -> dict[str, float]:
    """Parsuje pedestrian_summary.txt (spacje, wiersz nagłówka + wartości)."""
    metrics: dict[str, float] = {}
    if not path.exists():
        return metrics
    lines = [ln.strip() for ln in path.read_text(encoding="utf-8").splitlines() if ln.strip()]
    if len(lines) < 2:
        return metrics
    headers = lines[0].split()
    values = lines[1].split()
    for key, val in zip(headers, values):
        num = _to_float(val)
        if num is not None:
            metrics[key] = num
    return metrics


def parse_detailed_csv(path: Path, seq_name: str) -> dict[str, float]:
    """Parsuje pedestrian_detailed.csv i zwraca wiersz dla danej sekwencji (lub COMBINED)."""
    metrics: dict[str, float] = {}
    if not path.exists():
        return metrics
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
    chosen = None
    for row in rows:
        if row.get("seq") == seq_name:
            chosen = row
            break
    if chosen is None:
        for row in rows:
            if row.get("seq") in {"COMBINED", "COMBINED_SEQ"}:
                chosen = row
                break
    if chosen is None and rows:
        chosen = rows[0]
    if not chosen:
        return metrics
    for key, val in chosen.items():
        if key == "seq":
            continue
        num = _to_float(val)
        if num is not None:
            metrics[key] = num
    return metrics


def find_trackeval_metrics(tracker_folder: str, seq_name: str) -> dict[str, float]:
    """
    Szuka wyników TrackEval w typowych lokalizacjach:
      results/{tracker}/pedestrian_summary.txt
      results/{tracker}/pedestrian_detailed.csv
    """
    root = SYSTEM_DIR / "results" / tracker_folder
    candidates_summary = [
        root / "pedestrian_summary.txt",
        root / "data" / "pedestrian_summary.txt",
    ]
    candidates_detailed = [
        root / "pedestrian_detailed.csv",
        root / "data" / "pedestrian_detailed.csv",
    ]

    # Dodatkowe przeszukanie (TrackEval czasem zapisuje głębiej)
    if root.exists():
        candidates_summary.extend(root.rglob("pedestrian_summary.txt"))
        candidates_detailed.extend(root.rglob("pedestrian_detailed.csv"))

    metrics: dict[str, float] = {}
    for path in candidates_summary:
        if path.exists():
            metrics.update(parse_summary_txt(path))
            logging.info("  → metryki summary: %s", path)
            break
    for path in candidates_detailed:
        if path.exists():
            seq_metrics = parse_detailed_csv(path, seq_name)
            # detailed ma pierwszeństwo dla per-seq
            metrics.update(seq_metrics)
            logging.info("  → metryki detailed: %s", path)
            break
    return metrics


def append_summary_row(row: dict[str, Any]) -> None:
    SUMMARY_CSV.parent.mkdir(parents=True, exist_ok=True)
    write_header = not SUMMARY_CSV.exists()
    with open(SUMMARY_CSV, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=SUMMARY_FIELDS, extrasaction="ignore")
        if write_header:
            writer.writeheader()
        writer.writerow({k: row.get(k, "") for k in SUMMARY_FIELDS})


def empty_metrics_row(
    model: str,
    tracker: str,
    seq_name: str,
    *,
    status: str,
    error: str = "",
    avg_time: float | None = None,
    fps: float | None = None,
) -> dict[str, Any]:
    return {
        "Model": model,
        "Tracker": tracker,
        "Sequence": seq_name,
        "HOTA": "",
        "MOTA": "",
        "IDF1": "",
        "IDSW": "",
        "AssA": "",
        "DetA": "",
        "Avg_Inference_Time": "" if avg_time is None else f"{avg_time:.4f}",
        "FPS": "" if fps is None else f"{fps:.4f}",
        "Status": status,
        "Error": error,
    }


def run_single_experiment(
    base_config: dict[str, Any],
    *,
    model: str,
    tracker: str,
    seq_name: str,
    index: int,
    total: int,
    dry_run: bool = False,
    max_frames: int | None = None,
    skip_eval: bool = False,
) -> dict[str, Any]:
    logging.info("=" * 72)
    logging.info("Uruchamianie eksperymentu %d z %d...", index, total)
    logging.info("Model=%s | Tracker=%s | Sequence=%s", model, tracker, seq_name)
    if max_frames is not None:
        logging.info("Limit klatek: %d", max_frames)

    backend = resolve_tracker_backend(tracker)
    if backend is None:
        msg = (
            f"Tracker '{tracker}' nie jest wspierany "
            f"(dostępne: {list(TRACKER_BACKENDS.keys())})."
        )
        logging.warning(msg)
        row = empty_metrics_row(model, tracker, seq_name, status="skipped", error=msg)
        append_summary_row(row)
        return row

    tracker_folder = sanitize_tracker_folder(model, tracker)
    cfg = apply_experiment_config(
        base_config,
        model=model,
        tracker=tracker,
        seq_name=seq_name,
        tracker_folder=tracker_folder,
    )

    if dry_run:
        logging.info("[dry-run] detection.conf=%s tracker_folder=%s", cfg["detection"]["conf"], tracker_folder)
        row = empty_metrics_row(model, tracker, seq_name, status="dry_run")
        append_summary_row(row)
        return row

    save_config(cfg, CONFIG_PATH)

    n_frames = count_sequence_frames(seq_name, cfg)
    if max_frames is not None and n_frames > 0:
        n_frames = min(n_frames, max_frames)

    main_elapsed = None
    try:
        main_cmd = [
            sys.executable,
            "main.py",
            "--config",
            str(CONFIG_PATH),
            "--dataset",
            "mot17",
            "--seq",
            seq_name,
        ]
        if max_frames is not None:
            main_cmd.extend(["--max-frames", str(max_frames)])

        _, main_elapsed = run_subprocess(main_cmd, "main.py")

        if not skip_eval:
            run_subprocess(
                [
                    sys.executable,
                    "run_eval.py",
                    "--config",
                    str(CONFIG_PATH),
                    "--dataset",
                    "mot17",
                    "--seq",
                    seq_name,
                    "--tracker-name",
                    tracker_folder,
                ],
                "run_eval.py",
            )
    except Exception as exc:
        logging.exception("Eksperyment nieudany — przechodzę dalej.")
        row = empty_metrics_row(
            model,
            tracker,
            seq_name,
            status="failed",
            error=str(exc),
            avg_time=main_elapsed,
            fps=(n_frames / main_elapsed) if (main_elapsed and n_frames) else None,
        )
        append_summary_row(row)
        return row

    metrics = {} if skip_eval else find_trackeval_metrics(tracker_folder, seq_name)
    fps = (n_frames / main_elapsed) if (main_elapsed and n_frames > 0) else None

    if skip_eval:
        status = "ok_smoke"
        error = ""
    elif metrics:
        status = "ok"
        error = ""
    else:
        status = "eval_missing_metrics"
        error = "Nie znaleziono plików summary/detailed TrackEval"

    row = {
        "Model": model,
        "Tracker": tracker,
        "Sequence": seq_name,
        "HOTA": metrics.get("HOTA", ""),
        "MOTA": metrics.get("MOTA", ""),
        "IDF1": metrics.get("IDF1", ""),
        "IDSW": metrics.get("IDSW", ""),
        "AssA": metrics.get("AssA", ""),
        "DetA": metrics.get("DetA", ""),
        "Avg_Inference_Time": f"{main_elapsed:.4f}" if main_elapsed is not None else "",
        "FPS": f"{fps:.4f}" if fps is not None else "",
        "Status": status,
        "Error": error,
    }
    append_summary_row(row)
    logging.info(
        "Wynik: HOTA=%s MOTA=%s IDF1=%s IDSW=%s | time=%.2fs FPS=%s | status=%s",
        row["HOTA"],
        row["MOTA"],
        row["IDF1"],
        row["IDSW"],
        main_elapsed or -1,
        row["FPS"],
        status,
    )
    return row


def build_grid(models: list[str], trackers: list[str], seq_names: list[str]) -> list[tuple[str, str, str]]:
    return list(product(models, trackers, seq_names))


def main() -> int:
    parser = argparse.ArgumentParser(description="Grid-search MCMT (Model × Tracker × Sequence)")
    parser.add_argument("--models", nargs="+", default=DEFAULT_MODELS)
    parser.add_argument("--trackers", nargs="+", default=DEFAULT_TRACKERS)
    parser.add_argument("--seqs", nargs="+", default=None, help="Sekwencje MOT17 (domyślnie z experiment.yaml)")
    parser.add_argument("--dry-run", action="store_true", help="Tylko wypisz siatkę, bez uruchamiania")
    parser.add_argument("--limit", type=int, default=None, help="Ogranicz liczbę eksperymentów (debug)")
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="Szybki smoke test: YOLOv8m + ByteTrack + MOT17-02-FRCNN, 30 klatek",
    )
    parser.add_argument("--max-frames", type=int, default=None, help="Limit klatek przekazywany do main.py")
    parser.add_argument(
        "--skip-eval",
        action="store_true",
        help="Pomiń TrackEval (przydatne w smoke, gdy chcesz tylko sprawdzić tracking)",
    )
    args = parser.parse_args()

    setup_logging()

    if not CONFIG_PATH.exists():
        logging.error("Brak pliku konfiguracyjnego: %s", CONFIG_PATH)
        return 1

    base_config = load_config(CONFIG_PATH)
    # Kopia zapasowa — przywracamy po gridzie
    shutil.copy2(CONFIG_PATH, CONFIG_BACKUP_PATH)

    models = args.models
    trackers = args.trackers
    seq_names = args.seqs
    max_frames = args.max_frames
    skip_eval = args.skip_eval

    if args.smoke:
        models = ["yolov8m.pt"]
        trackers = ["ByteTrack"]
        seq_names = ["MOT17-02-FRCNN"]
        max_frames = 30 if max_frames is None else max_frames
        logging.info(
            "TRYB SMOKE: model=%s tracker=%s seq=%s max_frames=%d",
            models[0],
            trackers[0],
            seq_names[0],
            max_frames,
        )

    if not seq_names:
        seq_names = list(base_config.get("datasets", {}).get("mot17", {}).get("sequences", DEFAULT_SEQ_NAMES))
        if not seq_names:
            seq_names = DEFAULT_SEQ_NAMES

    grid = build_grid(models, trackers, seq_names)
    if args.limit is not None:
        grid = grid[: args.limit]

    total = len(grid)
    logging.info(
        "Siatka: %d eksperymentów (%d modeli × %d trackerów × %d sekwencji)",
        total,
        len(models),
        len(trackers),
        len(seq_names),
    )
    logging.info("MIN detection.conf = %s", MIN_DETECTION_CONF)
    logging.info("Podsumowanie CSV: %s", SUMMARY_CSV)

    try:
        for i, (model, tracker, seq_name) in enumerate(grid, start=1):
            run_single_experiment(
                base_config,
                model=model,
                tracker=tracker,
                seq_name=seq_name,
                index=i,
                total=total,
                dry_run=args.dry_run,
                max_frames=max_frames,
                skip_eval=skip_eval,
            )
    finally:
        if CONFIG_BACKUP_PATH.exists():
            shutil.copy2(CONFIG_BACKUP_PATH, CONFIG_PATH)
            logging.info("Przywrócono bazowy experiment.yaml z kopii zapasowej.")

    logging.info("Zakończono grid-search. Wyniki: %s", SUMMARY_CSV)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
