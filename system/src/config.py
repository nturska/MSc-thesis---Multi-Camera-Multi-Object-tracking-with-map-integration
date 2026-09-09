"""Loader konfiguracji eksperymentu i generator YAML trackerów SCT (Ultralytics)."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import numpy as np
import yaml

DEFAULT_CONFIG_PATH = Path(__file__).resolve().parent.parent / "configs" / "experiment.yaml"
BOTSORT_RUNTIME_PATH = Path(__file__).resolve().parent.parent / "configs" / "custom_botsort.yaml"
TRACKER_RUNTIME_PATH = Path(__file__).resolve().parent.parent / "configs" / "runtime_tracker.yaml"
SYSTEM_ROOT = Path(__file__).resolve().parent.parent

# Mapowanie nazw z siatki → źródło implementacji SCT (decoupled adapter)
# "ultralytics" = klasy z ultralytics.trackers.* ; "local" = system/trackers/*
TRACKER_SOURCES: dict[str, str] = {
    "ByteTrack": "ultralytics",
    "BoT-SORT": "ultralytics",
    "Deep_OC-SORT": "ultralytics",
    "Hybrid-SORT": "local",
    "BoostTrack++": "local",
    "TrackTrack": "ultralytics",
}

# Alias kompatybilności (stary kod / YAML Ultralytics)
TRACKER_BACKENDS: dict[str, str | None] = {
    "ByteTrack": "bytetrack",
    "BoT-SORT": "botsort",
    "Deep_OC-SORT": "deepocsort",
    "Hybrid-SORT": "hybridsort",
    "BoostTrack++": "boosttrackpp",
    "TrackTrack": "tracktrack",
}

# Nazwa metody SCT → sekcja w experiment.yaml
TRACKER_CONFIG_SECTIONS: dict[str, str] = {
    "ByteTrack": "bytetrack",
    "BoT-SORT": "botsort",
    "Deep_OC-SORT": "deepocsort",
    "Hybrid-SORT": "hybridsort",
    "BoostTrack++": "boosttrackpp",
    "TrackTrack": "tracktrack",
}

# Aliasów z artykułów → klucze YAML Ultralytics
_TRACKER_KEY_ALIASES: dict[str, str] = {
    "det_thresh": "track_high_thresh",
    "iou_threshold": "match_thresh",
    "alpha": "alpha_fixed_emb",
    "lambda": "inertia",
    "tau_p": "penalty_p",
    "tau_q": "penalty_q",
}


def load_config(path: str | os.PathLike | None = None) -> dict[str, Any]:
    """Wczytuje experiment.yaml (domyślnie system/configs/experiment.yaml)."""
    config_path = Path(path) if path is not None else DEFAULT_CONFIG_PATH
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    if not isinstance(config, dict):
        raise ValueError(f"Nieprawidłowy format konfiguracji: {config_path}")
    return config


def save_config(config: dict[str, Any], path: str | os.PathLike | None = None) -> str:
    """Zapisuje pełny config eksperymentu do YAML."""
    config_path = Path(path) if path is not None else DEFAULT_CONFIG_PATH
    config_path.parent.mkdir(parents=True, exist_ok=True)
    with open(config_path, "w", encoding="utf-8") as f:
        yaml.safe_dump(config, f, default_flow_style=False, sort_keys=False, allow_unicode=True)
    return str(config_path)


def resolve_tracker_source(tracker_name: str) -> str | None:
    """Zwraca 'ultralytics' | 'local' albo None, jeśli tracker nie jest wspierany."""
    if tracker_name in TRACKER_SOURCES:
        return TRACKER_SOURCES[tracker_name]
    return None


def resolve_tracker_backend(tracker_name: str) -> str | None:
    """
    Zwraca identyfikator backendu SCT albo None, jeśli tracker nie jest wspierany.
    Dla lokalnych trackerów zwraca 'hybridsort' / 'boosttrackpp' (nie None).
    """
    if tracker_name in TRACKER_BACKENDS:
        return TRACKER_BACKENDS[tracker_name]
    known = {v for v in TRACKER_BACKENDS.values() if v}
    if tracker_name.lower() in known:
        return tracker_name.lower()
    return None


def _normalize_tracker_keys(raw: dict[str, Any]) -> dict[str, Any]:
    """Mapuje aliasy z artykułów na klucze YAML Ultralytics (w obrębie jednej sekcji)."""
    out: dict[str, Any] = {}
    for key, value in raw.items():
        if key in _TRACKER_KEY_ALIASES:
            continue
        out[key] = value
    for key, value in raw.items():
        if key not in _TRACKER_KEY_ALIASES:
            continue
        canon = _TRACKER_KEY_ALIASES[key]
        # Jawny klucz Ultralytics w tej samej sekcji ma pierwszeństwo przed aliasem
        if canon not in raw:
            out[canon] = value
        if key == "det_thresh" and "new_track_thresh" not in raw:
            out.setdefault("new_track_thresh", value)
    return out


def get_sct_tracker_cfg(config: dict[str, Any], tracker_name: str | None = None) -> dict[str, Any]:
    """
    Składa słownik parametrów SCT:
      1) wspólna baza z sekcji botsort (progi asocjacji),
      2) nadpisania z sekcji dedykowanej (deepocsort / tracktrack / …),
      3) normalizacja aliasów (det_thresh → track_high_thresh itd.) per sekcja.
    """
    name = tracker_name or config.get("sct_tracker", {}).get("name", "BoT-SORT")
    section = TRACKER_CONFIG_SECTIONS.get(name)
    if section is None:
        section = name.lower().replace("-", "").replace(" ", "")

    merged: dict[str, Any] = {}
    if isinstance(config.get("botsort"), dict):
        merged.update(_normalize_tracker_keys(config["botsort"]))
    if name == "ByteTrack" and isinstance(config.get("bytetrack"), dict):
        merged.update(_normalize_tracker_keys(config["bytetrack"]))
    if section and section != "botsort" and isinstance(config.get(section), dict):
        merged.update(_normalize_tracker_keys(config[section]))

    return merged


def write_tracker_yaml(
    tracker_cfg: dict[str, Any],
    tracker_name: str = "BoT-SORT",
    output_path: str | os.PathLike | None = None,
) -> str:
    """
    Generuje YAML trackera SCT dla Ultralytics (legacy / narzędzia pomocnicze).
    Lokalne trackery (Hybrid-SORT, BoostTrack++) nie używają tego pliku w main.py.
    """
    source = resolve_tracker_source(tracker_name)
    backend = resolve_tracker_backend(tracker_name)
    if backend is None or source == "local":
        raise ValueError(
            f"write_tracker_yaml nie dotyczy trackera '{tracker_name}' "
            f"(source={source}). Wspierane Ultralytics: "
            f"{[k for k, s in TRACKER_SOURCES.items() if s == 'ultralytics']}"
        )

    tracker_cfg = _normalize_tracker_keys(dict(tracker_cfg))
    out = Path(output_path) if output_path is not None else TRACKER_RUNTIME_PATH

    payload: dict[str, Any] = {
        "tracker_type": backend,
        "track_high_thresh": tracker_cfg.get("track_high_thresh", 0.25),
        "track_low_thresh": tracker_cfg.get("track_low_thresh", 0.1),
        "new_track_thresh": tracker_cfg.get("new_track_thresh", 0.3),
        "track_buffer": tracker_cfg.get("track_buffer", 60),
        "match_thresh": tracker_cfg.get("match_thresh", 0.8),
        "fuse_score": tracker_cfg.get("fuse_score", True),
    }

    if backend in {"botsort", "deepocsort", "tracktrack"}:
        payload["gmc_method"] = tracker_cfg.get("gmc_method", "sparseOptFlow")
        payload["proximity_thresh"] = tracker_cfg.get("proximity_thresh", 0.5)
        payload["appearance_thresh"] = tracker_cfg.get("appearance_thresh", 0.25)
        payload["with_reid"] = tracker_cfg.get("with_reid", True)
        payload["model"] = tracker_cfg.get("model", "auto")

    if backend == "deepocsort":
        payload["delta_t"] = tracker_cfg.get("delta_t", 3)
        payload["inertia"] = tracker_cfg.get("inertia", 0.2)
        payload["use_byte"] = tracker_cfg.get("use_byte", True)
        payload["alpha_fixed_emb"] = tracker_cfg.get("alpha_fixed_emb", 0.95)

    if backend == "tracktrack":
        payload["lost_match_thr"] = tracker_cfg.get("lost_match_thr", 0.0)
        payload["iou_weight"] = tracker_cfg.get("iou_weight", 0.5)
        payload["reid_weight"] = tracker_cfg.get("reid_weight", 0.5)
        payload["conf_weight"] = tracker_cfg.get("conf_weight", 0.1)
        payload["angle_weight"] = tracker_cfg.get("angle_weight", 0.05)
        payload["penalty_p"] = tracker_cfg.get("penalty_p", 0.2)
        payload["penalty_q"] = tracker_cfg.get("penalty_q", 0.4)
        payload["reduce_step"] = tracker_cfg.get("reduce_step", 0.05)
        payload["tai_thr"] = tracker_cfg.get("tai_thr", 0.55)
        payload["min_track_len"] = tracker_cfg.get("min_track_len", 3)

    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        yaml.safe_dump(payload, f, default_flow_style=False, sort_keys=False)

    # Zachowaj też legacy custom_botsort.yaml dla kompatybilności wstecznej
    if backend == "botsort":
        with open(BOTSORT_RUNTIME_PATH, "w", encoding="utf-8") as f:
            yaml.safe_dump(payload, f, default_flow_style=False, sort_keys=False)

    return str(out)


def write_botsort_yaml(botsort_cfg: dict[str, Any], output_path: str | os.PathLike | None = None) -> str:
    """Alias kompatybilności — domyślnie generuje BoT-SORT."""
    return write_tracker_yaml(botsort_cfg, tracker_name="BoT-SORT", output_path=output_path)


def ensure_homography(path: str | os.PathLike) -> str:
    """
    Zwraca ścieżkę do macierzy H. Jeśli plik nie istnieje, zapisuje macierz jednostkową
    (placeholder — wystarczy do ewaluacji bboxów MOT; BEV nie będzie wtedy geometrycznie poprawne).
    """
    path = Path(path)
    if path.exists():
        return str(path)

    path.parent.mkdir(parents=True, exist_ok=True)
    np.save(path, np.eye(3, dtype=np.float64))
    print(f"UWAGA: brak homografii {path} — utworzono macierz jednostkową (placeholder).")
    return str(path)


def list_sequences(config: dict[str, Any], dataset: str | None = None) -> list[str]:
    """Lista sekwencji zdefiniowanych dla aktywnego (lub wskazanego) zbioru."""
    experiment_cfg = config["experiment"]
    dataset_name = (dataset or experiment_cfg["dataset"]).lower()
    dataset_cfg = config["datasets"][dataset_name]
    return list(dataset_cfg.get("sequences", []))


def resolve_run(
    config: dict[str, Any],
    dataset: str | None = None,
    seq_name: str | None = None,
) -> dict[str, Any]:
    """
    Buduje opis jednego przebiegu eksperymentu:
    - seq_name, dataset, tracker_name, bev_size
    - cameras: lista {id, source, source_type, matrix, name?}
    - results_stem: bazowa nazwa pliku wynikowego
    """
    experiment_cfg = config["experiment"]
    dataset_name = (dataset or experiment_cfg["dataset"]).lower()
    if dataset_name not in config["datasets"]:
        raise KeyError(f"Nieznany zbiór '{dataset_name}'. Dostępne: {list(config['datasets'])}")

    dataset_cfg = config["datasets"][dataset_name]
    sequences = list(dataset_cfg.get("sequences", []))
    active_seq = seq_name or experiment_cfg.get("seq_name") or (sequences[0] if sequences else dataset_name)

    cameras: list[dict[str, Any]] = []
    source_type = dataset_cfg.get("source_type", "opencv")
    homo_dir = dataset_cfg.get("homography_dir", "data/calibration")

    if dataset_name == "mot17":
        split = dataset_cfg.get("split", "train")
        pattern = dataset_cfg["video_pattern"]
        video_source = pattern.format(split=split, seq_name=active_seq)
        matrix_path = ensure_homography(Path(homo_dir) / f"{active_seq}.npy")
        cameras.append({
            "id": 1,
            "name": active_seq,
            "source": video_source,
            "source_type": source_type,
            "matrix": matrix_path,
        })
        results_stem = active_seq

    elif dataset_name == "wildtrack":
        for cam in dataset_cfg["cameras"]:
            cam_name = cam.get("name", f"C{cam['id']}")
            matrix_path = ensure_homography(Path(homo_dir) / f"{cam_name}.npy")
            cameras.append({
                "id": cam["id"],
                "name": cam_name,
                "source": cam["source"],
                "source_type": source_type,
                "matrix": matrix_path,
            })
        results_stem = active_seq

    else:
        raise ValueError(f"Brak resolvera dla zbioru: {dataset_name}")

    return {
        "dataset": dataset_name,
        "seq_name": active_seq,
        "tracker_name": experiment_cfg["tracker_name"],
        "bev_size": experiment_cfg["bev_size"],
        "cameras": cameras,
        "results_stem": results_stem,
        "per_camera_results": dataset_name == "wildtrack",
    }


# Kompatybilność wsteczna dla starszych wywołań
def resolve_video_source(experiment_cfg: dict[str, Any]) -> str:
    """Buduje ścieżkę źródła wideo/sekwencji (legacy, głównie MOT17)."""
    if "video_pattern" in experiment_cfg and "seq_name" in experiment_cfg:
        return experiment_cfg["video_pattern"].format(
            seq_name=experiment_cfg["seq_name"],
            split=experiment_cfg.get("split", "train"),
        )
    raise KeyError("resolve_video_source wymaga video_pattern i seq_name — użyj resolve_run().")
