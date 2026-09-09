"""
MCMT pipeline — Decoupled Detection and Tracking.

1) Ultralytics YOLO / RT-DETR: wyłącznie detekcja (`predict`, niski conf).
2) SCTTrackerAdapter: ramki detekcji → lokalny tracker SCT (`.update()`).
3) GlobalTracker + zapis MOT Challenge (bez zmian formatu).
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import cv2
import numpy as np
import torch
import multiprocessing as mp
from queue import Empty
from ultralytics import YOLO, RTDETR

from src.config import load_config, resolve_run, list_sequences
from src.reid_extractor import ReIDExtractor
from src.global_tracker import GlobalTracker
from trackers.adapter import SCTTrackerAdapter


# ---------------------------------------------------------------------------
# Uniwersalny adapter SCT — klasa używana w potoku (definicja: trackers.adapter)
# ---------------------------------------------------------------------------
# SCTTrackerAdapter:
#   - ByteTrack / BoT-SORT  → ultralytics.trackers.byte_tracker.BYTETracker /
#                             ultralytics.trackers.bot_sort.BOTSORT
#   - Deep_OC-SORT / TrackTrack → ultralytics.trackers.*
#   - Hybrid-SORT / BoostTrack++ → system/trackers/{hybridsort,boosttrackpp}


def load_detector(model_name: str, device):
    """Ładuje YOLO lub RT-DETR na podstawie nazwy wag."""
    name = model_name.lower()
    aliases = {
        "rt-detr-l.pt": "rtdetr-l.pt",
        "yolov11m.pt": "yolo11m.pt",
        "yolov26m.pt": "yolo26m.pt",  # Ultralytics: bez „v” w nazwie pliku
        "yolo26m.pt": "yolo26m.pt",
    }
    resolved = aliases.get(name, model_name)

    if "rtdetr" in resolved.lower() or "rt-detr" in resolved.lower():
        model = RTDETR(resolved)
    else:
        model = YOLO(resolved)
    model.to(device)
    return model


def iter_frames(source: str, source_type: str = "opencv", max_frames: int | None = None):
    """Iterator (frame_idx, frame) — OpenCV pattern/wideo albo katalog obrazów (WILDTRACK)."""
    if source_type == "image_dir":
        if not os.path.isdir(source):
            raise FileNotFoundError(f"Brak katalogu obrazów: {source}")
        files = sorted(
            f for f in os.listdir(source)
            if f.lower().endswith((".png", ".jpg", ".jpeg", ".bmp"))
        )
        if not files:
            raise FileNotFoundError(f"Brak obrazów w: {source}")
        for frame_idx, name in enumerate(files, start=1):
            if max_frames is not None and frame_idx > max_frames:
                break
            frame = cv2.imread(os.path.join(source, name))
            if frame is None:
                continue
            yield frame_idx, frame
        return

    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        raise FileNotFoundError(f"Nie można otworzyć źródła wideo: {source}")
    frame_idx = 1
    while True:
        if max_frames is not None and frame_idx > max_frames:
            break
        ret, frame = cap.read()
        if not ret:
            break
        yield frame_idx, frame
        frame_idx += 1
    cap.release()


def _boxes_to_numpy(result) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Wyciąga xyxy / conf / cls z wyniku Ultralytics predict (puste tablice, gdy brak detekcji)."""
    boxes = result.boxes
    if boxes is None or len(boxes) == 0:
        return (
            np.zeros((0, 4), dtype=np.float32),
            np.zeros((0,), dtype=np.float32),
            np.zeros((0,), dtype=np.float32),
        )
    xyxy = boxes.xyxy.cpu().numpy().astype(np.float32)
    conf = boxes.conf.cpu().numpy().astype(np.float32)
    cls = boxes.cls.cpu().numpy().astype(np.float32)
    return xyxy, conf, cls


# ==========================================
# 1. PROCES POJEDYNCZEJ KAMERY
# ==========================================
def camera_worker(camera_id, source, matrix_path, output_queue, worker_cfg, source_type="opencv", max_frames=None):
    os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

    detection_cfg = worker_cfg["detection"]
    reid_cfg = worker_cfg["reid"]
    experiment_config = worker_cfg["config"]
    sct_name = worker_cfg["sct_name"]
    video_name = worker_cfg.get("video_name")

    model = load_detector(detection_cfg["model"], device)
    reid_extractor = ReIDExtractor(
        model_name=reid_cfg["model_name"],
        min_crop_size=reid_cfg["min_crop_size"],
    )
    # Uniwersalny adapter SCT — parametry z experiment.yaml, update() klatka po klatce
    sct_tracker = SCTTrackerAdapter(
        experiment_config,
        tracker_name=sct_name,
        device=device,
        video_name=video_name,
    )

    if not os.path.exists(matrix_path):
        print(f"[Kamera {camera_id}] BŁĄD: Brak macierzy {matrix_path}")
        return
    H = np.load(matrix_path)

    try:
        for frame_idx, frame in iter_frames(source, source_type=source_type, max_frames=max_frames):
            # 1) Wyłącznie detekcja (niski próg conf — druga faza asocjacji ByteTrack/BoT-SORT)
            results = model.predict(
                frame,
                device=device,
                verbose=False,
                classes=detection_cfg["classes"],
                conf=detection_cfg["conf"],
                iou=detection_cfg["iou"],
            )
            xyxy, conf, cls = _boxes_to_numpy(results[0])

            # 2) Lokalny tracker SCT
            tracks = sct_tracker.update(xyxy, conf, cls, frame)

            frame_detections = []
            for trk in tracks:
                x1, y1, x2, y2 = trk["box"]
                x1_i, y1_i, x2_i, y2_i = map(int, (x1, y1, x2, y2))

                u, v = int((x1 + x2) / 2), int(y2)
                point_camera = np.array([[[u, v]]], dtype=np.float32)
                point_bev = cv2.perspectiveTransform(point_camera, H)
                bx, by = int(point_bev[0][0][0]), int(point_bev[0][0][1])

                reid_vector = reid_extractor.extract_features(frame, np.array([x1, y1, x2, y2], dtype=np.float32))

                if reid_vector is not None:
                    frame_detections.append({
                        "bev": (bx, by),
                        "reid": reid_vector,
                        "box": (x1_i, y1_i, x2_i, y2_i),
                        "camera_id": camera_id,
                        "local_track_id": trk["track_id"],
                    })

            if frame_detections:
                output_queue.put({
                    "camera_id": camera_id,
                    "frame_idx": frame_idx,
                    "detections": frame_detections,
                })
    except FileNotFoundError as exc:
        print(f"[Kamera {camera_id}] BŁĄD: {exc}")
        return
    except Exception as exc:
        print(f"[Kamera {camera_id}] BŁĄD trackera/detekcji: {exc}")
        raise

    print(f"[Kamera {camera_id}] Zakończyła przetwarzanie.")


# ==========================================
# 2. JEDEN PRZEBIEG (jedna sekwencja / jeden dataset run)
# ==========================================
def run_sequence(config, run_info, start_method_set=True, max_frames=None):
    detection_cfg = config["detection"]
    global_cfg = config["global_tracker"]
    sct_name = config.get("sct_tracker", {}).get("name", "BoT-SORT")

    if not start_method_set:
        try:
            mp.set_start_method("spawn")
        except RuntimeError:
            pass

    data_queue = mp.Queue()
    global_tracker = GlobalTracker(
        max_visual_cost=global_cfg["max_visual_cost"],
        max_spatial_dist=global_cfg["max_spatial_dist"],
        max_missed_frames=global_cfg["max_missed_frames"],
    )

    worker_cfg = {
        "detection": detection_cfg,
        "reid": config["reid"],
        "config": config,
        "sct_name": sct_name,
        "video_name": run_info["seq_name"],
    }

    if max_frames is not None:
        print(f"SMOKE / limit: przetwarzam tylko pierwsze {max_frames} klatek.")

    print(f"SCT tracker: {sct_name} | detektor: {detection_cfg['model']} | conf={detection_cfg['conf']}")

    processes = []
    for cam in run_info["cameras"]:
        p = mp.Process(
            target=camera_worker,
            args=(
                cam["id"],
                cam["source"],
                cam["matrix"],
                data_queue,
                worker_cfg,
                cam.get("source_type", "opencv"),
                max_frames,
            ),
        )
        p.start()
        processes.append(p)

    tracker_name = run_info["tracker_name"]
    output_dir = Path("results") / tracker_name / "data"
    output_dir.mkdir(parents=True, exist_ok=True)

    results_files = {}
    if run_info["per_camera_results"]:
        for cam in run_info["cameras"]:
            path = output_dir / f"{run_info['results_stem']}_{cam['name']}.txt"
            results_files[cam["id"]] = open(path, "w", encoding="utf-8")
            print(f"Wyniki kamery {cam['name']}: {path}")
    else:
        path = output_dir / f"{run_info['results_stem']}.txt"
        results_files[None] = open(path, "w", encoding="utf-8")
        print(f"Rozpoczęto śledzenie [{run_info['dataset']}/{run_info['seq_name']}]. Wyniki: {path}")

    try:
        while True:
            try:
                packet = data_queue.get(timeout=1.0)
            except Empty:
                if not any(p.is_alive() for p in processes):
                    break
                continue

            frame_idx = packet["frame_idx"]
            camera_id = packet["camera_id"]
            detections = packet["detections"]

            matched_targets = global_tracker.update(detections)

            out_file = results_files.get(camera_id) or results_files.get(None)
            for target in matched_targets:
                x1, y1, x2, y2 = target["box"]
                g_id = target["global_id"]
                width = x2 - x1
                height = y2 - y1
                # Format MOT Challenge (bez zmian)
                out_file.write(f"{frame_idx},{g_id},{x1},{y1},{width},{height},1,-1,-1,-1\n")

    except KeyboardInterrupt:
        print("\nPrzerwanie przez użytkownika.")

    finally:
        for p in processes:
            p.terminate()
            p.join()
        for f in results_files.values():
            f.close()
        print(f"Zakończono: {run_info['dataset']}/{run_info['seq_name']}.")


def main(config_path=None, dataset=None, seq_name=None, run_all=None, max_frames=None):
    config = load_config(config_path)
    experiment_cfg = config["experiment"]

    if dataset is not None:
        experiment_cfg["dataset"] = dataset
    if seq_name is not None:
        experiment_cfg["seq_name"] = seq_name
    if run_all is None:
        run_all = bool(experiment_cfg.get("run_all_sequences", False))

    try:
        mp.set_start_method("spawn")
    except RuntimeError:
        pass

    dataset_name = experiment_cfg["dataset"].lower()
    if run_all:
        sequences = list_sequences(config, dataset_name)
    else:
        run = resolve_run(config, dataset=dataset_name, seq_name=experiment_cfg.get("seq_name"))
        sequences = [run["seq_name"]]

    sct_name = config.get("sct_tracker", {}).get("name", "BoT-SORT")
    print(f"Dataset: {dataset_name} | sekwencje: {sequences} | SCT: {sct_name}")

    for seq in sequences:
        run_info = resolve_run(config, dataset=dataset_name, seq_name=seq)
        run_sequence(
            config,
            run_info,
            start_method_set=True,
            max_frames=max_frames,
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="MCMT pipeline — MOT17 / WILDTRACK (decoupled det+track)")
    parser.add_argument("--config", default=None, help="Ścieżka do experiment.yaml")
    parser.add_argument("--dataset", choices=["mot17", "wildtrack"], default=None)
    parser.add_argument("--seq", dest="seq_name", default=None, help="Nazwa sekwencji (np. MOT17-02-FRCNN)")
    parser.add_argument("--all", dest="run_all", action="store_true", help="Wszystkie sekwencje zbioru")
    parser.add_argument(
        "--max-frames",
        type=int,
        default=None,
        help="Limit klatek (np. 30 do smoke testu); domyślnie cała sekwencja",
    )
    args = parser.parse_args()
    main(
        config_path=args.config,
        dataset=args.dataset,
        seq_name=args.seq_name,
        run_all=args.run_all if args.run_all else None,
        max_frames=args.max_frames,
    )
