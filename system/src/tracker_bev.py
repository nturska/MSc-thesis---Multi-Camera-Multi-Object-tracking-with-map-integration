import cv2
import os
import sys
import numpy as np
import torch
from ultralytics import YOLO

# Umożliwia uruchomienie jako python src/tracker_bev.py z katalogu system/
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.config import load_config, write_botsort_yaml, resolve_run


def main(config_path=None):
    config = load_config(config_path)
    detection_cfg = config["detection"]
    experiment_cfg = config["experiment"]
    tracker_yaml = write_botsort_yaml(config["botsort"])

    # Domyślnie pierwsza kamera z aktywnego datasetu (MOT17 / WILDTRACK)
    run = resolve_run(config, dataset=experiment_cfg.get("dataset", "mot17"))
    cam = run["cameras"][0]
    video_path = cam["source"]
    matrix_path = cam["matrix"]

    # Dostosowanie ścieżek, gdy skrypt startuje z system/src
    if cam.get("source_type") == "opencv":
        img_dir = video_path.split("%")[0]
        if not os.path.isdir(img_dir):
            alt_video = os.path.join("..", video_path)
            alt_dir = alt_video.split("%")[0]
            if os.path.isdir(alt_dir):
                video_path = alt_video
                matrix_path = os.path.join("..", matrix_path)

    if not os.path.exists(matrix_path):
        print(f"Błąd: Nie znaleziono macierzy {matrix_path}.")
        return

    H = np.load(matrix_path)

    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Uruchamiam system na urządzeniu: {device}")
    print(f"Źródło: {video_path}")

    model_path = detection_cfg["model"]
    if not os.path.exists(model_path):
        candidate = f"models/{os.path.basename(model_path)}"
        if os.path.exists(candidate):
            model_path = candidate
        else:
            candidate = f"../system/models/{os.path.basename(model_path)}"
            if os.path.exists(candidate):
                model_path = candidate

    model = YOLO(model_path)
    model.to(device)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("Błąd: Nie można otworzyć źródła wideo/sekwencji.")
        return

    BEV_WIDTH, BEV_HEIGHT = experiment_cfg["bev_size"]

    print("Uruchamianie pętli przetwarzania... Naciśnij 'q', aby przerwać.")

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        results = model.track(
            frame,
            tracker=tracker_yaml,
            device=device,
            persist=True,
            verbose=False,
            classes=detection_cfg["classes"],
            conf=detection_cfg["conf"],
            iou=detection_cfg["iou"],
        )

        bev_map = np.zeros((BEV_HEIGHT, BEV_WIDTH, 3), dtype=np.uint8)

        for i in range(0, BEV_WIDTH, 50):
            cv2.line(bev_map, (i, 0), (i, BEV_HEIGHT), (50, 50, 50), 1)
            cv2.line(bev_map, (0, i), (BEV_WIDTH, i), (50, 50, 50), 1)

        if results[0].boxes.id is not None:
            boxes = results[0].boxes.xyxy.cpu().numpy()
            track_ids = results[0].boxes.id.int().cpu().numpy()

            for box, track_id in zip(boxes, track_ids):
                x1, y1, x2, y2 = box

                u = int((x1 + x2) / 2)
                v = int(y2)

                cv2.circle(frame, (u, v), 5, (0, 0, 255), -1)
                cv2.putText(frame, f"ID:{track_id}", (int(x1), int(y1) - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

                point_camera = np.array([[[u, v]]], dtype=np.float32)
                point_bev = cv2.perspectiveTransform(point_camera, H)

                bx, by = int(point_bev[0][0][0]), int(point_bev[0][0][1])

                if 0 <= bx < BEV_WIDTH and 0 <= by < BEV_HEIGHT:
                    cv2.circle(bev_map, (bx, by), 6, (0, 255, 255), -1)
                    cv2.putText(bev_map, f"{track_id}", (bx + 8, by + 4),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)

        cv2.imshow("Widok z kamery (SCT)", frame)
        cv2.imshow("Widok z góry - BEV (Map Intergation)", bev_map)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
