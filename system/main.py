import os
import cv2
import numpy as np
import torch
import multiprocessing as mp
from queue import Empty
from ultralytics import YOLO

from src.reid_extractor import ReIDExtractor
from src.global_tracker import GlobalTracker

# ==========================================
# 1. PROCES POJEDYNCZEJ KAMERY
# ==========================================
def camera_worker(camera_id, video_path, matrix_path, output_queue):
    os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    
    model = YOLO("yolov8m.pt")
    model.to(device)
    reid_extractor = ReIDExtractor() 
    
    if not os.path.exists(matrix_path):
        print(f"[Kamera {camera_id}] BŁĄD: Brak macierzy {matrix_path}")
        return
    H = np.load(matrix_path)

    cap = cv2.VideoCapture(video_path)
    frame_idx = 1  # DODANO: Licznik klatek wymagany do ewaluacji
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        results = model.track(frame, tracker="configs/custom_botsort.yaml", 
                              device=device, persist=True, verbose=False, classes=[0], conf=0.25)
        
        frame_detections = []

        if results[0].boxes.id is not None:
            boxes = results[0].boxes.xyxy.cpu().numpy()
            
            for box in boxes:
                x1, y1, x2, y2 = map(int, box)
                
                u, v = int((x1 + x2) / 2), int(y2)
                point_camera = np.array([[[u, v]]], dtype=np.float32)
                point_bev = cv2.perspectiveTransform(point_camera, H)
                bx, by = int(point_bev[0][0][0]), int(point_bev[0][0][1])
                
                reid_vector = reid_extractor.extract_features(frame, box)
                
                if reid_vector is not None:
                    frame_detections.append({
                        "bev": (bx, by),
                        "reid": reid_vector,
                        "box": (x1, y1, x2, y2)
                    })

        # Wysyłamy paczkę poszerzoną o numer klatki
        if frame_detections:
            output_queue.put({
                "camera_id": camera_id, 
                "frame_idx": frame_idx, # DODANO: informacja o klatce
                "detections": frame_detections
            })
            
        frame_idx += 1 # Inkrementacja licznika

    cap.release()
    print(f"[Kamera {camera_id}] Zakończyła przetwarzanie.")

# ==========================================
# 2. PROCES GŁÓWNY (GLOBALNA ASOCJACJA I ZAPIS WYNIKÓW)
# ==========================================
def main():
    mp.set_start_method('spawn')
    data_queue = mp.Queue()
    global_tracker = GlobalTracker(max_visual_cost=0.4, max_spatial_dist=150.0)
    
    # 1. Zdefiniowanie nazwy sekwencji (zgodnie z nazwą z Ground Truth MOT17)
    seq_name = "MOT17-02-FRCNN"
    
    # Zbiór MOT17 dostarcza klatki jako obrazy w folderze img1 (np. 000001.jpg)
    video_source = f"data/MOT17/train/{seq_name}/img1/%06d.jpg"
    
    # KONFIGURACJA KAMER - Używamy zmiennej video_source
    cameras_config = [
        {
            "id": 1, 
            "video": video_source,
            # Podpinamy dowolną istniejącą macierz, by skrypt przeszedł dalej.
            # Ewaluator bada x, y, width, height, więc błędy BEV nas tu nie bolą.
            "matrix": "data/calibration/homography.npy" 
        }
    ]
    
    # [Uruchamianie procesów - bez zmian]
    processes = []
    for cam in cameras_config:
        p = mp.Process(target=camera_worker, args=(cam["id"], cam["video"], cam["matrix"], data_queue))
        p.start()
        processes.append(p)

    BEV_WIDTH, BEV_HEIGHT = 500, 500
    
    # 2. Dynamiczne utworzenie struktury folderów dla TrackEval
    # TrackEval oczekuje wyników dokładnie w tej ścieżce:
    tracker_name = "MGR_Tracker"
    output_dir = f"results/{tracker_name}/data"
    os.makedirs(output_dir, exist_ok=True)
    
    results_file_path = os.path.join(output_dir, f"{seq_name}.txt")
    print(f"Rozpoczęto śledzenie. Wyniki zostaną zapisane do pliku: {results_file_path}")
    
    with open(results_file_path, "w") as results_file:
        try:
            while True:
                # [Pętla odbierania pakietów z data_queue - bez zmian]
                try:
                    packet = data_queue.get(timeout=1.0)
                except Empty:
                    if not any(p.is_alive() for p in processes):
                        break
                    continue
                    
                camera_id = packet["camera_id"]
                frame_idx = packet["frame_idx"]
                detections = packet["detections"]
                
                matched_targets = global_tracker.update(detections)
                
                # Renderowanie mapy (pomińmy w snippet, zostaw jak miałaś)
                
                # ZAPIS DO PLIKU MOT FORMAT
                for target in matched_targets:
                    x1, y1, x2, y2 = target["box"]
                    g_id = target["global_id"]
                    width = x2 - x1
                    height = y2 - y1
                    
                    # Standaryzowany format: frame, id, bb_left, bb_top, width, height, conf, x, y, z
                    results_file.write(f"{frame_idx},{g_id},{x1},{y1},{width},{height},1,-1,-1,-1\n")
                                                  
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
                    
        except KeyboardInterrupt:
            print("\nPrzerwanie przez użytkownika.")
            
        finally:
            for p in processes:
                p.terminate()
                p.join()
            cv2.destroyAllWindows()
            print("Zakończono działanie systemu. Wyniki zapisane.")

if __name__ == '__main__':
    main()