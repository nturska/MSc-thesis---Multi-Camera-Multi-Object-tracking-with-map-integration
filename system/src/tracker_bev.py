import cv2
import os
import numpy as np
import torch
from ultralytics import YOLO

def main():
    # 1. Ścieżki do plików (dostosuj, jeśli pliki są w innych folderach)
    video_path = "../system/data/videos/grzybowska.mp4"
    matrix_path = "../system/data/calibration/homography.npy"
    
    # Sprawdzenie, czy macierz homografii istnieje
    if not os.path.exists(matrix_path):
        print(f"Błąd: Nie znaleziono pliku macierzy w {matrix_path}. Uruchom najpierw calibration.py!")
        return

    # Wczytanie macierzy homografii H
    H = np.load(matrix_path)

    # 2. Konfiguracja sprzętu (Apple MPS)
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Uruchamiam system na urządzeniu: {device}")

    # Ładowanie modelu YOLOv8 Medium
    model = YOLO("../system/models/yolov8m.pt")
    model.to(device)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("Błąd: Nie można otworzyć wideo.")
        return

    # Rozmiar naszej docelowej mapy BEV (zgodny z plikiem kalibracyjnym)
    BEV_WIDTH = 500
    BEV_HEIGHT = 500

    print("Uruchamianie pętli przetwarzania... Naciśnij 'q', aby przerwać.")

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # A) Detekcja i śledzenie lokalne (SCT) z użyciem naszego trackera
        # Używamy wbudowanego BoT-SORT lub domyślnego trackera
        results = model.track(frame, tracker="../system/configs/custom_botsort.yaml", device=device, persist=True, verbose=False, classes=[0], conf=0.2)
        
        # Tworzymy puste tło dla naszej mapy BEV (czarny kwadrat 500x500)
        bev_map = np.zeros((BEV_HEIGHT, BEV_WIDTH, 3), dtype=np.uint8)

        # Rysujemy statyczny plan/siatkę na mapie BEV (opcjonalnie, np. linie siatki co 100px)
        for i in range(0, BEV_WIDTH, 50):
            cv2.line(bev_map, (i, 0), (i, BEV_HEIGHT), (50, 50, 50), 1)
            cv2.line(bev_map, (0, i), (BEV_WIDTH, i), (50, 50, 50), 1)

        # B) Przetwarzanie wykrytych obiektów
        if results[0].boxes.id is not None:
            boxes = results[0].boxes.xyxy.cpu().numpy()  # Współrzędne ramek [x1, y1, x2, y2]
            track_ids = results[0].boxes.id.int().cpu().numpy()  # ID obiektów

            for box, track_id in zip(boxes, track_ids):
                x1, y1, x2, y2 = box
                
                # Wyznaczamy punkt styku z podłożem: środek dolnej krawędzi ramki
                # [u, v] w układzie pikselowym obrazu kamery
                u = int((x1 + x2) / 2)
                v = int(y2)

                # Rysujemy punkt na oryginalnym wideo (stopa pieszego)
                cv2.circle(frame, (u, v), 5, (0, 0, 255), -1)
                cv2.putText(frame, f"ID:{track_id}", (int(x1), int(y1) - 10), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

                # C) RZUTOWANIE PUNKTOWE ZA POMOCĄ HOMOGRAFII (IPM)
                # Tworzymy wektor jednorodny punktu [u, v, 1]
                point_camera = np.array([[[u, v]]], dtype=np.float32)
                
                # Przekształcenie macierzowe: punkt na mapie BEV [X, Y, Z] / lambda
                point_bev = cv2.perspectiveTransform(point_camera, H)
                
                bx, by = int(point_bev[0][0][0]), int(point_bev[0][0][1])

                # D) Rysowanie pozycji na mapie BEV, jeśli mieści się w granicach 500x500
                if 0 <= bx < BEV_WIDTH and 0 <= by < BEV_HEIGHT:
                    cv2.circle(bev_map, (bx, by), 6, (0, 255, 255), -1)
                    cv2.putText(bev_map, f"{track_id}", (bx + 8, by + 4), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)

        # Wyświetlanie obu okien: obraz z kamery oraz rzut BEV z naniesionymi pozycjami
        cv2.imshow("Widok z kamery (SCT)", frame)
        cv2.imshow("Widok z góry - BEV (Map Intergation)", bev_map)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()