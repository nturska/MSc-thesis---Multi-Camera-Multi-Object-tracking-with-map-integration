import cv2
import numpy as np
import os

# Globalna lista na współrzędne kliknięte przez użytkownika
clicked_points = []

def mouse_callback(event, x, y, flags, param):
    """Funkcja wywoływana przy każdym kliknięciu myszą w oknie OpenCV."""
    global clicked_points
    if event == cv2.EVENT_LBUTTONDOWN:
        if len(clicked_points) < 4:
            clicked_points.append([x, y])
            print(f"Zaznaczono punkt {len(clicked_points)}: ({x}, {y})")

def run_calibration(video_path, output_matrix_path):
    global clicked_points
    
    # 1. Wczytanie pierwszej klatki wideo
    cap = cv2.VideoCapture(video_path)
    ret, frame = cap.read()
    cap.release()
    
    if not ret:
        print("Nie udało się wczytać wideo.")
        return

    # Klonujemy klatkę do rysowania
    clone = frame.copy()
    cv2.namedWindow("Kalibracja - Zaznacz 4 punkty na plaskim podlozu (Ziemia)")
    cv2.setMouseCallback("Kalibracja - Zaznacz 4 punkty na plaskim podlozu (Ziemia)", mouse_callback)

    print("--- INSTRUKCJA ---")
    print("Zaznacz dokładnie 4 punkty na płaskim podłożu (np. rogi przejścia dla pieszych, kwadrat z płyt chodnikowych).")
    print("Kolejność klikania: Lewy-Górny, Prawy-Górny, Prawy-Dolny, Lewy-Dolny.")
    print("Naciśnij 'r' aby zresetować punkty, lub 'q' aby anulować.")

    while True:
        display_frame = clone.copy()
        
        # Rysowanie kropek i linii między klikniętymi punktami
        for i, pt in enumerate(clicked_points):
            cv2.circle(display_frame, tuple(pt), 5, (0, 0, 255), -1)
            if i > 0:
                cv2.line(display_frame, tuple(clicked_points[i-1]), tuple(pt), (0, 255, 0), 2)
            if len(clicked_points) == 4:
                cv2.line(display_frame, tuple(clicked_points[3]), tuple(clicked_points[0]), (0, 255, 0), 2)

        cv2.imshow("Kalibracja - Zaznacz 4 punkty na plaskim podlozu (Ziemia)", display_frame)
        key = cv2.waitKey(1) & 0xFF

        if len(clicked_points) == 4:
            print("Zaznaczono 4 punkty. Wyznaczam macierz homografii...")
            break
        elif key == ord("r"):
            clicked_points = []
            print("Zresetowano punkty.")
        elif key == ord("q"):
            print("Anulowano kalibrację.")
            cv2.destroyAllWindows()
            return

    cv2.destroyAllWindows()

    # 2. Definiowanie punktów źródłowych (z perspektywy kamery)
    pts_src = np.array(clicked_points, dtype=np.float32)

    # 3. Definiowanie punktów docelowych na mapie z góry (BEV)
    # Zakładamy, że zaznaczony obszar to w rzeczywistości prostokąt.
    # Ustawiamy sztuczną rozdzielczość "widoku z lotu ptaka", np. 500x500 pikseli.
    BEV_WIDTH = 500
    BEV_HEIGHT = 500
    pts_dst = np.array([
        [0, 0],                           # Lewy-Górny
        [BEV_WIDTH, 0],                   # Prawy-Górny
        [BEV_WIDTH, BEV_HEIGHT],          # Prawy-Dolny
        [0, BEV_HEIGHT]                   # Lewy-Dolny
    ], dtype=np.float32)

    # 4. Obliczanie macierzy homografii H
    H, status = cv2.findHomography(pts_src, pts_dst)
    
    # 5. Zapisanie macierzy do pliku numpy (.npy)
    os.makedirs(os.path.dirname(output_matrix_path), exist_ok=True)
    np.save(output_matrix_path, H)
    print(f"Sukces! Macierz homografii zapisana w: {output_matrix_path}")

    # TEST: Rzutowanie obrazu na BEV
    warped_frame = cv2.warpPerspective(frame, H, (BEV_WIDTH, BEV_HEIGHT))
    cv2.imshow("TEST - Rzutowanie z lotu ptaka (BEV)", warped_frame)
    print("Naciśnij dowolny klawisz, aby zamknąć okno testowe...")
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == '__main__':
    # Upewnij się, że ścieżki są poprawne względem miejsca uruchamiania skryptu!
    VIDEO_PATH = "../system/data/videos/grzybowska.mp4"
    MATRIX_PATH = "../system/data/calibration/homography.npy"
    run_calibration(VIDEO_PATH, MATRIX_PATH)