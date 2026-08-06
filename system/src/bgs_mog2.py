import cv2

def run_background_subtraction(video_path):
    print(f"Uruchamiam wyodrębnianie tła dla pliku: {video_path}")
    
    # 1. Inicjalizacja algorytmu MOG2
    # history: liczba klatek brana pod uwagę do modelowania tła
    # varThreshold: próg odległości Mahalanobisa (wpływa na czułość detekcji)
    # detectShadows: zaznacza cienie na szaro (przydatne do ich późniejszego odfiltrowania)
    backSub = cv2.createBackgroundSubtractorMOG2(history=500, varThreshold=16, detectShadows=True)
    
    # Otwarcie strumienia wideo
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        print("Błąd: Nie można otworzyć pliku wideo. Sprawdź ścieżkę.")
        return

    # Przygotowanie jądra (kernel) do późniejszego czyszczenia szumów
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Koniec nagrania wideo.")
            break
            
        # 2. Zastosowanie algorytmu MOG2 do aktualnej klatki
        # Zwraca maskę, gdzie 255 to ruch, 127 to cień, a 0 to tło
        fgMask = backSub.apply(frame)
        
        # 3. Filtracja szumów (Operacje morfologiczne)
        # Usuwamy cienie (wartość 127) zostawiając tylko pewny ruch (wartość 255)
        _, fgMask = cv2.threshold(fgMask, 254, 255, cv2.THRESH_BINARY)
        
        # Operacja otwarcia usuwa drobne, pojedyncze białe piksele (np. szum matrycy, drżące liście)
        fgMask_cleaned = cv2.morphologyEx(fgMask, cv2.MORPH_OPEN, kernel)
        
        # 4. Wyświetlanie wyników w dwóch osobnych oknach
        cv2.imshow('Oryginalne Wideo', frame)
        cv2.imshow('Maska Ruchu (MOG2) - Wyczyszczona', fgMask_cleaned)
        
        # Naciśnij 'q', aby wyjść
        if cv2.waitKey(30) & 0xFF == ord('q'):
            break

    # Zwolnienie zasobów
    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    # Zmień ścieżkę na właściwą, jeśli Twój plik jest w folderze data/
    run_background_subtraction("data/grzybowska.mp4")