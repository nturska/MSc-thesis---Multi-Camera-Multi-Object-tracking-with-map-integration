import numpy as np
from scipy.spatial.distance import cosine
from scipy.optimize import linear_sum_assignment

class GlobalTracker:
    def __init__(self, max_visual_cost=0.4, max_spatial_dist=100.0, max_missed_frames=300):
        # Baza danych globalnych tożsamości
        # Format: { global_id: {"reid_vector": [...], "last_bev": (x, y), "missed_frames": 0} }
        self.global_tracks = {}
        self.next_global_id = 1
        
        # Progi odcięcia (powyżej nich algorytm uzna, że to dwie różne osoby)
        self.max_visual_cost = max_visual_cost
        self.max_spatial_dist = max_spatial_dist
        self.max_missed_frames = max_missed_frames

    def update(self, current_detections):
        """
        Główna funkcja asocjacji. 
        current_detections to lista słowników: [{"reid": wektor, "bev": (x,y), "camera_id": 1}]
        """
        # Jeśli nie mamy jeszcze nikogo w bazie, po prostu dodajemy wszystkich jako nowych
        if not self.global_tracks:
            return self._register_new_tracks(current_detections)

        # 1. PRZYGOTOWANIE MACIERZY KOSZTÓW
        num_existing = len(self.global_tracks)
        num_new = len(current_detections)
        
        cost_matrix = np.zeros((num_existing, num_new))
        existing_ids = list(self.global_tracks.keys())

        for i, global_id in enumerate(existing_ids):
            track_data = self.global_tracks[global_id]
            for j, det in enumerate(current_detections):
                
                # A) Koszt wizualny (Odległość cosinusowa: 0 to identyczne, 1 to zupełnie inne)
                vis_cost = cosine(track_data["reid_vector"].flatten(), det["reid"].flatten())
                
                # B) Koszt przestrzenny (Odległość euklidesowa na mapie BEV)
                dist_x = track_data["last_bev"][0] - det["bev"][0]
                dist_y = track_data["last_bev"][1] - det["bev"][1]
                spatial_dist = np.sqrt(dist_x**2 + dist_y**2)

                # C) Fuzja kosztów i odrzucanie par niemożliwych
                if vis_cost > self.max_visual_cost or spatial_dist > self.max_spatial_dist:
                    cost_matrix[i, j] = 999.0 # Koszt zaporowy (odrzucenie fizycznie/wizualnie niemożliwych)
                else:
                    # Finalny koszt: miks wyglądu i geometrii 
                    # (W pełnym wdrożeniu można tu dodać wagi: alpha * vis_cost + beta * spatial_dist)
                    cost_matrix[i, j] = vis_cost 

        # 2. ALGORYTM WĘGIERSKI (Dopasowanie w grafie dwudzielnym o min. koszcie)
        row_indices, col_indices = linear_sum_assignment(cost_matrix)

        assigned_detections = set()
        matched_results = []

        # 3. ZATWIERDZENIE DOPASOWAŃ
        for r, c in zip(row_indices, col_indices):
            if cost_matrix[r, c] < 999.0: # Jeśli dopasowanie jest dozwolone
                global_id = existing_ids[r]
                det = current_detections[c]
                
                # Aktualizacja bazy danych (odświeżamy wygląd i pozycję na mapie)
                # Opcjonalnie: można tu zastosować EMA (Exponential Moving Average) do płynnej aktualizacji wektora
                self.global_tracks[global_id]["reid_vector"] = det["reid"]
                self.global_tracks[global_id]["last_bev"] = det["bev"]
                self.global_tracks[global_id]["missed_frames"] = 0
                
                det["global_id"] = global_id
                matched_results.append(det)
                assigned_detections.add(c)

        # 4. REJESTRACJA NOWYCH OSÓB (tych, których nie dopasowano do bazy)
        unmatched_detections = [current_detections[i] for i in range(num_new) if i not in assigned_detections]
        matched_results.extend(self._register_new_tracks(unmatched_detections))

        # 5. ZARZĄDZANIE "MARTWYMI" DUSZAMI (Czyszczenie bazy)
        for global_id in existing_ids:
            if global_id not in [res["global_id"] for res in matched_results]:
                self.global_tracks[global_id]["missed_frames"] += 1
                
            # Jeśli obiektu nie ma przez max_missed_frames klatek, usuwamy z RAM
            if self.global_tracks.get(global_id, {}).get("missed_frames", 0) > self.max_missed_frames:
                del self.global_tracks[global_id]

        return matched_results

    def _register_new_tracks(self, detections):
        """Dodaje całkowicie nowe tożsamości do systemu"""
        results = []
        for det in detections:
            new_id = self.next_global_id
            self.global_tracks[new_id] = {
                "reid_vector": det["reid"],
                "last_bev": det["bev"],
                "missed_frames": 0
            }
            det["global_id"] = new_id
            results.append(det)
            self.next_global_id += 1
        return results