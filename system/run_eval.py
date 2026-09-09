import argparse
import os
import sys
import subprocess

from src.config import load_config


def run_evaluation(config_path=None, dataset=None, seq_name=None, tracker_name=None):
    config = load_config(config_path)
    experiment_cfg = config["experiment"]
    dataset_name = (dataset or experiment_cfg.get("dataset", "mot17")).lower()
    tracker_name = tracker_name or experiment_cfg.get("tracker_name", "MGR_Tracker")
    seq_name = seq_name or experiment_cfg.get("seq_name")

    print("Inicjalizacja ewaluatora wizyjnego (TrackEval)...")

    # Wrapper z patch NumPy (np.float) — nie edytujemy sklonowanego TrackEval
    trackeval_script = os.path.join(os.path.dirname(os.path.abspath(__file__)), "run_mot_challenge_compat.py")
    trackers_folder = "results"

    if dataset_name == "mot17":
        split = config["datasets"]["mot17"].get("split", "train")
        gt_folder = os.path.join("data", "MOT17", split)
        benchmark = "MOT17"
        split_to_eval = split
    elif dataset_name == "wildtrack":
        gt_folder = os.path.join("data", "WILDTRACK_MOT", "train")
        benchmark = "MOT17"
        split_to_eval = "train"
        print(
            "UWAGA: TrackEval dla WILDTRACK wymaga GT w formacie MOT "
            f"(katalog {gt_folder}). Wyniki trackerów: results/{tracker_name}/data/WILDTRACK_*.txt"
        )
    else:
        raise ValueError(f"Nieobsługiwany dataset do ewaluacji: {dataset_name}")

    command = [
        sys.executable, trackeval_script,
        "--BENCHMARK", benchmark,
        "--SPLIT_TO_EVAL", split_to_eval,
        "--TRACKERS_TO_EVAL", tracker_name,
        "--GT_FOLDER", gt_folder,
        "--TRACKERS_FOLDER", trackers_folder,
        "--METRICS", "HOTA", "CLEAR", "Identity",
        "--USE_PARALLEL", "False",
        "--NUM_PARALLEL_CORES", "1",
        # GT/results leżą bezpośrednio w GT_FOLDER / TRACKERS_FOLDER
        # (bez pośredniego katalogu MOT17-train).
        "--SKIP_SPLIT_FOL", "True",
    ]
    if seq_name:
        command.extend(["--SEQ_INFO", seq_name])

    print(f"Dataset: {dataset_name} | GT: {gt_folder} | tracker: {tracker_name} | seq: {seq_name}")
    subprocess.run(command, check=True)
    print("\nSukces! Zestawienie wyników wygenerowane.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="TrackEval — MOT17 / WILDTRACK")
    parser.add_argument("--config", default=None)
    parser.add_argument("--dataset", choices=["mot17", "wildtrack"], default=None)
    parser.add_argument("--seq", dest="seq_name", default=None)
    parser.add_argument("--tracker-name", default=None)
    args = parser.parse_args()
    try:
        run_evaluation(
            config_path=args.config,
            dataset=args.dataset,
            seq_name=args.seq_name,
            tracker_name=args.tracker_name,
        )
    except subprocess.CalledProcessError as e:
        print(f"\nBŁĄD: Ewaluacja zakończyła się niepowodzeniem. Kod: {e.returncode}")
        sys.exit(e.returncode or 1)
    except Exception as e:
        print(f"\nBŁĄD: {e}")
        sys.exit(1)
