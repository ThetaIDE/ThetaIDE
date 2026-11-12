#!/usr/bin/env python3
import os
import csv
import math
import argparse
import numpy as np
from collections import defaultdict

def sigmoid(x):
    return 1.0 / (1.0 + math.exp(-x))

def to_float(x, default=0.0):
    try:
        if x is None or x == "":
            return default
        return float(x)
    except Exception:
        return default

def load_csv(path):
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))

def write_csv(path, rows, fieldnames):
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow(r)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--volume-dir", default=".", help="Path to THEMES_src/volume")
    parser.add_argument("--semesters", default="S21,S22,S24,S25,F24")
    parser.add_argument("--early-frac", type=float, default=0.2)
    parser.add_argument("--early-col", default="pctCorrect")
    parser.add_argument("--weight-col", default="NLG_z")
    parser.add_argument("--output-prefix", default="ITS")
    args = parser.parse_args()

    volume_dir = args.volume_dir
    semesters = [s.strip() for s in args.semesters.split(",") if s.strip()]
    random_dir = os.path.join(volume_dir, "Semester_based_Random")

    all_trajs = []
    all_rows_by_vid = {}

    for sem in semesters:
        csv_path = os.path.join(volume_dir, sem, "prob_level_data_NLG_post.csv")
        if not os.path.exists(csv_path):
            raise FileNotFoundError(csv_path)

        rows = load_csv(csv_path)

        good_path = os.path.join(random_dir, f"{sem}_good_students.npy")
        bad_path = os.path.join(random_dir, f"{sem}_bad_students.npy")
        if not os.path.exists(good_path) or not os.path.exists(bad_path):
            raise FileNotFoundError(f"Missing good/bad npy for {sem}")

        good = set(np.load(good_path, allow_pickle=True).astype(int))
        bad = set(np.load(bad_path, allow_pickle=True).astype(int))

        by_user = defaultdict(list)
        for r in rows:
            uid = int(float(r["userID"]))
            by_user[uid].append(r)

        for uid, traj in by_user.items():
            if uid not in good and uid not in bad:
                continue

            traj = sorted(traj, key=lambda r: to_float(r["MinutesFromArrival"]))
            k = max(1, int(math.ceil(args.early_frac * len(traj))))
            early_score = np.mean([to_float(r[args.early_col]) for r in traj[:k]])

            final_good = uid in good
            nlg_post = to_float(traj[0].get("NLG_post", 0.0))
            nlg_z = to_float(traj[0].get(args.weight_col, 0.0))

            vid = f"{sem}_{uid}"

            all_trajs.append({
                "sem": sem,
                "uid": uid,
                "vid": vid,
                "early_score": early_score,
                "final_good": final_good,
                "nlg_post": nlg_post,
                "nlg_z": nlg_z,
                "length": len(traj),
            })

            new_traj = []
            for t, r in enumerate(traj):
                rr = dict(r)
                rr["Semester"] = sem
                rr["VisitIdentifier"] = vid
                rr["orig_userID"] = uid
                rr["Step"] = t
                rr["Action"] = int(float(rr["Action"]))
                rr["w_t_gp"] = sigmoid(to_float(rr.get(args.weight_col, 0.0)))
                rr["w_t_gp_norm"] = rr["w_t_gp"]
                new_traj.append(rr)

            all_rows_by_vid[vid] = new_traj

    early_scores = [x["early_score"] for x in all_trajs]
    early_threshold = float(np.median(early_scores))

    q00, q10, q01, q11 = [], [], [], []
    halide_weights = []
    actual_labels = []
    lstm_pred = []
    f_shock = []
    f_shock_prime = []

    selected_vids = sorted([x["vid"] for x in all_trajs])
    traj_by_vid = {x["vid"]: x for x in all_trajs}

    clean_rows = []

    for idx, vid in enumerate(selected_vids):
        info = traj_by_vid[vid]
        early_good = info["early_score"] >= early_threshold
        final_good = info["final_good"]

        if early_good and final_good:
            q00.append(idx)
        elif (not early_good) and final_good:
            q10.append(idx)
        elif early_good and (not final_good):
            q01.append(idx)
        else:
            q11.append(idx)

        weight = sigmoid(info["nlg_z"])
        halide_weights.append(weight)
        actual_labels.append(0 if final_good else 1)
        lstm_pred.append(0 if early_good else 1)
        f_shock.append(1.0 - info["early_score"])
        f_shock_prime.append(1.0 - weight)

        clean_rows.extend(all_rows_by_vid[vid])

    out_csv = os.path.join(volume_dir, f"{args.output_prefix}_halide_filtered_cleantrain_dataframe_md.csv")
    out_csv_gp = os.path.join(volume_dir, f"{args.output_prefix}_halide_filtered_cleantrain_dataframe_md_gp_weights.csv")
    out_npy = os.path.join(volume_dir, f"{args.output_prefix}_halide_quality_scores_cleantrain_md.npy")

    fieldnames = list(clean_rows[0].keys())
    write_csv(out_csv, clean_rows, fieldnames)
    write_csv(out_csv_gp, clean_rows, fieldnames)

    quality = {
        "sel_idx": np.arange(len(selected_vids)),
        "actual_labels": np.array(actual_labels),
        "lstm_pred": np.array(lstm_pred),
        "f_shock": np.array(f_shock),
        "f_shock_prime": np.array(f_shock_prime),
        "halide_weights": np.array(halide_weights),
        "quadrant_idx": {
            "nonshock_nonshock": np.array(q00, dtype=int),
            "shock_nonshock": np.array(q10, dtype=int),
            "nonshock_shock": np.array(q01, dtype=int),
            "shock_shock": np.array(q11, dtype=int),
        },
    }

    np.save(out_npy, quality)

    print("Saved:")
    print(out_csv)
    print(out_csv_gp)
    print(out_npy)
    print()
    print("Total trajectories:", len(selected_vids))
    print("Total rows:", len(clean_rows))
    print("Early threshold:", early_threshold)
    print("Q00 early_good -> final_good:", len(q00))
    print("Q10 early_bad  -> final_good:", len(q10))
    print("Q01 early_good -> final_bad :", len(q01))
    print("Q11 early_bad  -> final_bad :", len(q11))
    print("Actions:", sorted(set(int(float(r["Action"])) for r in clean_rows)))

if __name__ == "__main__":
    main()