import argparse
import csv
import os
from pathlib import Path

import numpy as np
from ase.db import connect
from dscribe.descriptors import MBTR
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor


DEFAULT_DB_PATH = "/home/luis-menezes/Documents/MSc/c2db_free_atoms/data/raw/c2db.db"
DEFAULT_SELECTION = "gap>=0"


def get_all_species(db_path, selection):
    db = connect(db_path)
    all_elements = set()
    n_atoms_max = 0
    for row in db.select(selection):
        all_elements.update(row.symbols)
        n_atoms_max = max(n_atoms_max, row.natoms)
    return sorted(all_elements), n_atoms_max


def load_dataset(db_path, selection, target_key, max_samples=None):
    db = connect(db_path)
    atoms_list = []
    targets = []
    ids = []

    count = 0
    for row in db.select(selection):
        gap_val = getattr(row, target_key, None)
        if gap_val is None or np.isnan(gap_val):
            continue

        atoms = row.toatoms()
        if atoms is None or len(atoms) == 0:
            continue

        atoms_list.append(atoms)
        targets.append(float(gap_val))
        ids.append(row.id)
        count += 1

        if max_samples is not None and count >= max_samples:
            break

    return atoms_list, np.asarray(targets, dtype=np.float32), np.asarray(ids, dtype=np.int32)


def compute_grid_n(grid_min, grid_max, resolution):
    return int((grid_max - grid_min + 1) * resolution - (resolution - 1))


def build_mbtr_k2(species, grid_min, grid_max, resolution, grid_sigma, weight_rcut, weight_threshold):
    grid_n = compute_grid_n(grid_min, grid_max, resolution)
    return MBTR(
        species=species,
        geometry={"function": "inverse_distance"},
        grid={"min": grid_min, "max": grid_max, "n": grid_n, "sigma": grid_sigma},
        weighting={
            "function": "exp",
            # "scale": weight_scale,
            "r_cut": weight_rcut,
            "threshold": weight_threshold,
        },
        periodic=True,
        sparse=False,
        normalization="none",
    )


def compute_features(mbtr, atoms_list, batch_size, n_jobs):
    if batch_size is None or batch_size <= 0:
        return np.asarray(mbtr.create(atoms_list, n_jobs=n_jobs), dtype=np.float32)

    features = []
    for i in range(0, len(atoms_list), batch_size):
        batch = atoms_list[i : i + batch_size]
        feat = np.asarray(mbtr.create(batch, n_jobs=n_jobs), dtype=np.float32)
        features.append(feat)
    return np.vstack(features)


def configure_xgboost(device):
    if device == "cpu":
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
        return {"tree_method": "hist", "device": "cpu"}

    return {"tree_method": "hist", "device": "cuda"}


def train_and_evaluate(x_train, x_test, y_train, y_test, xgb_params, n_jobs):
    model = XGBRegressor(
        objective="reg:squarederror",
        n_estimators=500,
        max_depth=8,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        n_jobs=n_jobs,
        random_state=42,
        **xgb_params,
    )
    model.fit(x_train, y_train)
    preds = model.predict(x_test).reshape(-1)

    rmse = float(np.sqrt(mean_squared_error(y_test, preds)))
    mae = float(mean_absolute_error(y_test, preds))
    r2 = float(r2_score(y_test, preds))
    return rmse, mae, r2


def main():
    parser = argparse.ArgumentParser(
        description="Train an MLP on MBTR features for multiple k2 hyperparameters.",
    )
    parser.add_argument("--db-path", type=Path, default=Path(DEFAULT_DB_PATH))
    parser.add_argument("--selection", type=str, default=DEFAULT_SELECTION)
    parser.add_argument("--target", type=str, default="gap", choices=["gap", "gap_dir_nosoc"])
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--random-seed", type=int, default=24)
    parser.add_argument("--n-jobs", type=int, default=16)
    parser.add_argument("--batch-size", type=int, default=256)

    parser.add_argument("--grid-min", type=float, default=0.0)
    parser.add_argument("--grid-max", type=float, default=1.0)
    parser.add_argument("--grid-sigma", type=float, default=0.05)
    parser.add_argument("--resolution", type=int, default=128)
    parser.add_argument("--weight-rcut", type=float, default=0.7)
    parser.add_argument(
        "--sweep-param",
        type=str,
        required=True,
        choices=["grid_sigma", "resolution", "weight_rcut"],
        help="Single parameter to sweep while keeping others fixed.",
    )
    parser.add_argument(
        "--sweep-values",
        type=float,
        nargs="+",
        required=True,
        help="Values for the sweep parameter.",
    )
    # parser.add_argument("--weight-rcut", type=float, default=None)
    parser.add_argument("--weight-threshold", type=float, default=1e-3)
    parser.add_argument(
        "--device",
        type=str,
        default="gpu",
        choices=["cpu", "gpu"],
        help="Device for XGBoost training.",
    )
    parser.add_argument(
        "--out-file",
        type=Path,
        default=Path("mbtr_xgb_results.csv"),
        help="CSV file to store all results.",
    )

    args = parser.parse_args()

    xgb_params = configure_xgboost(args.device)

    species, n_atoms_max = get_all_species(str(args.db_path), args.selection)
    atoms_list, targets, ids = load_dataset(
        str(args.db_path),
        args.selection,
        args.target,
        max_samples=args.max_samples,
    )

    if len(targets) == 0:
        raise SystemExit("No valid structures with target values found.")

    indices = np.arange(len(targets))
    train_idx, test_idx = train_test_split(
        indices,
        test_size=args.test_size,
        random_state=args.random_seed,
        shuffle=True,
    )

    print(f"Loaded {len(targets)} structures (max atoms per structure: {n_atoms_max})")
    print(f"Train/test split: {len(train_idx)}/{len(test_idx)}")
    print(f"Species count: {len(species)}")

    results = []

    base_params = {
        "grid_sigma": args.grid_sigma,
        "resolution": args.resolution,
        "weight_rcut": args.weight_rcut,
    }
    sweep_values = args.sweep_values

    for value in sweep_values:
        params = dict(base_params)
        params[args.sweep_param] = value
        mbtr = build_mbtr_k2(
            species=species,
            grid_min=args.grid_min,
            grid_max=args.grid_max,
            resolution=int(params["resolution"]),
            grid_sigma=float(params["grid_sigma"]),
            # weight_scale=float(params["weight_scale"]),
            weight_rcut=float(params["weight_rcut"]),
            weight_threshold=args.weight_threshold,
        )

        features = compute_features(mbtr, atoms_list, args.batch_size, args.n_jobs)
        x_train = features[train_idx]
        x_test = features[test_idx]
        y_train = targets[train_idx]
        y_test = targets[test_idx]

        scaler = StandardScaler()
        x_train = scaler.fit_transform(x_train)
        x_test = scaler.transform(x_test)

        rmse, mae, r2 = train_and_evaluate(
            x_train, x_test, y_train, y_test, xgb_params, args.n_jobs
        )

        results.append(
            {
                "grid_sigma": float(params["grid_sigma"]),
                "resolution": int(params["resolution"]),
                "weight_rcut": float(params["weight_rcut"]),
                "rmse": rmse,
                "mae": mae,
                "r2": r2,
            }
        )

        print(
            "grid_sigma={:.4f} resolution={} weight_rcut={:.4f} | RMSE={:.4f} MAE={:.4f} R2={:.4f}".format(
                params["grid_sigma"], params["resolution"], params["weight_rcut"], rmse, mae, r2
            )
        )

    results_sorted = sorted(results, key=lambda x: x["rmse"])

    out_path = args.out_file
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["grid_sigma", "resolution", "weight_rcut", "rmse", "mae", "r2"],
        )
        writer.writeheader()
        writer.writerows(results_sorted)

    print(f"\nSaved results to: {out_path.resolve()}")

    print("\nTop configs by RMSE:")
    for row in results_sorted[:10]:
        print(
            "grid_sigma={:.4f} resolution={} weight_rcut={:.4f} | RMSE={:.4f} MAE={:.4f} R2={:.4f}".format(
                row["grid_sigma"],
                row["resolution"],
                row["weight_rcut"],
                row["rmse"],
                row["mae"],
                row["r2"],
            )
        )


if __name__ == "__main__":
    main()
