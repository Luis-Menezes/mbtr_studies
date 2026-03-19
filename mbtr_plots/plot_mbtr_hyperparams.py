from pathlib import Path
import argparse

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from ase import Atoms
from dscribe.descriptors import MBTR


def create_linear_carbon_chain(n=5, bond_length=1.34):
    positions = [(i * bond_length, 0.0, 0.0) for i in range(n)]
    return Atoms("C" * n, positions=positions)


def create_star_carbon_structure(n_outer=4, bond_length=1.34):
    positions = [(0.0, 0.0, 0.0)]  # center carbon
    for i in range(n_outer):
        angle = 2 * np.pi * i / n_outer
        positions.append((bond_length * np.cos(angle), bond_length * np.sin(angle), 0.0))
    return Atoms("C" * (n_outer + 1), positions=positions)

def get_n_grid(x_max, x_min, resolution):
    return ((x_max - x_min + 1)*resolution) - (resolution - 1)

def build_mbtr_k1(resolution=80, sigma=0.10):
    min_val, max_val = 0.0, 10.0
    n = get_n_grid(max_val, min_val, resolution)
    print(f"Building k1 with grid: min={min_val}, max={max_val}, n={n}, sigma={sigma}")
    return MBTR(
        species=["C"],
        geometry={"function": "atomic_number"},
        grid={"min": min_val, "max": max_val, "n": n, "sigma": sigma},
        periodic=False,
        sparse=False,
        normalization="none",
    )


def build_mbtr_k2(resolution=200, sigma=0.05, weight_scale=0.7, threshold=1e-3):
    x_max, x_min = 1.0, 0.0
    n = get_n_grid(x_max, x_min, resolution=resolution)
    return MBTR(
        species=["C"],
        geometry={"function": "inverse_distance"},
        grid={"min": x_min, "max": x_max, "n": n, "sigma": sigma},
        weighting={"function": "exp", "scale": weight_scale, "threshold": threshold},
        periodic=False,
        sparse=False,
        normalization="none",
    )


def build_mbtr_k3(n=180, sigma=3.0, weight_scale=0.7):
    return MBTR(
        species=["C"],
        k3={
            "geometry": {"function": "angle"},
            "grid": {"min": 0.0, "max": 180.0, "n": n, "sigma": sigma},
            "weighting": {"function": "exp", "scale": weight_scale, "threshold": 1e-3},
        },
        periodic=False,
        flatten=True,
        sparse=False,
        normalization="none",
    )


def mbtr_vector(descriptor, atoms):
    return np.asarray(descriptor.create(atoms)).reshape(-1)


def plot_structures(structures, outdir):
    fig, axes = plt.subplots(1, len(structures), figsize=(5 * len(structures), 4))
    if len(structures) == 1:
        axes = [axes]

    for ax, (name, atoms) in zip(axes, structures.items()):
        pos = atoms.get_positions()
        ax.scatter(pos[:, 0], pos[:, 1], s=120, color="tab:blue", edgecolors="black")
        for i, (x, y, _) in enumerate(pos):
            ax.text(x + 0.03, y + 0.03, f"C{i}", fontsize=8)
        ax.set_title(name)
        ax.set_xlabel("x (Å)")
        ax.set_ylabel("y (Å)")
        # ax.set_xlim(-1)
        ax.set_ylim(-1.5, 1.5)
        ax.set_aspect("equal")
        ax.grid(alpha=0.25)

    fig.tight_layout()
    fig.savefig(outdir / "structures.png", dpi=220)
    plt.close(fig)


def plot_baseline_terms(structures, builders, base_params, outdir, terms=['k2']):
    fig, axes = plt.subplots(3, len(structures), figsize=(5 * len(structures), 10))
    if len(structures) == 1:
        axes = np.expand_dims(axes, axis=1)

    for r, term in enumerate(terms):
        descriptor = builders[term](**base_params[term])
        for c, (name, atoms) in enumerate(structures.items()):
            vec = mbtr_vector(descriptor, atoms)
            print(f"Value for {name}: {vec.sum():.4f}")
            x = np.linspace(descriptor.grid["min"], descriptor.grid["max"], vec.size)
            ax = axes[r, c]
            ax.plot(x, vec, color="tab:blue", lw=1.25)
            ax.set_title(f"{name} — {term}")
            ax.set_xlabel("Normalized feature index")
            ax.set_ylabel("Intensity")
            ax.grid(alpha=0.25)

    fig.tight_layout()
    fig.savefig(outdir / "mbtr_baseline_k1_k2_k3.png", dpi=220)
    plt.close(fig)


def plot_hyperparameter_sweep(structures, builders, base_params, outdir, term, param_name, values):
    fig, axes = plt.subplots(1, len(structures), figsize=(5 * len(structures), 4))
    if len(structures) == 1:
        axes = [axes]

    for value in values:
        params = dict(base_params[term])
        params[param_name] = value
        descriptor = builders[term](**params)
        grid_max = descriptor.grid["max"]
        grid_min = descriptor.grid["min"]
        for ax, (name, atoms) in zip(axes, structures.items()):
            vec = mbtr_vector(descriptor, atoms)
            x = np.linspace(grid_min, grid_max, vec.size)
            ax.plot(x, vec, lw=1.2, label=f"{param_name}={value}")
            ax.set_title(f"{name} — {term}")
            ax.set_xlabel("Feature index")
            ax.set_ylabel("Intensity")
            ax.grid(alpha=0.25)

    for ax in axes:
        ax.legend(fontsize=8)

    fig.tight_layout()
    fig.savefig(outdir / f"{term}_{param_name}_sweep.png", dpi=220)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(
        description="Generate MBTR plots for simple carbon structures and hyperparameter sweeps."
    )
    parser.add_argument(
        "--outdir",
        type=Path,
        default=Path("mbtr_plots"),
        help="Directory where plots are saved.",
    )
    args = parser.parse_args()

    outdir = args.outdir
    outdir.mkdir(parents=True, exist_ok=True)

    structures = {
        "Linear C=C=C=C=C=C": create_linear_carbon_chain(n=6, bond_length=1.34),
        "Star C6 (center + 5 outer)": create_star_carbon_structure(n_outer=5, bond_length=1.34),
    }

    builders = {
        "k1": build_mbtr_k1,
        "k2": build_mbtr_k2,
        "k3": build_mbtr_k3,
    }

    base_params = {
        # "k1": {"resolution": 1, "sigma": 1},
        "k2": {"resolution": 200, "sigma": 0.05, "weight_scale": 0.7, "threshold": 1e-3},
        # "k3": {"resolution": 180, "sigma": 3.0, "weight_scale": 0.7},
    }

    sweeps = {
        # "k1": {"sigma": [1e-4, 0.10, 1], "resolution": [1, 2, 10, 100]},
        "k2": {"sigma": [0.001, 0.01, 0.15, 0.5, 0.7], "resolution": [5, 10, 100, 200], "weight_scale": [0.3, 0.7, 1.5], "threshold": [1e-4, 1e-3, 1e-2]},
        # "k3": {"sigma": [1.0, 3.0, 8.0], "resolution": [60, 180, 320], "weight_scale": [0.3, 0.7, 1.5]},
    }

    plot_structures(structures, outdir)
    plot_baseline_terms(structures, builders, base_params, outdir, terms=base_params.keys())

    for term, params in sweeps.items():
        for param_name, values in params.items():
            plot_hyperparameter_sweep(structures, builders, base_params, outdir, term, param_name, values)

    print(f"Saved MBTR plots to: {outdir.resolve()}")
    for f in sorted(outdir.glob("*.png")):
        print(f" - {f.name}")


if __name__ == "__main__":
    main()