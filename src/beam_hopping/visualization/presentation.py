#!/usr/bin/env python3
"""Generate presentation-ready plots from existing experiment CSV files.

The script does not rerun optimization. It only reads stored result tables and
creates clearer figures for external presentation.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parents[3]
RESULTS = ROOT / "results"
FIGURES = Path("runs/figures")
TOLERANCE = 1e-6


def configure_matplotlib() -> None:
    plt.rcParams.update(
        {
            "figure.dpi": 150,
            "savefig.dpi": 300,
            "font.size": 12,
            "axes.titlesize": 16,
            "axes.labelsize": 13,
            "legend.fontsize": 11,
            "xtick.labelsize": 11,
            "ytick.labelsize": 11,
            "axes.grid": True,
            "grid.alpha": 0.25,
            "grid.linestyle": "-",
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )


def save_figure(fig: plt.Figure, stem: str) -> None:
    FIGURES.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGURES / f"{stem}.png", bbox_inches="tight")
    fig.savefig(FIGURES / f"{stem}.pdf", bbox_inches="tight")
    plt.close(fig)


def plot_integrality_violation() -> None:
    df = pd.read_csv(RESULTS / "single_slot" / "phase_1_single_slot_results.csv")
    lp = df[df["formulation"] == "LP_relaxation"].copy()
    lp = lp.sort_values(["variable_count", "M", "N", "K"]).reset_index(drop=True)

    grouped = (
        lp.groupby(["M", "N", "K", "variable_count"], as_index=False)
        .agg(
            max_integrality_violation=("integrality_violation", "max"),
            mean_integrality_violation=("integrality_violation", "mean"),
            rows=("integrality_violation", "size"),
        )
        .sort_values("variable_count")
        .reset_index(drop=True)
    )

    fig, ax = plt.subplots(figsize=(8.8, 5.2))
    x = np.arange(len(grouped))
    ax.bar(
        x,
        grouped["max_integrality_violation"],
        width=0.58,
        color="#2E86AB",
        label="maximum observed violation",
    )
    ax.axhline(
        TOLERANCE,
        color="#C0392B",
        linestyle="--",
        linewidth=1.8,
        label=r"numerical tolerance $10^{-6}$",
    )

    ax.set_yscale("log")
    ax.set_title("LP Relaxation Returns Integral Beam-Cell Decisions")
    ax.set_ylabel(r"max distance of $x_t^{b,i}$ to $\{0,1\}$")
    ax.set_xlabel(r"Single-slot decision variables: $MN$ beam-cell pairs")
    ax.set_xticks(x)
    ax.set_xticklabels(
        [
            rf"$M={int(row.M)},\,N={int(row.N)},\,K={int(row.K)}$"
            + "\n"
            + rf"$MN={int(row.variable_count)}$ variables"
            for row in grouped.itertuples(index=False)
        ]
    )
    ax.legend(loc="upper left", frameon=True)
    ax.text(
        0.01,
        -0.31,
        r"$M$: number of candidate beams; "
        r"$N$: number of ground cells; "
        r"$K$: selected beam-cell pairs per slot."
        "\n"
        r"The LP relaxation uses $0\leq x_t^{b,i}\leq 1$; "
        r"all violations remain below the numerical tolerance.",
        transform=ax.transAxes,
        fontsize=10.5,
        va="top",
    )
    save_figure(fig, "presentation_integrality_violation")


def plot_mosek_solver_time() -> None:
    df = pd.read_csv(RESULTS / "single_slot" / "phase_1_large_scale_summary.csv")
    df = df.sort_values(["variable_count", "formulation"]).copy()

    labels = {
        "BILP": "Binary formulation",
        "LP_relaxation": "LP relaxation",
    }
    colors = {
        "BILP": "#D55E00",
        "LP_relaxation": "#0072B2",
    }
    markers = {
        "BILP": "s",
        "LP_relaxation": "o",
    }

    fig, ax = plt.subplots(figsize=(9.0, 5.4))
    for formulation in ["BILP", "LP_relaxation"]:
        part = df[df["formulation"] == formulation].sort_values("variable_count")
        ax.plot(
            part["variable_count"],
            part["mean_solver_time"],
            marker=markers[formulation],
            linewidth=2.4,
            markersize=7,
            color=colors[formulation],
            label=labels[formulation],
        )

    unique_sizes = (
        df[["M", "N", "K", "variable_count"]]
        .drop_duplicates()
        .sort_values("variable_count")
        .reset_index(drop=True)
    )

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_title("MOSEK Solver Time for Single-Slot Beam-Cell Matching")
    ax.set_xlabel(r"Number of decision variables $MN$")
    ax.set_ylabel("mean solver-reported time (seconds)")
    ax.set_xticks(unique_sizes["variable_count"])
    ax.set_xticklabels(
        [
            rf"$MN={int(row.variable_count)}$"
            + "\n"
            + rf"$M={int(row.M)},\,N={int(row.N)},\,K={int(row.K)}$"
            for row in unique_sizes.itertuples(index=False)
        ]
    )
    ax.legend(loc="upper left", frameon=True)
    ax.text(
        0.01,
        -0.33,
        r"Each variable is one beam-cell association decision $x_t^{b,i}$, "
        r"so the single-slot model has $MN$ variables."
        "\n"
        r"$K$ fixes how many beam-cell pairs are active in the slot and enters the cardinality constraint.",
        transform=ax.transAxes,
        fontsize=10.5,
        va="top",
    )
    save_figure(fig, "presentation_mosek_solver_time")


def plot_objective_ratio() -> None:
    df = pd.read_csv(RESULTS / "heuristic_comparison" / "phase_3_single_slot_comparison.csv")
    lp = (
        df[df["method"] == "LP_relaxation_TU"][["instance_id", "objective_value"]]
        .rename(columns={"objective_value": "lp_objective"})
    )
    merged = df.merge(lp, on="instance_id", how="inner")
    merged["objective_ratio"] = merged["objective_value"] / merged["lp_objective"]

    methods = [
        "LP_relaxation_TU",
        "BILP_benchmark",
        "greedy_throughput",
        "genetic_algorithm",
        "demand_priority_greedy",
        "random_feasible_matching",
    ]
    labels = [
        "LP relaxation",
        "Binary benchmark",
        "Greedy throughput",
        "Genetic algorithm",
        "Demand priority greedy",
        "Random feasible",
    ]
    data = [
        merged.loc[merged["method"] == method, "objective_ratio"].to_numpy()
        for method in methods
    ]

    fig, ax = plt.subplots(figsize=(10.4, 5.6))
    ax.boxplot(data, tick_labels=labels, showmeans=True)
    ax.axhline(1.0, color="#333333", linewidth=1.2, linestyle="--")
    ax.set_ylim(0.91, 1.005)
    ax.set_title("Objective Quality Relative to the Matching Optimum")
    ax.set_ylabel(r"objective ratio $J_{\mathrm{method}}/J_{\mathrm{LP}}$")
    ax.tick_params(axis="x", rotation=18)
    ax.text(
        0.01,
        -0.28,
        r"The LP relaxation is the certified optimum for each instance. "
        r"All methods use the same feasible edge set $\mathcal{E}_t$ and edge weights $w_t^{b,i}$.",
        transform=ax.transAxes,
        fontsize=10.5,
        va="top",
    )
    save_figure(fig, "presentation_phase3_objective_ratio")


def main(argv=None) -> int:
    global RESULTS, FIGURES
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", type=Path, default=RESULTS)
    parser.add_argument("--output-dir", type=Path, default=Path("runs/figures"))
    args = parser.parse_args(argv)
    RESULTS = args.results_dir.resolve()
    FIGURES = args.output_dir.resolve()
    if not RESULTS.is_dir():
        parser.error("results directory not found; supply --results-dir")
    if FIGURES.is_relative_to(RESULTS):
        parser.error("choose an output directory outside the recorded results archive")
    configure_matplotlib()
    plot_integrality_violation()
    plot_mosek_solver_time()
    plot_objective_ratio()
    print(f"Wrote presentation figures to {FIGURES}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
