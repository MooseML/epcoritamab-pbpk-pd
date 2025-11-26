# src/epco_model/plotting.py

from __future__ import annotations
from typing import Iterable, Optional, Sequence

import numpy as np
import matplotlib.pyplot as plt

from .parameters import ModelParameters
from .state_vector import StateIx
from math import pi

def _tumor_radius_from_cells(N_cells: np.ndarray) -> np.ndarray:
    TUMOR_CELL_DENSITY = 1e9  # cells / cm^3
    V_cm3 = N_cells / TUMOR_CELL_DENSITY
    r = np.zeros_like(V_cm3)
    mask = V_cm3 > 0
    r[mask] = (3.0 * V_cm3[mask] / (4.0 * pi)) ** (1.0 / 3.0)
    return r


# Plasma concentration
def plot_plasma_concentration(
    t: np.ndarray,
    y: np.ndarray,
    params: ModelParameters,
    ax: Optional[plt.Axes] = None,
    label: str = None,
) -> plt.Axes:
    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 4))

    C = y[StateIx.DRUG_PLASMA] / params.pk.Vplasma
    ax.plot(t, C, label=label or "Plasma")
    ax.set_xlabel("Time [days]")
    ax.set_ylabel("Concentration [nmol/L]")
    ax.set_title("Plasma concentration")
    ax.grid(True)
    if label:
        ax.legend()
    return ax


# Node trimers (tumor vs B)
def plot_node_trimers(
    t: np.ndarray,
    y: np.ndarray,
    params: ModelParameters,
    ax: Optional[plt.Axes] = None,
) -> plt.Axes:
    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 4))

    pk = params.pk

    C_tri_t = y[StateIx.TRIMER_TUMOR] / pk.Vnode
    C_tri_b = y[StateIx.TRIMER_NODE] / pk.Vnode

    ax.plot(t, C_tri_t, label="Tumor trimers")
    ax.plot(t, C_tri_b, "--", label="B-cell trimers")
    ax.set_xlabel("Time [days]")
    ax.set_ylabel("Trimer conc. [nmol/L]")
    ax.set_title("Trimers in tumor node")
    ax.set_yscale("log")
    ax.grid(True)
    ax.legend()
    return ax


# Tumor trajectories
def plot_tumor_trajectory(
    t: np.ndarray,
    y: np.ndarray,
    ax: Optional[plt.Axes] = None,
) -> plt.Axes:
    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 4))

    N_tumor = y[StateIx.TUMOR_CELLS_TOTAL]
    r_tumor = _tumor_radius_from_cells(N_tumor)

    ax.plot(t, N_tumor, label="Tumor cells")
    ax.set_yscale("log")
    ax.set_xlabel("Time [days]")
    ax.set_ylabel("Tumor cells [cells]")
    ax.set_title("Tumor burden trajectory")
    ax.grid(True)

    ax2 = ax.twinx()
    ax2.plot(t, r_tumor, color="tab:orange", label="Radius")
    ax2.set_ylabel("Radius [cm]")

    return ax


# Tumor-directed ATC
def plot_tumor_ATCs(
    t: np.ndarray,
    y: np.ndarray,
    ax: Optional[plt.Axes] = None,
) -> plt.Axes:
    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 4))

    ATC = y[StateIx.ATC_TUMOR_NODE]
    pATC = y[StateIx.PATC_TUMOR_NODE]

    ax.plot(t, ATC, label="ATC")
    ax.plot(t, pATC, "--", label="pATC")
    ax.set_yscale("log")
    ax.set_xlabel("Time [days]")
    ax.set_ylabel("Cells")
    ax.set_title("Tumor-directed ATCs")
    ax.grid(True)
    ax.legend()
    return ax


# B-cell depletion (blood)
def plot_bcell_depletion(
    t: np.ndarray,
    y: np.ndarray,
    ax: Optional[plt.Axes] = None,
) -> plt.Axes:
    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 4))

    B = y[StateIx.B_BLOOD]

    ax.plot(t, B)
    ax.set_yscale("log")
    ax.set_xlabel("Time [days]")
    ax.set_ylabel("B cells in blood")
    ax.set_title("B-cell depletion")
    ax.grid(True)
    return ax


# Population plotting helpers
def plot_population_tumor_trajectories(
    t: np.ndarray,
    Ys: Sequence[np.ndarray],
    n_show: int = 20,
) -> plt.Axes:
    fig, ax = plt.subplots(figsize=(6, 4))
    for i, y in enumerate(Ys[:n_show]):
        ax.plot(t, y[StateIx.TUMOR_CELLS_TOTAL], alpha=0.6)
    ax.set_yscale("log")
    ax.set_xlabel("Time [days]")
    ax.set_ylabel("Tumor cells")
    ax.set_title(f"Tumor trajectories (first {n_show} patients)")
    ax.grid(True)
    return ax


def plot_ORR_vs_dose(
    doses: Sequence[float],
    ORRs: Sequence[float],
) -> plt.Axes:
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(doses, ORRs, "o-")
    ax.set_xlabel("Dose [mg]")
    ax.set_ylabel("ORR")
    ax.set_title("ORR vs dose")
    ax.grid(True)
    return ax
