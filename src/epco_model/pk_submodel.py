# pk_submodel.py
"""
PK submodel for epcoritamab (minimal PBPK).
------------------------------------------

This module computes the contribution of the pharmacokinetic (drug-disposition)
processes to dy/dt for the drug-related states:

- SC depot
- Plasma
- Tight tissue ISF
- Leaky tissue ISF
- Spleen ISF
- Tumor lymph node
- Lymph

We work in amounts ( nmol or mg), with concentrations computed as:

    C_compartment = A_compartment / V_compartment

Time is in days, flows are in L/day, clearances in L/day, etc.
"""

from __future__ import annotations

import numpy as np

from .parameters import ModelParameters
from .state_vector import StateIx


def update_dydt_pk(
    t: float,
    y: np.ndarray,
    params: ModelParameters,
    dydt: np.ndarray,
) -> None:
    """
    Update PK-related entries of dydt in place.

    Args:
        t: Time (days). Included for compatibility; not used directly here.
        y: State vector [A_SC, A_PLASMA, A_TIGHT, A_LEAKY, A_SPLEEN, A_NODE, A_LYMPH, ...]
        params: ModelParameters containing pk sub-parameters.
        dydt: Array of same shape as y; this function *adds* PK contributions to dydt.
    """
    pk = params.pk

    # Unpack drug amounts
    A_sc = y[StateIx.DRUG_SC]
    A_plasma = y[StateIx.DRUG_PLASMA]
    A_tight = y[StateIx.DRUG_TIGHT]
    A_leaky = y[StateIx.DRUG_LEAKY]
    A_spleen = y[StateIx.DRUG_SPLEEN]
    A_node = y[StateIx.DRUG_NODE]
    A_lymph = y[StateIx.DRUG_LYMPH]

    # Convert to concentrations
    Vp = pk.Vplasma
    Vtight = pk.Vtight
    Vleaky = pk.Vleaky
    Vspleen = pk.Vspleen
    Vnode = pk.Vnode
    Vlymph = pk.Vlymph

    C_plasma = A_plasma / Vp if Vp > 0 else 0.0
    C_tight = A_tight / Vtight if Vtight > 0 else 0.0
    C_leaky = A_leaky / Vleaky if Vleaky > 0 else 0.0
    C_spleen = A_spleen / Vspleen if Vspleen > 0 else 0.0
    C_node = A_node / Vnode if Vnode > 0 else 0.0
    C_lymph = A_lymph / Vlymph if Vlymph > 0 else 0.0

    # ----------------------------------------------------------------------------
    # SC absorption
    # ----------------------------------------------------------------------------
    # Amount leaving SC depot goes to lymph (per paper Eq. S1).
    Absorption = pk.ka * A_sc  # amount/day
    dA_sc = -Absorption        # loss from SC depot

    # ----------------------------------------------------------------------------
    # Linear + nonlinear clearance from plasma
    # ----------------------------------------------------------------------------
    # Linear: CL * C_plasma  (amount/day)
    CL_linear = pk.CL * C_plasma

    # Nonlinear: Vmax * C_plasma / (Km + C_plasma)
    eps = 1e-12
    CL_nonlinear = pk.Vmax * C_plasma / (pk.Km + C_plasma + eps)

    # ----------------------------------------------------------------------------
    # Tissue <-> lymph & plasma <-> lymph exchange
    # ----------------------------------------------------------------------------
    # For each tissue (tight, leaky, spleen):
    #   Plasma -> tissue: L_j * (1 - sigma_j) * C_plasma
    #   Tissue -> lymph:  L_j * C_tissue
    #
    # Lymph -> plasma:
    #   Lymph -> plasma: L * C_lymph (UNIDIRECTIONAL per paper Figure S1 & equations)
    #
    # Note: Unlike tissue-lymph flows, lymph-plasma flow is one-way only.

    Ltight = pk.Ltight
    Lleaky = pk.Lleaky
    Lspleen = pk.Lspleen
    L_lymph = pk.L

    sigma_tight = pk.sigma_tight
    sigma_leaky = pk.sigma_leaky
    # sigma_lymph not used (no plasma->lymph flow per paper)

    # Plasma -> tight / tight -> lymph
    Distribution_plasma_to_tight = Ltight * (1.0 - sigma_tight) * C_plasma   # amount/day
    Distribution_tight_to_lymph = Ltight * C_tight                           # amount/day

    # Plasma -> leaky / leaky -> lymph
    Distribution_plasma_to_leaky = Lleaky * (1.0 - sigma_leaky) * C_plasma
    Distribution_leaky_to_lymph = Lleaky * C_leaky

    # Plasma -> spleen / spleen -> node
    # Use sigma_leaky for spleen as an approximation.
    Distribution_plasma_to_spleen = Lspleen * (1.0 - sigma_leaky) * C_plasma
    Distribution_spleen_to_node = Lspleen * C_spleen

    # Node -> lymph
    # No explicit separate flow is given in the table; we reuse Lspleen as the
    # outflow from node to lymph for simplicity.
    Distribution_node_to_lymph = Lspleen * C_node

    # Lymph -> plasma (UNIDIRECTIONAL per paper)
    Distribution_lymph_to_plasma = L_lymph * C_lymph
    # No reverse flow (plasma -> lymph) per paper Figure S1 and equations

    # ----------------------------------------------------------------------------
    # Assemble compartment derivatives
    # ----------------------------------------------------------------------------

    # Plasma:
    #   - CL_linear
    #   - CL_nonlinear
    #   - Distribution to tissues (leaky, tight, spleen)
    #   + Distribution_lymph_to_plasma (unidirectional: lymph → plasma only)
    #   ± Binding terms (handled in binding_submodel)
    dA_plasma = (
        - CL_linear
        - CL_nonlinear
        - Distribution_plasma_to_leaky
        - Distribution_plasma_to_tight
        - Distribution_plasma_to_spleen
        + Distribution_lymph_to_plasma
        # ± Binding_CD3 ± Binding_CD20 will be added in binding_submodel
    )

    # Tight tissue:
    #   + Distribution_plasma_to_tight
    #   - Distribution_tight_to_lymph
    dA_tight = (
        + Distribution_plasma_to_tight
        - Distribution_tight_to_lymph
    )

    # Leaky tissue:
    #   + Distribution_plasma_to_leaky
    #   - Distribution_leaky_to_lymph
    dA_leaky = (
        + Distribution_plasma_to_leaky
        - Distribution_leaky_to_lymph
    )

    # Spleen tissue:
    #   + Distribution_plasma_to_spleen
    #   - Distribution_spleen_to_node
    #   ± Binding terms will be added in binding_submodel
    dA_spleen = (
        + Distribution_plasma_to_spleen
        - Distribution_spleen_to_node
    )

    # Tumor lymph node:
    #   + Distribution_spleen_to_node
    #   - Distribution_node_to_lymph
    #   ± Binding terms will be added in binding_submodel
    dA_node = (
        + Distribution_spleen_to_node
        - Distribution_node_to_lymph
    )

    # Lymph:
    #   + Absorption (SC -> lymph, per paper Eq. S1 line 22)
    #   + Distribution_leaky_to_lymph
    #   + Distribution_tight_to_lymph
    #   + Distribution_node_to_lymph
    #   - Distribution_lymph_to_plasma (unidirectional: lymph → plasma only)
    #   ± Binding terms will be added in binding_submodel
    dA_lymph = (
        + Absorption
        + Distribution_leaky_to_lymph
        + Distribution_tight_to_lymph
        + Distribution_node_to_lymph
        - Distribution_lymph_to_plasma
    )

    # ----------------------------------------------------------------------------
    # Add contributions to global dydt
    # ----------------------------------------------------------------------------
    dydt[StateIx.DRUG_SC] += dA_sc
    dydt[StateIx.DRUG_PLASMA] += dA_plasma
    dydt[StateIx.DRUG_TIGHT] += dA_tight
    dydt[StateIx.DRUG_LEAKY] += dA_leaky
    dydt[StateIx.DRUG_SPLEEN] += dA_spleen
    dydt[StateIx.DRUG_NODE] += dA_node
    dydt[StateIx.DRUG_LYMPH] += dA_lymph
