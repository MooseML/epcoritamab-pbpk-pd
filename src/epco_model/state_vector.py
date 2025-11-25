# state_vector.py
# single source of truth for the order of states in the ODE vector y

# Once PK + T/B are working:

# Add tumor states to the enum and get_initial_state:

# e.g., TUMOR_CELLS_TOTAL, TUMOR_CELLS_REACHABLE, TUMOR_RADIUS, etc.

# Add binding species:

# e.g., FREE_CD3_BLOOD, DIMER_CD3_AB_BLOOD, TRIMER_BLOOD, etc. per compartment.

# Add ATC states:

# e.g., ATC_BLOOD, ATC_SPLEEN, ATC_TUMOR, …

# You don’t have to pre-plan every single index today; just keep them grouped logically and add at the end of the enum as you extend the model.

from enum import IntEnum
import numpy as np
from .parameters import ModelParameters
import numpy as np 

class StateIx(IntEnum):
    # PK states (amounts or concentrations: decide once & be consistent)
    DRUG_SC = 0
    DRUG_PLASMA = 1
    DRUG_TIGHT = 2
    DRUG_LEAKY = 3
    DRUG_SPLEEN = 4
    DRUG_NODE = 5
    DRUG_LYMPH = 6

    # Lymphocyte counts (one per compartment)
    T_BLOOD = 7
    T_SPLEEN = 8
    T_NODE = 9
    T_LYMPH = 10

    B_BLOOD = 11
    B_SPLEEN = 12
    B_NODE = 13
    B_LYMPH = 14

    AF_T = 15
    AF_B = 16
    INJ = 17
    
    # --- Binding states --- 
    # BLOOD
    CD3_AB_DIMER_BLOOD = 18
    CD20_AB_DIMER_BLOOD = 19
    TRIMER_BLOOD = 20

    # SPLEEN
    CD3_AB_DIMER_SPLEEN = 21
    CD20_AB_DIMER_SPLEEN = 22
    TRIMER_SPLEEN = 23

    # TUMOR NODE
    CD3_AB_DIMER_NODE = 24
    CD20_AB_DIMER_NODE = 25
    TRIMER_NODE = 26

    # LYMPH
    CD3_AB_DIMER_LYMPH = 27
    CD20_AB_DIMER_LYMPH = 28
    TRIMER_LYMPH = 29

    # Activated T cells against B cells (ATC_B) ----
    ATC_B_BLOOD = 30
    ATC_B_SPLEEN = 31
    ATC_B_NODE = 32
    ATC_B_LYMPH = 33

    # Clonally expanded activated T cells against B cells (pATC_B)
    PATC_B_BLOOD = 34
    PATC_B_SPLEEN = 35
    PATC_B_NODE = 36
    PATC_B_LYMPH = 37

    # ---- Tumor-specific ATC states ----
    ATC_TUMOR_NODE = 38
    PATC_TUMOR_NODE = 39

    # ---- Tumor burden ----
    TUMOR_CELLS_TOTAL = 40
    
    # ---- Tumor-specific binding in tumor node ----
    CD20_AB_DIMER_TUMOR = 41
    TRIMER_TUMOR = 42


N_STATES = max(StateIx) + 1  # assumes enum values are 0..N-1


def get_initial_state(params: ModelParameters) -> np.ndarray:
    y0 = np.zeros(N_STATES, dtype=float)

    # --- PK: no drug initially ---
    y0[StateIx.DRUG_SC] = 0.0
    y0[StateIx.DRUG_PLASMA] = 0.0
    y0[StateIx.DRUG_TIGHT] = 0.0
    y0[StateIx.DRUG_LEAKY] = 0.0
    y0[StateIx.DRUG_SPLEEN] = 0.0
    y0[StateIx.DRUG_NODE] = 0.0
    y0[StateIx.DRUG_LYMPH] = 0.0

    traf = params.trafficking
    pk = params.pk

    # ---------- Helper: total pools from "2% in blood" assumption ----------
    Vblood_L = traf.Vblood
    factor = Vblood_L * 1e6  # 1 L = 1e6 mm^3

    # Baseline blood densities (cells/mm^3)
    TC_base = traf.TCplasma_base
    BC_base = traf.BCplasma_base

    # Blood counts at baseline (this is the 2% slice)
    T_blood_baseline = TC_base * factor
    B_blood_baseline = BC_base * factor

    # Total body pools given "2% in blood"
    TOTAL_T = T_blood_baseline / 0.02
    TOTAL_B = B_blood_baseline / 0.02

    # Fractions: 2% blood, 60% spleen, 38% (node + lymph)
    frac_blood  = 0.02
    frac_spleen = 0.60
    frac_rest   = 0.38

    # Split rest between node and lymph by relative volume
    Vnode = pk.Vnode
    Vlymph = pk.Vlymph
    frac_node = Vnode / (Vnode + Vlymph)
    frac_lymph = Vlymph / (Vnode + Vlymph)

    # ---------- T cells ----------
    y0[StateIx.T_BLOOD] = TOTAL_T * frac_blood
    y0[StateIx.T_SPLEEN] = TOTAL_T * frac_spleen
    y0[StateIx.T_NODE] = TOTAL_T * frac_rest * frac_node
    y0[StateIx.T_LYMPH] = TOTAL_T * frac_rest * frac_lymph

    # ---------- B cells ----------
    y0[StateIx.B_BLOOD] = TOTAL_B * frac_blood
    y0[StateIx.B_SPLEEN] = TOTAL_B * frac_spleen
    y0[StateIx.B_NODE] = TOTAL_B * frac_rest * frac_node
    y0[StateIx.B_LYMPH] = TOTAL_B * frac_rest * frac_lymph

    # ---------- Homeostasis & injection ----------
    y0[StateIx.AF_T] = 1.0
    y0[StateIx.AF_B] = 1.0
    y0[StateIx.INJ]  = 0.0

    # ---------- BLOOD ----------
    y0[StateIx.CD3_AB_DIMER_BLOOD] = 0.0
    y0[StateIx.CD20_AB_DIMER_BLOOD] = 0.0
    y0[StateIx.TRIMER_BLOOD]  = 0.0

    # ---------- SPLEEN ----------
    y0[StateIx.CD3_AB_DIMER_SPLEEN] = 0.0
    y0[StateIx.CD20_AB_DIMER_SPLEEN] = 0.0
    y0[StateIx.TRIMER_SPLEEN]  = 0.0

    # ---------- TUMOR NODE----------
    y0[StateIx.CD3_AB_DIMER_NODE] = 0.0
    y0[StateIx.CD20_AB_DIMER_NODE] = 0.0
    y0[StateIx.TRIMER_NODE]  = 0.0

    # ---------- Lymph ----------
    y0[StateIx.CD3_AB_DIMER_LYMPH] = 0.0
    y0[StateIx.CD20_AB_DIMER_LYMPH] = 0.0
    y0[StateIx.TRIMER_LYMPH]  = 0.0

    # ---------- Activated T cells against B cells (ATC_B) ----------
    y0[StateIx.ATC_B_BLOOD] = 0.0
    y0[StateIx.ATC_B_SPLEEN] = 0.0
    y0[StateIx.ATC_B_NODE]  = 0.0
    y0[StateIx.ATC_B_LYMPH]  = 0.0

    # ----------  Clonally expanded activated T cells against B cells (pATC_B) ----------
    y0[StateIx.PATC_B_BLOOD] = 0.0
    y0[StateIx.PATC_B_SPLEEN] = 0.0
    y0[StateIx.PATC_B_NODE]  = 0.0
    y0[StateIx.PATC_B_LYMPH]  = 0.0

    # ---------- Tumor-specific ATC states ----------
    y0[StateIx.ATC_TUMOR_NODE] = 0.0
    y0[StateIx.PATC_TUMOR_NODE] = 0.0

    # ---------- Tumor baseline ----------
    tum = params.tumor
    r0_cm = tum.Tumor_r0  # cm

    # Same density as in tumor_submodel
    rho = 1e9  # cells / cm^3
    V0_cm3 = (4.0 / 3.0) * np.pi * (r0_cm ** 3)
    N0_cells = rho * V0_cm3

    y0[StateIx.TUMOR_CELLS_TOTAL] = N0_cells

    # ---------- Tumor-specific binding (in tumor node) ----------
    y0[StateIx.CD20_AB_DIMER_TUMOR] = 0.0
    y0[StateIx.TRIMER_TUMOR] = 0.0
    return y0

