# parameters.py
from dataclasses import dataclass, field
from typing import Dict
import copy
import math
import numpy as np

@dataclass
class PKSubmodelParameters:
    ISF: float = 15.6 # (L) Volume of ISF
    Kp: float = 0.8 # Available fraction of ISF for native IgG1 distribution
    Vplasma: float = 2.6 # (L) Volume of plasma
    Vtight: float = 8.11 # (L) Interstitial volume of tight tissue compartment 
    Vleaky: float = 4.37 # (L) Interstitial volume of leaky tissue compartment 
    Vspleen: float = 0.0433 # (L) Interstitial volume of spleen compartment
    Vnode: float = 0.0082 # (L) Volume of lymph node compartment
    Vlymph: float = 5.2 # (L) Volume of lymph compartment
    Ltight: float = 0.957 # (L/d) Lymph flow through tight tissue compartment
    Lleaky: float = 1.94 # (L/d) Lymph flow through leaky tissue compartment
    Lspleen: float = 0.304 # (L/d) Lymph flow through spleen
    L: float = 2.9 # (L/d) Lymph flow through lymph compartment
    sigma_tight: float = 0.95 # Leakiness of tight tissue compartment 
    sigma_leaky: float = 0.85 # Leakiness of leaky tissue compartment 
    sigma_lymph: float = 0.2 # Leakiness of lymph tissue compartment 
    ka: float = 0.131 # (1/day) Epcoritamab absorption rate
    CL: float = 2.47 # (L/d) Epcoritamab clearance
    Km: float = 0.0461 # (nM/L) Epcoritamab concentration that leads to 50% saturation of nonlinear clearance
    Vmax: float = 0.185 # (nM/d) Maximum nonlinear clearance rate of epcoritamab


@dataclass
class LymphocyteTraffickingParameters:
    Vblood: float = 4.73 # (L) Volume of blood
    Vspleen_tissue: float = 0.221 # (L) Volume of spleen tissue
    TCplasma_base: float = 1469 # (cells/mm^3) # Baseline of T-cell count in blood
    kinTC: float = 310e6 # (cells/d) Production rate of T cells
    koutTC: float = 0.00422 # (1/day) Natural death rate of T cells
    BCplasma_base: float = 236 # (cells/mm^3) Baseline B-cell count in blood 
    kinBC: float = 223e6 # (cells/d) Production rate of B cells
    koutBC: float = 0.0189 # (1/day) Natural death rate of B cells
    kpt: float = 50 # (1/day) Describes trafficking of lymphocytes from blood/plasma to spleen tissue
    ktn: float = 1.67 # (1/day) Describes trafficking of lymphocytes from spleen tissue to lymph node
    knl: float = 1672 # (1/day) Describes trafficking of lymphocytes from lymph node to lymph
    klp: float = 2.63 # (1/day) Describes trafficking of lymphocytes from lymph to blood/plasma
    kdecay: float = 0.681 # (1/day) Duration of injection effect on lymphocyte trafficking
    INJ_Scaler: float = 9.08 # (1/item) Magnitude of injection effect on lymphocyte trafficking
    kt: float = 0.00027 # (1/day) Lymphocyte homeostasis control rate
    r: float = 2.24 # Strength of lymphocyte homeostasis control


@dataclass
class EpcoritamabBindingParameters:
    konCD3: float = 18.1 # (L/nmol/d) Epcoritamab association rate constant to CD3
    koffCD3: float = 285 # (1/day) Epcoritamab dissociation rate constant to CD3
    konCD20: float = 4.15 # (L/nmol/d) Epcoritamab association rate constant to CD20
    koffCD20: float = 22.5 # (1/day) Epcoritamab dissociation rate constant to CD20
    kdegCD3: float = 1.584 # (1/day) Natural degradation rate of CD3
    kdegCD20: float = 1.584 # (1/day) Natural degradation rate of CD20
    kintCD3: float = 1.584 # (1/day) Internalization rate of epcoritamab–CD3 dimer
    kintCD20: float = 1.584 # (1/day) Internalization rate of epcoritamab–CD20 dimer
    RCD3: float = 30000 # (receptors/cell) Level of CD3 expression on T cells
    RCD20: float = 100000 # (molecules/cell) Level of CD20 expression on B cells
    RCD20tumor: float = 30000 # (molecules/cell) Level of CD20 expression on tumor cells

@dataclass
class TCellActivationParameters:
    sim_slope: float = 0.007 # ((10^6 cells/day) * (cell/molecule)) Rate of T-cell activation against B cells
    sim_slopetumor: float = 7e-6 # ((10^6 cells/day) * (cell/molecule)) Rate of T-cell activation against tumor cells
    Trimer_Threshold: float = 24 # (molecules/cell) Trimer formation threshold leading to 100% of first-order T-cell activation
    expand_factor: float = 9.27 # (1/day) First-order rate constant describing clonal expansion of activated T cells
    koutATC: float = 0.05 # (1/day) Describes elimination of activated and clonally expanded T cells 
    TAD: float = 3 # (day) T-cell activation delay after first dose
    Tp: float = 10.8 # (day) Lag time between the first dose of epcoritamab and the first natural death of activated T cells 


@dataclass
class BCellKillingParameters:
    kkill_BC: float = 0.544 # (L^2 / (10^6 cells · day)) Rate of B-cell killing by activated T cells 


@dataclass
class TumorKillingParameters:
    kgrowth_FL: float = 0.0038 # (1/day) Rate of growth for FL
    kgrowth_DLBCL: float = 0.0301 # (1/day) Rate of growth for DLBCL
    Tumor_r0: float = 1.40 # (cm) Radius of tumor at baseline
    tumor_capacity: float = 1e-6 # (1 / 10^6 cells) Describes carrying capacity of the tumor 
    depth: float = 0.01 # (cm) Level of tumor-cell accessibility to activated T cells 
    kkill_tumor: float = 0.165 # (L^2 / (10^6 cells · day)) Rate of tumor-cell killing by activated T cells 


@dataclass
class ModelParameters:
    pk: PKSubmodelParameters = field(default_factory=PKSubmodelParameters)
    trafficking: LymphocyteTraffickingParameters = field(default_factory=LymphocyteTraffickingParameters)
    binding: EpcoritamabBindingParameters = field(default_factory=EpcoritamabBindingParameters)
    activation: TCellActivationParameters = field(default_factory=TCellActivationParameters)
    bkill: BCellKillingParameters = field(default_factory=BCellKillingParameters)
    tumor: TumorKillingParameters = field(default_factory=TumorKillingParameters)



def get_default_parameters() -> ModelParameters:
    """
    Return a ModelParameters object with all default (typical) values.
    """
    return ModelParameters()


# CV% values from Table S2 (only those w/ variability)
PARAMETER_CV: Dict[tuple[str, str], float] = {
    # PK
    ("pk", "ka"): 95.0,
    ("pk", "CL"): 77.1,
    ("pk", "Vmax"): 131.0,
    ("pk", "Km"): 140.0,

    # Trafficking / production
    ("trafficking", "kinTC"): 44.7,
    ("trafficking", "kinBC"): 46.4,

    # Activation / ATC
    ("activation", "koutATC"): 22.4,
    ("activation", "sim_slope"): 57.9,
    ("activation", "sim_slopetumor"): 57.9,
    ("activation", "expand_factor"): 48.6,
    ("activation", "Tp"): 20.0,

    # Killing & tumor & receptors
    ("bkill", "kkill_BC"): 50.0,
    ("tumor", "kkill_tumor"): 50.0,
    ("trafficking", "kdecay"): 50.0,
    ("trafficking", "INJ_Scaler"): 50.0,
    ("tumor", "kgrowth_FL"): 45.9, # or kgrowth_DLBCL depending on tumor type
    ("tumor", "kgrowth_DLBCL"): 45.9,
    ("tumor", "Tumor_r0"): 58.7,
    ("binding", "RCD3"): 44.7,
    ("binding", "RCD20"): 44.7,
    ("binding", "RCD20tumor"): 100.0,
}

def _lognormal_from_cv(mean: float, cv_percent: float, rng: np.random.Generator) -> float:
    """
    Sample a lognormal random variable with given mean and CV%.
    """
    if cv_percent <= 0:
        return mean
    cv = cv_percent / 100.0
    sigma = math.sqrt(math.log(1.0 + cv**2))
    eta = rng.normal(0.0, sigma)
    return mean * math.exp(eta)

def sample_virtual_patient(
    base: ModelParameters | None = None,
    rng: np.random.Generator | None = None,
) -> ModelParameters:
    """
    Apply lognormal inter-individual variability (Table S2) to the base parameters
    and return a new ModelParameters instance representing one virtual patient.
    """
    if base is None:
        base = get_default_parameters()
    if rng is None:
        rng = np.random.default_rng()

    params = copy.deepcopy(base)

    for (block_name, field_name), cv in PARAMETER_CV.items():
        block = getattr(params, block_name)
        mean_val = getattr(block, field_name)
        new_val = _lognormal_from_cv(mean_val, cv, rng)
        setattr(block, field_name, new_val)

    return params
