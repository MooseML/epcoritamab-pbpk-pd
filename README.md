# Epcoritamab PBPK/PD Model

A mechanistic physiologically-based pharmacokinetic/pharmacodynamic (PBPK/PD) model for **epcoritamab**, a CD3xCD20 bispecific T-cell engager (TCE) for the treatment of B-cell malignancies. This implementation replicates and extends the model published in Li et al. (2022).

## Overview

Epcoritamab is a bispecific antibody that simultaneously binds CD3 on T-cells and CD20 on B-cells/tumor cells, forming an immunological synapse that triggers T-cell-mediated cytotoxicity. This model integrates:

- **PK**: Multi-compartment distribution, target-mediated drug disposition
- **Immune trafficking**: T-cell and B-cell trafficking between blood, spleen, lymph node, and lymph compartments
- **T-cell activation**: Trimer-dependent activation and clonal expansion
- **Tumor dynamics**: Logistic tumor growth with T-cell-mediated killing
- **B-cell depletion**: Normal B-cell killing (on-target, off-tumor effect)

## Features

- **Full PBPK/PD implementation** with 43 state variables
- **Population simulations** with inter-patient variability (lognormal sampling)
- **Figure replication** from Li et al. (2022) - Figures 2-7
- **Parallel execution** for multi-patient Monte Carlo simulations
- **Dose-adaptive solver settings** for numerical stability at high doses
- **Clinical dosing regimens** (step-up priming/intermediate/full doses)

## Repository Structure

```
epcoritamab-pbpk-pd/
├── src/epco_model/              # Core model implementation
│   ├── parameters.py            # Model parameters and inter-patient variability
│   ├── state_vector.py          # State variable definitions (43 states)
│   ├── pk_submodel.py           # Multi-compartment PK with TMDD
│   ├── trafficking_submodel.py  # Lymphocyte trafficking and homeostasis
│   ├── binding_submodel.py      # CD3/CD20 binding and trimer formation
│   ├── tcell_activation_submodel.py  # T-cell activation and expansion
│   ├── bcell_kill_submodel.py   # B-cell depletion kinetics
│   ├── tumor_submodel.py        # Tumor growth and T-cell-mediated kill
│   ├── odes.py                  # ODE system integration
│   ├── dosing.py                # Clinical dosing regimens
│   ├── simulation.py            # Single-patient simulation engine
│   ├── monte_carlo.py           # Population simulation framework
│   └── plotting.py              # Visualization utilities
│
├── notebooks/                   # Jupyter notebooks (tracked in git)
│   ├── 01_single_patient_baseline.ipynb  # Model validation
│   ├── 02_dose_response_ORR.ipynb        # Dose-response analysis
│   └── 03_figure_replication.ipynb       # Replicate Figures 2-7
│
├── README.md                    # This file
└── .gitignore                   # Git ignore rules

# Git-ignored directories (not tracked):
├── diagnostics/                 # Debugging scripts and diagnostics
├── examples/                    # Standalone figure scripts
├── original_paper/              # Reference paper PDFs
├── docs/                        # Development notes
├── scripts/                     # Utility scripts
├── tests/                       # Test suite (to be developed)
├── configs/                     # Configuration files (to be developed)
└── data/                        # Simulation outputs (mostly empty)
```

## Installation

### Prerequisites
- Python 3.11+
- Conda or pip 

### Setup

```bash
# Clone the repository
git clone https://github.com/yourusername/epcoritamab-pbpk-pd.git
cd epcoritamab-pbpk-pd

# Create conda environment (recommended)
conda create -n epco python=3.11
conda activate epco

# Install dependencies
pip install numpy scipy matplotlib pandas joblib tqdm

```

## Notebooks

### 01: Single Patient Baseline Simulations
Validates core PBPK/PD components through baseline simulations:
- PK sanity check (single SC bolus)
- Immune trafficking homeostasis
- Tumor growth dynamics
- Full system integration

### 02: Dose-Response Analysis (ORR)
Characterizes dose-response relationship for Overall Response Rate (ORR):
- 8 dose levels (0.0128 - 60 mg)
- 128 virtual patients per dose
- Response criterion: >30% tumor reduction
- ~10-15 minutes runtime

### 03: Figure Replication
Replicates key figures from Li et al. (2022):
- **Figure 2**: Exposure-Response (ORR vs C_avg)
- **Figure 3**: Tumor Diameter Change from Baseline
- **Figure 4**: B-cell Depletion
- **Figure 5**: T-cell Dynamics (Panels a & b)
- **Figure 6**: Exposure-Safety Analysis (CRS)
- **Figure 7**: Trimer Formation and ORR by Tumor Type

**Note**: Figure 7 has a quick version (6 doses, ~4 min) and full version (15 doses, ~3-5 hours)

## Model Components

### State Variables (43 total)
- **PK (8)**: Drug amounts in SC depot, plasma, tight tissues, leaky tissues, spleen, lymph node, lymph
- **B-cells (4)**: Blood, spleen, lymph node, lymph
- **T-cells (4)**: Blood, spleen, lymph node, lymph
- **Activated T-cells (8)**: Against B-cells and tumors in multiple compartments
- **Binding (12)**: CD3/CD20 binding, dimers, trimers on B-cells and tumor
- **Tumor (2)**: Total tumor cells, tumor receptor occupancy
- **Other (5)**: Injection effect, homeostatic feedback

### Key Parameters
- **PK**: V_plasma = 2.6 L, CL = 2.47 L/day, k_a = 0.131 day⁻¹
- **Lymphocyte distribution**: 2% blood, 60% spleen, 38% lymph (Westermann & Pabst 1992)
- **T-cell activation**: Trimer threshold = 24 molecules/cell, expansion rate = 9.27 day⁻¹
- **Tumor kill**: k_kill = 0.165 L²/(10⁶ cells·day)
- **Tumor growth**: k_growth (DLBCL) = 0.0301 day⁻¹, k_growth (FL) = 0.0038 day⁻¹
- **Tumor density**: 1×10⁹ cells/cm³ (from Li et al. 2022, standard oncology assumption)

### Inter-Patient Variability
Lognormal sampling with CV% from Li et al. (2022):
- PK parameters: 77-140% CV
- T-cell production: 45-48% CV  
- Activation/expansion: 20-58% CV
- Tumor parameters: 46-100% CV

## Figure Replication Results

### Figure 2: Exposure-Response
**Status**: Excellent match
- Emax model with E_max = 76.2%, EC50 = 0.788 nM
- 5th/50th/95th percentile curves with 90% CIs
- Demonstrates dose-dependent ORR saturation

### Figure 3: Tumor Diameter Change
**Status**: Qualitative match, quantitative differences
- **Issue**: Model shows tumor responses only at ≥12 mg
- **Paper shows**: Responses at ≥0.12 mg
- **Root cause**: Insufficient trimer formation at low doses (<1 trimers/cell vs 24 threshold)
- **Implication**: Model parameters may need recalibration for low-dose tumor kill

### Figure 4: B-cell Depletion
**Status**: Good match
- Dose-dependent B-cell depletion over 28 days
- Median and 90% CI bands across 12 doses
- Demonstrates on-target, off-tumor effects

### Figure 5: T-cell Dynamics
**Status**: Excellent match after bug fixes
- **Panel (a)**: Cycle 1 (0-28 days) shows injection-induced oscillations
- **Panel (b)**: Cycles 1-6 (0-168 days) shows dose-dependent expansion
- **Key fix**: Added `INJ` compartment update at each dose for transient redistribution
- **Key fix**: Corrected T-cell count to include activated + clonally expanded populations

### Figure 6: Exposure-Safety (CRS)
**Status**: Good match
- **Key finding**: Flat relationship between C_max and CRS risk within dose intervals
- **Risk varies by interval**: Priming (18-20%) < Intermediate (34-36%) > First full (30-33%) > Second full (16-20%)
- CIs widen at high C_max due to extrapolation uncertainty

### Figure 7: Trimer Formation and ORR
**Status**: Qualitative match
- DLBCL shows higher ORR than FL due to faster growth (more target cells)
- Trimer formation increases with dose (0.001 - 1000 molecules/cell)
- **Full version runtime**: 3-5 hours (15 doses, 100 patients, 100 trials)

## Key Insights from Modeling

### 1. T-cell Dynamics are Dose-Dependent
- **Low doses (<1 mg)**: Minimal T-cell expansion, insufficient for tumor kill
- **Therapeutic doses (12-60 mg)**: 5-10× T-cell expansion, robust tumor kill
- **Mechanism**: Trimer threshold (24 molecules/cell) creates nonlinear dose-response

### 2. Injection Effects Drive Short-Term Oscillations
- Each SC injection causes transient T-cell redistribution (INJ compartment)
- Oscillations visible in Figure 5a but dampen over time (k_decay = 0.681 day⁻¹)
- Critical for matching clinical T-cell kinetics post-dose

### 3. CRS Risk is Dose-Interval Dependent, Not C_max Dependent
- **Priming dose**: Lower risk (~20%) despite variable C_max
- **Intermediate/First full**: Higher risk (~33%) - peak T-cell activation
- **Second full**: Lower risk (~18%) - immune tolerance or B-cell depletion
- Flat C_max relationship suggests T-cell memory/exhaustion rather than acute drug exposure drives CRS

### 4. Tumor Type Affects Efficacy via Growth Rate
- **DLBCL** (k_growth = 0.0301 day⁻¹): Higher baseline tumor burden → more targets → higher ORR
- **FL** (k_growth = 0.0038 day⁻¹): Indolent growth → fewer accessible targets → lower ORR
- Highlights importance of tumor accessibility beyond just drug exposure

### 5. Low-Dose Tumor Kill Requires Model Calibration
- Current parameters underestimate low-dose efficacy (0.12-0.38 mg range)
- Possible adjustments:
  - **Lower trimer threshold** (24 → 5-10 molecules/cell)
  - **Higher tumor kill rate** (k_kill_tumor)
  - **Lower activation threshold** (sim_slopetumor)
- Future work: Fit to patient-level tumor response data

### 6. Numerical Stability Requires Dose-Adaptive Settings
- **Low doses** (<3 mg): Standard tolerances (rtol=1e-4, atol=1e-7)
- **High doses** (>24 mg): Relaxed tolerances + smaller max_step to avoid stiffness
- Dynamic timeout scaling prevents hangs at extreme doses

## Known Limitations

1. **Low-dose tumor responses**: Model requires higher doses than clinical data suggests for tumor kill
2. **Parameter uncertainty**: Some parameters are literature-derived, not fitted to epcoritamab data
3. **CRS model**: Logistic regression parameters are illustrative, not published values
4. **Single tumor**: Model assumes one representative tumor lesion, not multi-site disease
5. **No resistance mechanisms**: Model assumes constant drug sensitivity over time

## Future Directions

- **Parameter calibration**: Fit tumor kill parameters to patient-level response data
- **Resistance mechanisms**: Add PD-L1/exhaustion dynamics for long-term simulations
- **Combination therapy**: Extend to CD20+CD3 + checkpoint inhibitors
- **Spatial heterogeneity**: Multi-compartment tumor with penetration limits
- **Test suite**: Comprehensive unit and integration tests

## References

**Primary Reference:**
- Li C, et al. (2022). "Semimechanistic Physiologically-Based Pharmacokinetic/Pharmacodynamic Model of Epcoritamab in Patients with Relapsed or Refractory B-Cell Non-Hodgkin Lymphoma." *Clinical Pharmacology & Therapeutics*, 112(6), 1294-1308. https://doi.org/10.1002/cpt.2729

**Key Supporting References:**
- Westermann J, Pabst R. (1992). "Distribution of lymphocyte subsets and natural killer cells in the human body." *Clin Investig*, 70, 539-544. (Source of "2% in blood" lymphocyte distribution)


## License

This is an academic/research implementation based on published literature. For commercial use, please consult the original paper and relevant intellectual property holders.

## Citation

If you use this code in your research, please cite:

```bibtex
@article{li2022epcoritamab,
  title={Semimechanistic Physiologically-Based Pharmacokinetic/Pharmacodynamic Model of Epcoritamab in Patients with Relapsed or Refractory B-Cell Non-Hodgkin Lymphoma},
  author={Li, Chao and others},
  journal={Clinical Pharmacology \& Therapeutics},
  volume={112},
  number={6},
  pages={1294--1308},
  year={2022},
  publisher={Wiley Online Library}
}
```

---

**Last Updated**: November 2025
