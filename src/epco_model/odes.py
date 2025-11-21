# odes.py

# odes.py
"""
Global ODE system for the epcoritamab PBPK/PD model.

dy/dt = f(t, y, params)

This module orchestrates contributions from each submodel:
- PK (drug distribution and clearance)
- Lymphocyte trafficking
- Binding / trimer formation
- T-cell activation and proliferation
- B-cell killing
- Tumor growth and killing

Initially, we implement PK only and add the others incrementally.
"""

from __future__ import annotations

import numpy as np

from .parameters import ModelParameters
from .state_vector import N_STATES
from .pk_submodel import update_dydt_pk
from .trafficking_submodel import update_dydt_trafficking
from .binding_submodel import update_dydt_binding
from .tcell_activation_submodel import update_dydt_tcell_activation
# from .bcell_kill_submodel import update_dydt_bcell_kill
# from .tumor_submodel import update_dydt_tumor


def rhs(t: float, y: np.ndarray, params: ModelParameters) -> np.ndarray:
    """
    Full right-hand side of the ODE system dy/dt = f(t, y, params).

    Args:
        t: Time (days)
        y: State vector [N_STATES]
        params: ModelParameters instance

    Returns:
        dydt: Derivative vector [N_STATES]
    """
    dydt = np.zeros_like(y)

    # PK submodel: drug disposition
    update_dydt_pk(t, y, params, dydt)
    update_dydt_trafficking(t, y, params, dydt)
    update_dydt_binding(t, y, params, dydt)
    update_dydt_tcell_activation(t, y, params, dydt)
    # TODO: uncomment/add these as you implement the corresponding modules
    # update_dydt_bcell_kill(t, y, params, dydt)
    # update_dydt_tumor(t, y, params, dydt)

    return dydt

