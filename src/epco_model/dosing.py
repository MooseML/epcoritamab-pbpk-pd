# dosing.py

from __future__ import annotations
from dataclasses import dataclass
from typing import List
import numpy as np

# Approximate molecular weight for epcoritamab 
MW_EPCO_G_PER_MOL = 150_000.0  # 150 kDa; update if exact MW given

@dataclass
class DoseEvent:
    time: float          # days
    amount: float        # nmol (goes into DRUG_SC)
    route: str = "SC"    # future extension: "IV"


@dataclass
class DosingRegimen:
    events: List[DoseEvent]

    def times(self) -> List[float]:
        return [e.time for e in self.events]


def mg_to_nmol(dose_mg: float, mw_g_per_mol: float = MW_EPCO_G_PER_MOL) -> float:
    """Convert dose in mg → nmol."""
    dose_g = dose_mg * 1e-3
    mol = dose_g / mw_g_per_mol
    return mol * 1e9


# Simple regimens
def single_sc_bolus(
    dose_amount: float | None = None,
    t0: float = 0.0,
    dose_nmol: float | None = None,
) -> DosingRegimen:
    """
    Single SC bolus at time t0.

    Backwards compatible:
        - old code: single_sc_bolus(dose_amount=..., t0=...)
        - old code: single_sc_bolus(<nmol>)
        - new code: single_sc_bolus(dose_nmol=..., t0=...)
    """
    # Handle positional call: single_sc_bolus(<nmol_value>)
    if dose_amount is not None and dose_nmol is None:
        amount = dose_amount
    # Handle keyword nmol usage
    elif dose_nmol is not None:
        amount = dose_nmol
    else:
        raise ValueError("Provide dose_amount or dose_nmol (in nmol).")

    return DosingRegimen([DoseEvent(time=t0, amount=amount, route="SC")])



def single_sc_bolus_mg(dose_mg: float, t0: float = 0.0) -> DosingRegimen:
    """Single SC bolus in mg."""
    return single_sc_bolus(mg_to_nmol(dose_mg), t0=t0)


def weekly_sc_dosing_mg(
    dose_mg: float,
    n_doses: int,
    start_day: float = 0.0,
    interval_days: float = 7.0,
) -> DosingRegimen:
    """Weekly (or custom interval) SC regimen in mg."""
    dose_nmol = mg_to_nmol(dose_mg)
    events = [
        DoseEvent(time=start_day + i * interval_days, amount=dose_nmol)
        for i in range(n_doses)
    ]
    return DosingRegimen(events)



# Clinical phase I/II schedule (step-up)
def clinical_phaseI_regimen_mg(
    full_mg: float,
    priming_fraction: float = 0.1,
    intermediate_fraction: float = 0.5,
    t_end: float = 84.0,
    first_dose_day: float = 0.0,
    mw_g_per_mol: float = MW_EPCO_G_PER_MOL,
) -> DosingRegimen:
    """
    Phase I/II-like step-up regimen, using *fractions* of the full dose:

    Cycle 1 (relative to first_dose_day):
      Day 0  : priming_fraction * full_mg
      Day 7  : intermediate_fraction * full_mg
      Day 14 : full_mg
      Day 21 : full_mg

    Then weekly full doses until t_end
    (this matches the “Cycles 1-3 weekly” assumption used for simulations).
    """
    events: List[DoseEvent] = []

    prime_mg  = priming_fraction * full_mg
    interm_mg = intermediate_fraction * full_mg

    # Validate calculations didn't produce None
    if prime_mg is None or interm_mg is None:
        raise ValueError(f"Calculated doses are None: prime_mg={prime_mg}, interm_mg={interm_mg}")

    prime_nmol  = mg_to_nmol(prime_mg,  mw_g_per_mol)
    interm_nmol = mg_to_nmol(interm_mg, mw_g_per_mol)
    full_nmol   = mg_to_nmol(full_mg,   mw_g_per_mol)

    # Validate nmol values
    if prime_nmol is None or interm_nmol is None or full_nmol is None:
        raise ValueError(f"mg_to_nmol returned None: prime_nmol={prime_nmol}, interm_nmol={interm_nmol}, full_nmol={full_nmol}")

    d0  = first_dose_day          # trial day 1
    d7  = first_dose_day + 7.0    # trial day 8
    d14 = first_dose_day + 14.0   # trial day 15
    d21 = first_dose_day + 21.0   # trial day 22

    # Validate time values
    if d0 is None or d7 is None or d14 is None or d21 is None:
        raise ValueError(f"Calculated dose times are None: d0={d0}, d7={d7}, d14={d14}, d21={d21}")

    events.append(DoseEvent(time=d0,  amount=prime_nmol, route="SC"))
    events.append(DoseEvent(time=d7,  amount=interm_nmol, route="SC"))
    events.append(DoseEvent(time=d14, amount=full_nmol, route="SC"))
    events.append(DoseEvent(time=d21, amount=full_nmol, route="SC"))

    # Weekly full doses after day 21 until t_end
    t = d21 + 7.0
    while t <= t_end + 1e-6:
        events.append(DoseEvent(time=t, amount=full_nmol, route="SC"))
        t += 7.0

    return DosingRegimen(events=events)

