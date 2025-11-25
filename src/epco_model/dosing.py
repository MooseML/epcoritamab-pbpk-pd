# # dosing.py
# # Encapsulate dose regimens:

# # Priming doses

# # Weekly maintenance doses

# # Optionally: alternative schedules if you want to explore.

# # @dataclass
# # class DoseEvent:
# #     time: float  # days
# #     amount_mg: float

# # @dataclass
# # class DosingRegimen:
# #     route: str  # "SC"
# #     events: list[DoseEvent]


# # You can then inject doses into the ODE system via the SC compartment, or use event-based updates (e.g., in a time-stepping loop instead of solve_ivp’s continuous approach).

# # dosing.py
# """
# Dosing regimens for epcoritamab.

# For now, we support simple SC bolus dosing. More complex regimens (priming,
# step-up, weekly maintenance) can be added as needed.
# """

# from __future__ import annotations
# import numpy as np 
# from dataclasses import dataclass
# from typing import List


# @dataclass
# class DoseEvent:
#     """
#     Single dosing event.

#     Attributes:
#         time: Time of dose (days)
#         amount: Amount administered (same units as state amounts, e.g. nmol or mg)
#         route: Route of administration ("SC" for now)
#     """
#     time: float
#     amount: float
#     route: str = "SC"


# @dataclass
# class DosingRegimen:
#     """
#     A collection of dosing events.

#     For now, we assume all doses go into the SC depot.
#     """
#     events: List[DoseEvent]


# def single_sc_bolus(dose_amount: float, time: float = 0.0) -> DosingRegimen:
#     """
#     Convenience helper to create a single SC bolus at a given time.

#     Args:
#         dose_amount: Amount to inject into the SC depot (units consistent with model).
#         time: Time of dose (days), default 0.0.

#     Returns:
#         DosingRegimen with one DoseEvent.
#     """
#     return DosingRegimen(events=[DoseEvent(time=time, amount=dose_amount, route="SC")])

# # dosing.py
# MW_EPCORITAMAB_G_PER_MOL = 150_000.0   # ~150 kDa; replace with exact value if given

# def mg_to_nmol(dose_mg: float, mw_g_per_mol: float = MW_EPCORITAMAB_G_PER_MOL) -> float:
#     """
#     Convert a dose in mg to nmol, given the molecular weight (in g/mol).

#         mg  ->  g         : mg * 1e-3
#         g   ->  mol       : g / MW
#         mol ->  nmol      : mol * 1e9
#     """
#     moles = (dose_mg * 1e-3) / mw_g_per_mol
#     nmol = moles * 1e9
#     return nmol


# def single_sc_bolus_mg(
#     dose_mg: float,
#     time: float = 0.0,
# ) -> DosingRegimen:
#     """
#     Single SC bolus defined in mg. Wrapper around your nmol-based regimen.
#     """
#     amount_nmol = mg_to_nmol(dose_mg)
#     return DosingRegimen(events=[DoseEvent(time=time, amount=amount_nmol, route="sc")])

# def weekly_sc_dosing_mg(
#     dose_mg: float,
#     n_weeks: int,
#     start_day: float = 0.0,
# ) -> DosingRegimen:
#     """
#     Simple q1w SC dosing: same dose every 7 days.
#     Used for the Cycle 1–3 weekly simulation in the paper.
#     """
#     amount_nmol = mg_to_nmol(dose_mg)
#     events: List[DoseEvent] = [
#         DoseEvent(time=start_day + 7.0 * k, amount=amount_nmol, route="sc")
#         for k in range(n_weeks)
#     ]
#     return DosingRegimen(events=events)



# def trial_like_cycle1to3_weekly_mg(
#     full_dose_mg: float,
#     priming_fraction: float = 0.1,
#     intermediate_fraction: float = 0.5,
#     n_weeks_total: int = 12,
# ) -> DosingRegimen:
#     """
#     Approximate the clinical schedule for the *simulation* period:

#     - Day 0:   priming dose  = priming_fraction * full_dose
#     - Day 7:   intermediate  = intermediate_fraction * full_dose
#     - Day 14:  full dose
#     - Then:    weekly full dose through ~week 12 (day 84)

#     The actual trial has more complex q2w/q4w spacing in later cycles,
#     but for the model-based dose selection they explicitly say they used
#     weekly dosing in Cycles 1–3. This helper matches that.

#     full_dose_mg:      eg. 48 mg
#     priming_fraction:  eg. 0.1 → 4.8 mg
#     intermediate_fraction: eg. 0.5 → 24 mg
#     """
#     prime_mg = priming_fraction * full_dose_mg
#     interm_mg = intermediate_fraction * full_dose_mg

#     prime_nmol = mg_to_nmol(prime_mg)
#     interm_nmol = mg_to_nmol(interm_mg)
#     full_nmol = mg_to_nmol(full_dose_mg)

#     events: List[DoseEvent] = [
#         DoseEvent(time=0.0,  amount=prime_nmol,  route="sc"),
#         DoseEvent(time=7.0,  amount=interm_nmol, route="sc"),
#         DoseEvent(time=14.0, amount=full_nmol,   route="sc"),
#     ]

#     # Weekly full dosing from day 21 up to day ≈ 7 * n_weeks_total
#     t_end = 7.0 * (n_weeks_total - 1)
#     extra_times = np.arange(21.0, t_end + 0.1, 7.0)
#     for tt in extra_times:
#         events.append(DoseEvent(time=float(tt), amount=full_nmol, route="sc"))

#     return DosingRegimen(events=events)


# def build_epco_regimen(
#     priming_mg: float,
#     intermediate_mg: float,
#     full_mg: float,
#     weekly: bool = True,
# ) -> DosingRegimen:

#     events = []

#     # Cycle 1 step-up
#     day1 = [1, priming_mg]
#     day8 = [8, intermediate_mg]
#     day15 = [15, full_mg]
#     day22 = [22, full_mg]

#     for day, dm in [day1, day8, day15, day22]:
#         events.append(DoseEvent(time=day, amount=mg_to_nmol(dm)))

#     # Expansion assumption: weekly dosing for cycles 2–3 (paper simulation)
#     weekly_days = [29, 36, 43, 50, 57, 64, 71, 78]

#     for d in weekly_days:
#         events.append(DoseEvent(time=d, amount=mg_to_nmol(full_mg)))

#     return DosingRegimen(events)
# dosing.py

# dosing.py

from __future__ import annotations
from dataclasses import dataclass
from typing import List
import numpy as np

# Approximate molecular weight for epcoritamab (IgG1-like)
MW_EPCO_G_PER_MOL = 150_000.0  # 150 kDa; update if exact MW found


# -----------------------------
# Dose event + regimen classes
# -----------------------------

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


# -----------------------------
# Unit conversion
# -----------------------------

def mg_to_nmol(dose_mg: float, mw_g_per_mol: float = MW_EPCO_G_PER_MOL) -> float:
    """Convert dose in mg → nmol."""
    if dose_mg is None:
        raise ValueError(f"dose_mg cannot be None")
    if mw_g_per_mol is None:
        raise ValueError(f"mw_g_per_mol cannot be None")
    if not isinstance(dose_mg, (int, float)):
        raise ValueError(f"dose_mg must be numeric, got {type(dose_mg)}: {dose_mg}")
    if not isinstance(mw_g_per_mol, (int, float)):
        raise ValueError(f"mw_g_per_mol must be numeric, got {type(mw_g_per_mol)}: {mw_g_per_mol}")
    
    dose_g = dose_mg * 1e-3
    mol = dose_g / mw_g_per_mol
    result = mol * 1e9
    if result is None or not isinstance(result, (int, float)):
        raise ValueError(f"mg_to_nmol returned invalid value: {result} (type: {type(result)})")
    return result


# -----------------------------
# Simple regimens
# -----------------------------

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


# -----------------------------------------
# Clinical phase I/II schedule (step-up)
# -----------------------------------------
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
    (this matches the “Cycles 1–3 weekly” assumption used for simulations).
    """
    # Validate inputs
    if full_mg is None:
        raise ValueError(f"full_mg cannot be None")
    if not isinstance(full_mg, (int, float)):
        raise ValueError(f"full_mg must be numeric, got {type(full_mg)}: {full_mg}")
    
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

    events.append(DoseEvent(time=d0,  amount=prime_nmol,  route="SC"))
    events.append(DoseEvent(time=d7,  amount=interm_nmol, route="SC"))
    events.append(DoseEvent(time=d14, amount=full_nmol,   route="SC"))
    events.append(DoseEvent(time=d21, amount=full_nmol,   route="SC"))

    # Weekly full doses after day 21 until t_end
    t = d21 + 7.0
    while t <= t_end + 1e-6:
        events.append(DoseEvent(time=t, amount=full_nmol, route="SC"))
        t += 7.0

    return DosingRegimen(events=events)

