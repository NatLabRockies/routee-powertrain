"""Physical plausibility checks for a trained energy model.

Where ``validation.errors`` measures accuracy against held-out data, this module
asks a different question: is the learned function *physically possible*, and is
it predicting plausible real-world energy? Both are answered from a synthetic
sweep of links, so no ground truth is needed and any model can be checked at any
time.

Two kinds of statement are produced:

- **Checks** — pass/fail predicates. The first five need no vehicle knowledge at
  all, so a failure is unambiguous. The rest need the vehicle mass and report
  ``not_applicable`` when it is unknown.
- **Diagnostics** — descriptive numbers with no pass/fail: implied drivetrain and
  regeneration efficiency, flat-ground fuel economy, the grade and speed range the
  model actually responds over, and how stable its per-mile rate is with link
  length.

Predictions are scored *before* the real-world adjustment factor and *before* any
output guardrail, because the point is to measure the function that was learned.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Callable, Dict, List, NamedTuple, Optional, Tuple

import numpy as np
import pandas as pd
from pydantic import BaseModel, Field

from routee.powertrain.core.features import DataColumn
from routee.powertrain.core.model_config import ModelConfig
from routee.powertrain.core.powertrain_type import PowertrainType
from routee.powertrain.estimators.estimator_interface import Estimator

if TYPE_CHECKING:
    from routee.powertrain.core.model import Model

log = logging.getLogger(__name__)

# -- physical constants -----------------------------------------------------

G = 9.80665
"""Standard gravity, m/s^2."""

LB_TO_KG = 0.45359237
MI_TO_M = 1609.344
MPH_TO_MS = 0.44704
KPH_TO_MPH = 0.621371
AIR_DENSITY = 1.225
"""Sea-level air density, kg/m^3."""

#: Joules of energy in one unit of each recognized target, used to compare a
#: prediction against mechanical work. Electric targets are exact; fuel targets
#: use the lower heating value.
ENERGY_CONTENT_J: Dict[str, float] = {
    "kilowatt-hour": 3.6e6,
    "kilowatt hour": 3.6e6,
    "kwh": 3.6e6,
    "kwhs": 3.6e6,
    "kwhs electricity": 3.6e6,
    "gallons gasoline": 1.2132e8,
    "gallons of gasoline": 1.2132e8,
    "gge": 1.2132e8,
    "gallons diesel": 1.3662e8,
    "gallons of diesel": 1.3662e8,
    "gde": 1.3662e8,
    "kilograms of hydrogen": 1.20e8,
    "kg h2": 1.20e8,
}

#: Target units whose value can never be negative — burned fuel does not go back
#: into the tank. Electric targets are absent because regeneration is real.
COMBUSTION_UNITS = {
    "gallons gasoline",
    "gallons of gasoline",
    "gge",
    "gallons diesel",
    "gallons of diesel",
    "gde",
    "kilograms of hydrogen",
    "kg h2",
}

#: Neutral values for features that carry no physical role, keyed by normalized
#: units — the value that describes an ordinary, featureless link. Preferred
#: over the midpoint of a feature's declared range, because a midpoint is only
#: neutral for a range centered on the benign value: the midpoint of a
#: ``0..180`` bearing delta is a 90-degree turn on every link, not a straight one.
NEUTRAL_BY_UNITS: Dict[str, float] = {
    "degrees fahrenheit": 70.0,
    "degrees celsius": 21.0,
    # Turn angle and bearing delta: straight ahead.
    "degrees": 0.0,
    # Sinuosity and similar shape ratios: a link with no curvature is 1.0.
    "ratio": 1.0,
    "category": 0.0,
}

#: Seconds per unit for the recognized time units. A feature carrying link
#: traversal time is *derived* from the sweep rather than held, since time is
#: fixed by distance and speed.
TIME_UNITS: Dict[str, float] = {
    "seconds": 1.0,
    "second": 1.0,
    "sec": 1.0,
    "s": 1.0,
    "minutes": 60.0,
    "minute": 60.0,
    "min": 60.0,
    "hours": 3600.0,
    "hour": 3600.0,
    "hr": 3600.0,
    "h": 3600.0,
}

SPEED_UNITS = {"mph", "miles per hour", "mi/h", "mi per hour"}
SPEED_UNITS_KPH = {"kph", "km/h", "kmh", "kilometers per hour"}
GRADE_UNITS_PERCENT = {"percent", "%", "pct"}
GRADE_UNITS_DECIMAL = {"decimal", "fraction", "ratio", "unitless", "dimensionless"}
MASS_UNITS = {"pounds", "lbs", "lb", "pound"}
DISTANCE_UNITS_MILES = {"miles", "mile", "mi"}

#: Plausible flat-ground economy envelopes by powertrain type. Deliberately wide
#: — a model outside these is obviously wrong, not marginally wrong. Electric
#: entries are mi/kWh; fuel entries are mpg (gasoline-equivalent).
ECONOMY_BANDS: Dict[PowertrainType, Tuple[float, float]] = {
    PowertrainType.BEV: (1.5, 8.0),
    PowertrainType.PHEV_EV_MODE: (1.5, 8.0),
    PowertrainType.ICE: (8.0, 60.0),
    PowertrainType.HEV: (20.0, 80.0),
    PowertrainType.PHEV_HEV_MODE: (20.0, 80.0),
    PowertrainType.HEAVY_DUTY: (2.0, 20.0),
}

# Sweep geometry.
SPEEDS_MPH = np.array([5.0, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70])
GRADES_PCT = np.array([0.5, 1.0, 2.0, 3.0, 4.0, 6.0, 8.0, 10.0])
DISTANCES_MI = np.array([0.05, 0.1, 0.25, 0.5, 1.0])
TRIP_LEN = 12
"""Links per synthetic trip for a lookback model; the last one is scored."""

#: Shallow-grade band used to fit implied efficiencies. Steep grades saturate in
#: most forests, which would bias the slope.
SHALLOW_GRADE_PCT = 3.0

TOL = 1e-12


def is_combustion_target(units: str, powertrain_type: PowertrainType) -> bool:
    """Whether a target counts burned fuel, which never comes back.

    Decided by the normalized units, except for an electric vehicle: a BEV
    whose target is in gasoline-gallon equivalents is still storing
    electricity, and its regeneration is real.
    """
    if powertrain_type in (PowertrainType.BEV, PowertrainType.PHEV_EV_MODE):
        return False
    return units in COMBUSTION_UNITS


def normalize_units(units: Optional[str]) -> str:
    """Collapse a units string to a comparable form.

    The library spells the same unit several ways (``gallons gasoline`` and
    ``gallons_gasoline``, ``kilowatt-hour`` and ``kwh``), so every lookup goes
    through here first.
    """
    if not units:
        return ""
    out = units.strip().lower()
    for ch in ("_", "-"):
        out = out.replace(ch, " ")
    return " ".join(out.split())


class PhysicsAssumptions(BaseModel):
    """Vehicle constants used by the checks that need more than mass.

    Every default is deliberately generous — a high rolling resistance and drag
    area raise the ceiling, a low drivetrain efficiency raises it further, and a
    high regeneration efficiency lowers the floor. A violation against these
    numbers is a violation against any plausible vehicle.
    """

    #: Coefficient of rolling resistance.
    crr: float = 0.020
    #: Drag area (Cd * frontal area), m^2.
    cda_m2: float = 1.50
    #: Air density, kg/m^3.
    air_density: float = AIR_DENSITY
    #: Battery-to-wheel efficiency, used for electric targets. Below any real
    #: driveline, because the ceiling exists to catch divergence rather than to
    #: audit efficiency, and a tight ceiling flags correct models.
    eta_drive: float = 0.50
    #: Fraction of braking energy a vehicle can return to storage.
    eta_regen: float = 0.75
    #: Tank-to-wheel efficiency, used for combustion targets. A peak-efficiency
    #: figure would be about 0.25, but a link averaging a given speed is
    #: stop-and-go rather than steady cruise, so real part-load efficiency is far
    #: lower. Applying the electric figure here produces false failures.
    eta_tank_to_wheel: float = 0.12
    #: Accessory and idle draw in watts of stored energy, spent per unit time
    #: rather than per unit distance. Without it the ceiling collapses at low
    #: speed, where a link takes many minutes and the vehicle is running the
    #: whole time. Generous: roughly a third of a gallon per hour idling.
    accessory_watts_combustion: float = 12000.0
    accessory_watts_electric: float = 3000.0


class PhysicsCheck(BaseModel):
    """The outcome of one physical predicate over the synthetic sweep."""

    name: str
    #: ``"pass"``, ``"fail"``, or ``"not_applicable"``.
    status: str
    #: Why a check did not run, or a one-line summary of what failed.
    reason: Optional[str] = None
    n_tested: int = 0
    n_violations: int = 0
    #: Fraction of tested cases violating the predicate, on raw estimator output.
    violation_rate: Optional[float] = None
    #: Same fraction, on output multiplied by the real-world adjustment factor —
    #: what a routing consumer actually receives.
    adjusted_violation_rate: Optional[float] = None
    #: Fraction of tested cases on which the corresponding output guardrail would
    #: have had to fire. Identical to ``violation_rate`` for a bound-shaped check;
    #: recorded separately because it stays informative once clamping ships, when
    #: the violation rate is zero by construction.
    would_bind_rate: Optional[float] = None
    #: Size of the worst violation, in target units.
    worst_margin: Optional[float] = None
    #: The link that produced the worst violation.
    worst_case: Optional[Dict[str, float]] = None

    @property
    def failed(self) -> bool:
        return self.status == "fail"


class PhysicsDiagnostics(BaseModel):
    """Descriptive measurements with no pass/fail — how the model behaves."""

    #: Flat-ground economy per speed. ``mi/kWh`` for electric targets, ``mpg`` for
    #: fuel targets. Keyed by speed in mph.
    flat_economy: Dict[str, float] = Field(default_factory=dict)
    #: The plausible envelope the economy figures are read against.
    economy_band: Optional[Tuple[float, float]] = None
    #: Speeds whose flat-ground economy falls outside that envelope.
    economy_outliers: List[float] = Field(default_factory=list)

    #: Implied battery/tank-to-wheel efficiency, from the climb-side slope over
    #: the shallow-grade band. Needs mass.
    implied_eta_drive: Optional[float] = None
    #: Implied regeneration efficiency, from the descent-side slope. A value
    #: above 1.0 is energy from nowhere.
    implied_eta_regen: Optional[float] = None
    #: True when ``implied_eta_regen > 1``.
    regen_exceeds_unity: bool = False

    #: Largest relative spread of the per-mile energy rate across link lengths.
    #: Zero for a RATE model by construction; a large value means predictions
    #: depend on how a route was segmented.
    length_invariance: Optional[float] = None

    #: Features held fixed during the sweep, and the value each was held at.
    held_features: Dict[str, float] = Field(default_factory=dict)
    #: Features computed from the sweep rather than held, because they are fixed
    #: by the quantities being swept (e.g. link traversal time).
    derived_features: List[str] = Field(default_factory=list)


class PhysicsReport(BaseModel):
    """Everything the physical validation found for one model."""

    checks: List[PhysicsCheck] = Field(default_factory=list)
    diagnostics: PhysicsDiagnostics = Field(default_factory=PhysicsDiagnostics)
    assumptions: PhysicsAssumptions = Field(default_factory=PhysicsAssumptions)
    #: The target the checks were run against.
    target: Optional[str] = None
    #: Vehicle mass used, in kg, and where it came from.
    mass_kg: Optional[float] = None
    mass_source: Optional[str] = None
    #: Anything a reader needs to interpret the result — a guessed hold value, a
    #: skipped check, an unrecognized unit.
    notes: List[str] = Field(default_factory=list)

    @property
    def passed(self) -> bool:
        """True when no applicable check failed."""
        return not any(c.failed for c in self.checks)

    @property
    def check_map(self) -> Dict[str, PhysicsCheck]:
        return {c.name: c for c in self.checks}

    def __repr__(self) -> str:
        lines = ["=" * 62]
        verdict = "PASS" if self.passed else "FAIL"
        lines.append(f"Physics validation: {verdict}   (target: {self.target})")
        lines.append("=" * 62)
        for c in self.checks:
            rate = "" if c.violation_rate is None else f"{100 * c.violation_rate:6.1f}%"
            lines.append(f"{c.name:<24} {c.status:<16} {rate:>8}")
            if c.status == "not_applicable" and c.reason:
                lines.append(f"{'':<24} {c.reason}")
            elif c.failed and c.worst_case:
                where = ", ".join(f"{k}={v:g}" for k, v in c.worst_case.items())
                lines.append(f"{'':<24} worst at {where}")
        d = self.diagnostics
        lines.append("-" * 62)
        if d.implied_eta_regen is not None:
            flag = "  <- energy from nowhere" if d.regen_exceeds_unity else ""
            lines.append(f"{'implied eta_regen':<24} {d.implied_eta_regen:.3f}{flag}")
        if d.implied_eta_drive is not None:
            lines.append(f"{'implied eta_drive':<24} {d.implied_eta_drive:.3f}")
        if d.economy_outliers:
            speeds = ", ".join(f"{s:g}" for s in d.economy_outliers)
            lines.append(f"{'economy out of band at':<24} {speeds} mph")
        if d.length_invariance is not None:
            lines.append(f"{'length invariance':<24} {d.length_invariance:.4f}")
        for note in self.notes:
            lines.append(f"note: {note}")
        lines.append("=" * 62)
        return "\n".join(lines)

    def _repr_html_(self) -> str:
        rows = ["<table border='1' style='border-collapse: collapse;'>"]
        verdict = "PASS" if self.passed else "FAIL"
        rows.append(
            "<tr><td colspan='4' style='border-bottom: 2px solid black;"
            f"text-align: center;'><b>Physics validation: {verdict}</b></td></tr>"
        )
        rows.append("<tr><th>Check</th><th>Status</th><th>Rate</th><th>Note</th></tr>")
        for c in self.checks:
            rate = "" if c.violation_rate is None else f"{100 * c.violation_rate:.1f}%"
            rows.append(
                f"<tr><td>{c.name}</td><td>{c.status}</td>"
                f"<td>{rate}</td><td>{c.reason or ''}</td></tr>"
            )
        rows.append("</table>")
        return "".join(rows)


# -- column roles -----------------------------------------------------------


class _Roles:
    """Which feature plays which physical role in a model's input frame.

    ``speed_scale`` multiplies a model value to give mph; ``grade_scale``
    multiplies a percent grade to give the model's value. ``_build_frame``
    applies both in the direction that builds a model frame, and
    ``physical_bounds`` in the direction that reads one.
    """

    def __init__(self) -> None:
        self.speed: Optional[DataColumn] = None
        self.speed_scale: float = 1.0
        self.grade: Optional[DataColumn] = None
        self.grade_scale: float = 1.0
        self.mass: Optional[DataColumn] = None
        self.time: Optional[DataColumn] = None
        #: Seconds per unit of the time column.
        self.time_scale: float = 1.0
        self.held: Dict[str, float] = {}


def _resolve_roles(
    config: ModelConfig,
    reference_df: Optional[pd.DataFrame],
    hold_values: Optional[Dict[str, float]],
    notes: List[str],
) -> _Roles:
    """Classify each feature by the physical quantity it carries.

    Resolution is by units, because the library spells the same quantity many
    ways (``grade_percent``, ``grade_pct``, ``grade_dec``, ``grade``). Where the
    units alone are ambiguous — ``decimal`` could be anything — the column name
    must also mention the quantity. A role that cannot be resolved is left unset
    and the checks that need it report ``not_applicable``; nothing is guessed.
    """
    roles = _Roles()
    distance_name = config.distance.name

    for column in config.all_features:
        # The distance column is swept directly, so it is never a held feature —
        # even when the RAW predict method also makes it an input feature.
        if column.name == distance_name:
            continue

        units = normalize_units(column.units)
        name = column.name.lower()

        if roles.speed is None and units in SPEED_UNITS:
            roles.speed, roles.speed_scale = column, 1.0
            continue
        if roles.speed is None and units in SPEED_UNITS_KPH:
            roles.speed, roles.speed_scale = column, KPH_TO_MPH
            continue
        if roles.grade is None and units in GRADE_UNITS_PERCENT:
            roles.grade, roles.grade_scale = column, 1.0
            continue
        if roles.grade is None and units in GRADE_UNITS_DECIMAL and "grade" in name:
            roles.grade, roles.grade_scale = column, 0.01
            continue
        if roles.mass is None and units in MASS_UNITS:
            roles.mass = column
            continue
        # Traversal time is not free to choose: it is distance over speed, both
        # of which the sweep varies. Holding it fixed would feed the model rows
        # describing a link that cannot exist.
        if roles.time is None and units in TIME_UNITS and "time" in name:
            roles.time, roles.time_scale = column, TIME_UNITS[units]
            continue

        roles.held[column.name] = _hold_value(column, reference_df, hold_values, notes)

    return roles


def _hold_value(
    column: DataColumn,
    reference_df: Optional[pd.DataFrame],
    hold_values: Optional[Dict[str, float]],
    notes: List[str],
) -> float:
    """Pick a fixed value for a feature with no physical role.

    Preference order runs from most to least informed: an explicit override, the
    median of real data, the midpoint of the declared constraints, a nominal
    value for the units, and finally zero — which is recorded as a note, since a
    guessed hold value colors every number in the report.
    """
    # An explicit override is trusted as given, bounds included.
    if hold_values and column.name in hold_values:
        return float(hold_values[column.name])

    def clamp(value: float) -> float:
        lower, upper = column.constraints.lower, column.constraints.upper
        if lower is not None:
            value = max(value, lower)
        if upper is not None:
            value = min(value, upper)
        return float(value)

    # Real data first: the median of what the model was trained on describes an
    # ordinary link better than any table can.
    if reference_df is not None and column.name in reference_df.columns:
        series = pd.to_numeric(reference_df[column.name], errors="coerce").dropna()
        if len(series):
            return clamp(float(series.median()))

    neutral = NEUTRAL_BY_UNITS.get(normalize_units(column.units))
    if neutral is not None:
        return clamp(neutral)

    lower, upper = column.constraints.lower, column.constraints.upper
    if lower is not None and upper is not None:
        return float((lower + upper) / 2.0)
    # A one-sided range gives the only defensible value: its own bound.
    if lower is not None:
        return float(lower)
    if upper is not None:
        return float(upper)

    notes.append(
        f"feature '{column.name}' has no neutral value, constraints, or "
        "reference data; held at 0.0"
    )
    return 0.0


def _sweep_range(
    column: Optional[DataColumn], values: np.ndarray, symmetric: bool
) -> np.ndarray:
    """Clip a sweep axis to a column's declared support.

    Testing a model outside the range it advertises would manufacture failures
    that say nothing about the model as it is meant to be used.
    """
    if column is None:
        return values
    lower, upper = column.constraints.lower, column.constraints.upper
    out = values
    limit = None
    if symmetric:
        # A symmetric axis (grade) is swept as matched +/- pairs, so both ends
        # must fit for a magnitude to be usable.
        if lower is not None:
            limit = abs(lower)
        if upper is not None:
            limit = upper if limit is None else min(limit, upper)
        if limit is not None:
            out = out[out <= limit + TOL]
    else:
        if lower is not None:
            out = out[out >= lower - TOL]
        if upper is not None:
            out = out[out <= upper + TOL]
    return out


# -- scoring ----------------------------------------------------------------


def _make_scorer(
    estimator: Estimator, config: ModelConfig
) -> Callable[[pd.DataFrame], np.ndarray]:
    """Build a function scoring one link per row, before any adjustment.

    A model with a lookback window is causal, so a bare one-row query would be
    scored against zero padding rather than against driving. Each query is
    instead repeated into a uniform trip and read off the last link, which is the
    only row whose window is fully populated.
    """
    target_name = config.target.target_name_list[0]
    spec = estimator.input_spec
    lookback = spec.lookback or 0
    group = spec.grouping_column

    def score(frame: pd.DataFrame) -> np.ndarray:
        if lookback <= 0 or group is None:
            out = estimator.predict(frame, config)
            return np.asarray(out[target_name], dtype=float)

        n = len(frame)
        expanded = frame.loc[frame.index.repeat(TRIP_LEN)].reset_index(drop=True)
        expanded[group] = np.repeat(np.arange(n), TRIP_LEN)
        out = estimator.predict(expanded, config)
        values = np.asarray(out[target_name], dtype=float)
        return values.reshape(n, TRIP_LEN)[:, -1]

    return score


def _build_frame(
    roles: _Roles,
    config: ModelConfig,
    speed_mph: np.ndarray,
    grade_pct: np.ndarray,
    distance_mi: np.ndarray,
    mass_lbs: Optional[float],
) -> pd.DataFrame:
    """Assemble a model-ready input frame from physical quantities."""
    frame = pd.DataFrame({config.distance.name: distance_mi})
    if roles.speed is not None:
        frame[roles.speed.name] = speed_mph / roles.speed_scale
    if roles.grade is not None:
        frame[roles.grade.name] = grade_pct * roles.grade_scale
    if roles.mass is not None and mass_lbs is not None:
        frame[roles.mass.name] = mass_lbs
    if roles.time is not None:
        seconds = np.divide(
            distance_mi * 3600.0,
            speed_mph,
            out=np.zeros_like(distance_mi, dtype=float),
            where=speed_mph > 0,
        )
        frame[roles.time.name] = seconds / roles.time_scale
    for name, value in roles.held.items():
        frame[name] = value
    return frame


def _resolve_mass(
    config: ModelConfig, roles: _Roles, mass_lbs: Optional[float]
) -> Tuple[Optional[float], Optional[float], str]:
    """Determine the vehicle mass to reason with.

    Returns ``(mass_lbs, mass_kg, source)``. An explicit argument wins; then the
    metadata field; then, for a model that takes mass as a feature, the midpoint
    of its declared range, because there the swept value is the true mass. Most
    published models carry no mass at all, and the mass-dependent checks then
    report ``not_applicable`` rather than guessing.
    """
    if mass_lbs is not None:
        return mass_lbs, mass_lbs * LB_TO_KG, "argument"
    if config.mass_lbs is not None:
        return config.mass_lbs, config.mass_lbs * LB_TO_KG, "metadata"
    if roles.mass is not None:
        lower, upper = roles.mass.constraints.lower, roles.mass.constraints.upper
        if lower is not None and upper is not None:
            mid = (lower + upper) / 2.0
            return mid, mid * LB_TO_KG, "mass feature midpoint"
    return None, None, "unavailable"


# -- check construction -----------------------------------------------------


class _Envelope(NamedTuple):
    """The energy terms bounding one link, in the target's own units."""

    #: Signed: positive on a climb, negative on a descent.
    potential: np.ndarray
    kinetic: np.ndarray
    resistance: np.ndarray
    accessory: np.ndarray
    eta_drive: float
    eta_regen: float

    @property
    def ceiling(self) -> np.ndarray:
        """The most a link could demand: lift, one full acceleration and road
        load through the driveline, plus accessories."""
        return (
            np.maximum(self.potential, 0.0) + self.kinetic + self.resistance
        ) / self.eta_drive + self.accessory

    @property
    def floor(self) -> np.ndarray:
        """The most a link could return: the drop and one full stop, at
        regeneration efficiency. Exactly zero for a combustion target."""
        return -self.eta_regen * (np.maximum(-self.potential, 0.0) + self.kinetic)


def _envelope(
    speed_mph: np.ndarray,
    grade_pct: np.ndarray,
    distance_mi: np.ndarray,
    mass_kg: "float | np.ndarray",
    joules_per_unit: float,
    is_combustion: bool,
    assumptions: PhysicsAssumptions,
) -> _Envelope:
    """The physical envelope of each link, from physical quantities.

    ``mass_kg`` may be a scalar or one value per link. Every term is generous
    by construction of ``assumptions``, so the band is a bound on any plausible
    vehicle rather than an estimate for this one.
    """
    eta_drive = (
        assumptions.eta_tank_to_wheel if is_combustion else assumptions.eta_drive
    )
    # Regeneration is a battery behavior; a combustion driveline stores nothing
    # back, so its only saving on a descent is fuel it does not burn.
    eta_regen = 0.0 if is_combustion else assumptions.eta_regen
    accessory_w = (
        assumptions.accessory_watts_combustion
        if is_combustion
        else assumptions.accessory_watts_electric
    )

    rise_m = distance_mi * MI_TO_M * grade_pct / 100.0
    potential = mass_kg * G * rise_m / joules_per_unit
    velocity = speed_mph * MPH_TO_MS
    kinetic = 0.5 * mass_kg * velocity**2 / joules_per_unit
    resistance = (
        (
            assumptions.crr * mass_kg * G
            + 0.5 * assumptions.air_density * assumptions.cda_m2 * velocity**2
        )
        * (distance_mi * MI_TO_M)
        / joules_per_unit
    )
    # Accessories are billed per unit time, so a slow link costs more of them.
    seconds = np.divide(
        distance_mi * MI_TO_M, velocity, out=np.zeros_like(velocity), where=velocity > 0
    )
    accessory = accessory_w * seconds / joules_per_unit
    return _Envelope(potential, kinetic, resistance, accessory, eta_drive, eta_regen)


def _na(name: str, reason: str) -> PhysicsCheck:
    return PhysicsCheck(name=name, status="not_applicable", reason=reason)


def _build_check(
    name: str,
    margin: np.ndarray,
    adjusted_margin: Optional[np.ndarray],
    context: Dict[str, np.ndarray],
    reason: Optional[str] = None,
) -> PhysicsCheck:
    """Summarize one predicate from its per-case margins.

    ``margin`` is negative exactly where the predicate is violated, and its
    magnitude there is how far into the impossible the prediction went.
    """
    margin = np.asarray(margin, dtype=float)
    finite = np.isfinite(margin)
    violated = finite & (margin < -TOL)
    n_tested = int(margin.size)
    n_violations = int(violated.sum())
    rate = n_violations / n_tested if n_tested else None

    adjusted_rate = None
    if adjusted_margin is not None and n_tested:
        adjusted = np.asarray(adjusted_margin, dtype=float)
        adjusted_rate = float(
            (np.isfinite(adjusted) & (adjusted < -TOL)).sum() / n_tested
        )

    check = PhysicsCheck(
        name=name,
        status="fail" if n_violations else "pass",
        reason=reason,
        n_tested=n_tested,
        n_violations=n_violations,
        violation_rate=rate,
        adjusted_violation_rate=adjusted_rate,
        would_bind_rate=rate,
    )
    if n_violations:
        worst = int(np.argmin(np.where(finite, margin, np.inf)))
        check.worst_margin = float(margin[worst])
        check.worst_case = {k: float(v[worst]) for k, v in context.items()}
    return check


def _fit_slope(x: np.ndarray, y: np.ndarray) -> Optional[float]:
    """Least-squares slope of ``y`` on ``x`` through the available points."""
    mask = np.isfinite(x) & np.isfinite(y)
    if mask.sum() < 2 or np.ptp(x[mask]) < TOL:
        return None
    return float(np.polyfit(x[mask], y[mask], 1)[0])


# -- entry point ------------------------------------------------------------


def check_model(model: "Model", **kwargs: object) -> PhysicsReport:
    """Check a loaded ``Model`` against physical law.

    A convenience wrapper over :func:`check_physics` for callers holding a
    ``Model`` rather than a bare estimator and config.

    Args:
        model: the model to check
        **kwargs: forwarded to :func:`check_physics`

    Returns: a ``PhysicsReport``
    """
    return check_physics(model.estimator, model.metadata.config, **kwargs)  # type: ignore[arg-type]


def check_physics(
    estimator: Estimator,
    config: ModelConfig,
    *,
    assumptions: Optional[PhysicsAssumptions] = None,
    reference_df: Optional[pd.DataFrame] = None,
    mass_lbs: Optional[float] = None,
    hold_values: Optional[Dict[str, float]] = None,
) -> PhysicsReport:
    """Check a trained estimator against physical law over a synthetic sweep.

    Args:
        estimator: the estimator to check, scored before any real-world
            adjustment and before any output guardrail
        config: the model configuration describing its inputs and target
        assumptions: vehicle constants for the checks that need more than mass;
            defaults are generous, so a violation against them is unambiguous
        reference_df: real data used to pick hold values for features with no
            physical role (e.g. the training test split). Optional
        mass_lbs: vehicle mass, enabling the mass-dependent checks when the
            model's metadata does not carry one
        hold_values: explicit values for features with no physical role,
            overriding every other source

    Returns: a ``PhysicsReport`` with one entry per check plus diagnostics
    """
    assumptions = assumptions or PhysicsAssumptions()
    notes: List[str] = []

    roles = _resolve_roles(config, reference_df, hold_values, notes)
    target = config.target.targets[0]
    target_units = normalize_units(target.units)
    joules_per_unit = ENERGY_CONTENT_J.get(target_units)
    is_combustion = is_combustion_target(target_units, config.powertrain_type)
    adjustment = config.real_world_adjustment_factor

    if len(config.target.targets) > 1:
        notes.append(
            f"model has {len(config.target.targets)} targets; "
            f"checks were run against '{target.name}'"
        )

    mass_lb, mass_kg, mass_source = _resolve_mass(config, roles, mass_lbs)
    report = PhysicsReport(
        assumptions=assumptions,
        target=target.name,
        mass_kg=mass_kg,
        mass_source=mass_source,
        notes=notes,
    )
    report.diagnostics.held_features = dict(roles.held)
    if roles.time is not None:
        report.diagnostics.derived_features = [roles.time.name]

    if roles.speed is None:
        notes.append("no speed feature could be resolved from units; nothing to sweep")
        report.checks = [
            _na(name, "no speed feature")
            for name in (
                "finite",
                "flat_energy_positive",
                "fuel_non_negative",
                "monotone_in_grade",
                "round_trip_convexity",
                "climb_floor",
                "regen_ceiling",
                "absolute_ceiling",
                "absolute_floor",
            )
        ]
        return report

    speeds = _sweep_range(roles.speed, SPEEDS_MPH, symmetric=False)
    grades = _sweep_range(roles.grade, GRADES_PCT, symmetric=True)
    distances = _sweep_range(config.distance, DISTANCES_MI, symmetric=False)
    if not len(speeds) or not len(distances):
        notes.append("declared constraints leave no usable sweep range")
        return report

    score = _make_scorer(estimator, config)

    # Every case is generated as a matched (flat, climb, descent) triple at one
    # speed, grade magnitude and length, so the round-trip statistic is exact
    # rather than interpolated.
    has_grade = roles.grade is not None and len(grades) > 0
    grade_axis = grades if has_grade else np.array([0.0])
    sp, gm, di = (
        a.ravel() for a in np.meshgrid(speeds, grade_axis, distances, indexing="ij")
    )

    def _score_at(grade_pct: np.ndarray) -> np.ndarray:
        frame = _build_frame(roles, config, sp, grade_pct, di, mass_lb)
        return score(frame)

    flat = _score_at(np.zeros_like(gm))
    if has_grade:
        climb = _score_at(gm)
        descent = _score_at(-gm)
    else:
        climb = descent = flat

    context = {"speed_mph": sp, "grade_pct": gm, "distance_mi": di}
    checks: List[PhysicsCheck] = []

    # -- constant-free checks -----------------------------------------------

    all_values = np.concatenate([flat, climb, descent])
    n_nonfinite = int((~np.isfinite(all_values)).sum())
    checks.append(
        PhysicsCheck(
            name="finite",
            status="fail" if n_nonfinite else "pass",
            reason="predictions contain NaN or infinity" if n_nonfinite else None,
            n_tested=int(all_values.size),
            n_violations=n_nonfinite,
            violation_rate=float(n_nonfinite / all_values.size),
        )
    )

    # A vehicle moving on level ground cannot produce net energy, whatever it
    # burns or stores.
    checks.append(
        _build_check(
            "flat_energy_positive",
            flat,
            flat * adjustment,
            context,
            reason="level-ground energy must be positive",
        )
    )

    if is_combustion:
        checks.append(
            _build_check(
                "fuel_non_negative",
                all_values,
                all_values * adjustment,
                {k: np.tile(v, 3) for k, v in context.items()},
                reason="burned fuel cannot return to the tank",
            )
        )
    else:
        checks.append(
            _na("fuel_non_negative", f"target '{target.name}' is not a fuel target")
        )

    if has_grade:
        # Steeper is never cheaper: both the climb and the descent must sit on
        # the correct side of level ground.
        monotone = np.minimum(climb - flat, flat - descent)
        checks.append(
            _build_check(
                "monotone_in_grade",
                monotone,
                monotone * adjustment,
                context,
                reason="energy must not decrease as grade increases",
            )
        )

        # The reported defect: a hill and its return leg cannot together cost
        # less than the same distance of level ground.
        excess = climb + descent - 2 * flat
        checks.append(
            _build_check(
                "round_trip_convexity",
                excess,
                excess * adjustment,
                context,
                reason="a round trip over a hill cannot cost less than flat ground",
            )
        )
    else:
        checks.append(_na("monotone_in_grade", "no grade feature"))
        checks.append(_na("round_trip_convexity", "no grade feature"))

    # -- mass-dependent checks ----------------------------------------------

    if mass_kg is None:
        skip = "no vehicle mass available"
    elif joules_per_unit is None:
        skip = f"unrecognized target units '{target.units}'"
    else:
        skip = ""

    mass_checks = ("climb_floor", "regen_ceiling", "absolute_ceiling", "absolute_floor")
    if skip:
        checks.extend(_na(name, skip) for name in mass_checks)
        report.checks = checks
        _add_diagnostics(report, config, sp, gm, di, flat, climb, descent, None)
        return report

    assert mass_kg is not None and joules_per_unit is not None
    env = _envelope(sp, gm, di, mass_kg, joules_per_unit, is_combustion, assumptions)
    potential, resistance = env.potential, env.resistance

    if has_grade:
        # Lifting the vehicle is work no drivetrain can avoid, so the climb must
        # cost at least the potential energy more than level ground.
        checks.append(
            _build_check(
                "climb_floor",
                (climb - flat) - potential,
                None,
                context,
                reason="a climb must cost at least its potential energy",
            )
        )
        if is_combustion:
            # A combustion vehicle on a descent can cut fuel entirely, so the
            # only true bound on what it saves is that it cannot burn less than
            # nothing -- which ``fuel_non_negative`` already checks exactly.
            checks.append(
                _na(
                    "regen_ceiling",
                    "fuel targets are bounded by fuel_non_negative instead",
                )
            )
        else:
            # A descent saves two different things: the energy it recovers from
            # the hill, and the road-load energy the flat leg would have spent
            # covering that ground anyway. Bounding only the first understates
            # the legitimate saving and would flag correct models.
            recoverable = np.minimum(resistance, np.abs(potential))
            max_saving = recoverable / env.eta_drive + env.eta_regen * (
                np.abs(potential) - recoverable
            )
            checks.append(
                _build_check(
                    "regen_ceiling",
                    max_saving - (flat - descent),
                    None,
                    context,
                    reason="a descent cannot return more energy than the hill holds",
                )
            )
    else:
        checks.extend(_na(name, "no grade feature") for name in mass_checks[:2])

    tiled = {k: np.tile(v, 3) for k, v in context.items()}
    everything = np.concatenate([flat, climb, descent])
    # The same envelope over the (flat, climb, descent) triples, so the grade
    # carries its sign and the bound is the one ``physical_bounds`` applies.
    signed = _envelope(
        np.tile(sp, 3),
        np.concatenate([np.zeros_like(gm), gm, -gm]),
        np.tile(di, 3),
        mass_kg,
        joules_per_unit,
        is_combustion,
        assumptions,
    )
    ceiling, floor = signed.ceiling, signed.floor

    checks.append(
        _build_check(
            "absolute_ceiling",
            ceiling - everything,
            ceiling - everything * adjustment,
            tiled,
            reason="prediction exceeds the energy the link could possibly demand",
        )
    )
    checks.append(
        _build_check(
            "absolute_floor",
            everything - floor,
            everything * adjustment - floor,
            tiled,
            reason="prediction returns more energy than the link could possibly hold",
        )
    )

    report.checks = checks
    _add_diagnostics(
        report,
        config,
        sp,
        gm,
        di,
        flat,
        climb,
        descent,
        (mass_kg, joules_per_unit),
    )
    return report


def physical_bounds(
    links_df: pd.DataFrame,
    config: ModelConfig,
    *,
    assumptions: Optional[PhysicsAssumptions] = None,
    mass_lbs: Optional[float] = None,
) -> Dict[str, Tuple[np.ndarray, np.ndarray]]:
    """The physical energy band of each link, per target, in target units.

    This is the band ``check_physics`` scores as ``absolute_floor`` and
    ``absolute_ceiling``, evaluated on real links instead of a sweep. It is
    loose by design (see ``PhysicsAssumptions``), and ``apply_guardrail`` uses
    its ceiling; see there for why the electric floor is reported but not
    enforced.

    Args:
        links_df: the frame handed to the estimator, in the model's own units
        config: the model configuration describing its inputs and targets
        assumptions: vehicle constants; the generous defaults when omitted
        mass_lbs: vehicle mass, used when the model carries none

    Returns: ``{target name: (floor, ceiling)}``, one array pair per target
        whose units are a recognized energy. Empty when the band cannot be
        evaluated at all: distance not in miles, no speed feature, or no
        vehicle mass. Rows with a negative or non-finite speed or distance get
        ``NaN`` bounds, so a caller can leave them alone.
    """
    assumptions = assumptions or PhysicsAssumptions()
    if normalize_units(config.distance.units) not in DISTANCE_UNITS_MILES:
        return {}
    roles = _resolve_roles(config, None, None, [])
    if roles.speed is None:
        return {}

    # Mass: one value per link when the model takes it as a feature, else the
    # metadata's, else the caller's. A mass feature's declared range only ever
    # stands in for the sweep, never for a real link.
    mass_kg: "float | np.ndarray | None"
    if roles.mass is not None and roles.mass.name in links_df.columns:
        mass_kg = links_df[roles.mass.name].to_numpy(dtype=float) * LB_TO_KG
    else:
        mass_kg = _resolve_mass(config, roles, mass_lbs)[1]
    if mass_kg is None:
        return {}

    speed_mph = links_df[roles.speed.name].to_numpy(dtype=float) * roles.speed_scale
    distance_mi = links_df[config.distance.name].to_numpy(dtype=float)
    if roles.grade is not None:
        grade_pct = links_df[roles.grade.name].to_numpy(dtype=float) / roles.grade_scale
    else:
        grade_pct = np.zeros_like(speed_mph)
    usable = (
        np.isfinite(speed_mph)
        & np.isfinite(distance_mi)
        & np.isfinite(grade_pct)
        & (speed_mph >= 0)
        & (distance_mi >= 0)
    )
    if not isinstance(mass_kg, float):
        usable &= np.isfinite(mass_kg) & (mass_kg > 0)
    nan = np.full(speed_mph.shape, np.nan)

    out: Dict[str, Tuple[np.ndarray, np.ndarray]] = {}
    for target in config.target.targets:
        units = normalize_units(target.units)
        joules_per_unit = ENERGY_CONTENT_J.get(units)
        if joules_per_unit is None:
            continue
        env = _envelope(
            speed_mph,
            grade_pct,
            distance_mi,
            mass_kg,
            joules_per_unit,
            is_combustion_target(units, config.powertrain_type),
            assumptions,
        )
        # ``+ 0.0`` turns a combustion floor of ``-0.0`` into ``0.0``.
        floor = np.where(usable, env.floor + 0.0, nan)
        ceiling = np.where(usable, env.ceiling, nan)
        out[target.name] = (floor, ceiling)
    return out


def apply_guardrail(
    predictions: pd.DataFrame, links_df: pd.DataFrame, config: ModelConfig
) -> pd.DataFrame:
    """Clip each target column of ``predictions`` to its physical ceiling, and
    to zero for a fuel target.

    The electric floor from ``physical_bounds`` is deliberately not enforced.
    Its kinetic term is one stop from the link's *average* speed, and a slow
    link's average hides the far higher speed the vehicle actually braked
    from: on simulated Bolt links, one in twenty returns more energy than that
    floor allows, one in four at 5-15 mph. Enforcing it would clip correct
    predictions. The ceiling has no such problem — a link's average speed
    cannot hide a demand — and it is what stops a model running away on a
    link far outside its training range. Burned fuel cannot be negative under
    any driving, so that floor is exact.

    Only rows with a finite bound are touched; a ``NaN`` prediction stays
    ``NaN``, and columns that are not targets (a standard deviation, say) pass
    through. Returns ``predictions``, modified in place.
    """
    bounds = physical_bounds(links_df, config)
    for target in config.target.targets:
        if target.name not in bounds:
            continue
        floor, ceiling = bounds[target.name]
        if not is_combustion_target(
            normalize_units(target.units), config.powertrain_type
        ):
            floor = np.full(ceiling.shape, -np.inf)
        name = target.name
        values = predictions[name].to_numpy(dtype=float)
        ok = np.isfinite(ceiling) & (floor <= ceiling)
        clipped = np.where(ok, np.clip(values, floor, ceiling), values)
        if log.isEnabledFor(logging.DEBUG):
            n_bound = int(np.count_nonzero(ok & (clipped != values)))
            log.debug(
                "guardrail bound %d of %d links of %s", n_bound, values.size, name
            )
        predictions[name] = clipped
    return predictions


def _add_diagnostics(
    report: PhysicsReport,
    config: ModelConfig,
    sp: np.ndarray,
    gm: np.ndarray,
    di: np.ndarray,
    flat: np.ndarray,
    climb: np.ndarray,
    descent: np.ndarray,
    mass_and_content: Optional[Tuple[float, float]],
) -> None:
    """Fill in the descriptive half of the report."""
    d = report.diagnostics
    target_units = normalize_units(config.target.targets[0].units)
    is_combustion = is_combustion_target(target_units, config.powertrain_type)

    # -- flat-ground economy, read against a plausible envelope --------------
    band = ECONOMY_BANDS.get(config.powertrain_type)
    d.economy_band = band
    # ``flat`` holds level-ground energy for every case, whatever grade magnitude
    # that case pairs with, so grouping by speed alone is what is wanted here.
    for speed in np.unique(sp):
        mask = sp == speed
        if not mask.any():
            continue
        rate_per_mile = np.divide(
            flat[mask], di[mask], out=np.full(mask.sum(), np.nan), where=di[mask] > 0
        )
        mean_rate = float(np.nanmean(rate_per_mile))
        if not np.isfinite(mean_rate) or abs(mean_rate) < TOL:
            continue
        # Distance per unit energy: mpg for a fuel target, mi/kWh for electric.
        economy = 1.0 / mean_rate
        d.flat_economy[f"{speed:g}"] = economy
        if band is not None and not (band[0] <= economy <= band[1]):
            d.economy_outliers.append(float(speed))

    # -- implied efficiencies, from the shallow-grade slopes -----------------
    if mass_and_content is not None:
        mass_kg, joules_per_unit = mass_and_content
        shallow = (np.abs(gm) > TOL) & (np.abs(gm) <= SHALLOW_GRADE_PCT)
        if shallow.any():
            potential = mass_kg * G * (di * MI_TO_M * gm / 100.0) / joules_per_unit
            # Slope of the extra energy a climb costs against the energy it must
            # store as height: the reciprocal of drivetrain efficiency.
            climb_slope = _fit_slope(potential[shallow], (climb - flat)[shallow])
            if climb_slope and climb_slope > TOL:
                d.implied_eta_drive = 1.0 / climb_slope
                if d.implied_eta_drive > 1.0:
                    report.notes.append(
                        f"implied drivetrain efficiency is "
                        f"{d.implied_eta_drive:.2f}; a value above 1.0 means the "
                        "model bills a climb for less than the height it gains"
                    )
            # Slope of the energy a descent gives back against the energy the
            # hill held. For an electric target this is regeneration efficiency
            # directly; for a fuel target it measures descent sensitivity, since
            # a combustion driveline stores nothing back.
            regen_slope = _fit_slope(potential[shallow], (flat - descent)[shallow])
            if regen_slope is not None:
                d.implied_eta_regen = regen_slope
                if not is_combustion:
                    d.regen_exceeds_unity = bool(regen_slope > 1.0)
                    if d.regen_exceeds_unity:
                        report.notes.append(
                            f"implied regeneration efficiency is {regen_slope:.2f}; "
                            "a value above 1.0 means the model returns energy the "
                            "hill never contained"
                        )

    # -- does the answer depend on how the route was segmented? --------------
    spreads = []
    for speed in np.unique(sp):
        for grade in np.unique(gm):
            mask = (sp == speed) & (gm == grade)
            if mask.sum() < 2:
                continue
            rate = np.divide(
                climb[mask],
                di[mask],
                out=np.full(mask.sum(), np.nan),
                where=di[mask] > 0,
            )
            scale = np.nanmean(np.abs(rate))
            if np.isfinite(scale) and scale > TOL:
                spreads.append(float(np.nanmax(rate) - np.nanmin(rate)) / scale)
    if spreads:
        d.length_invariance = float(np.max(spreads))
