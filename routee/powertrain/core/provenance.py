from __future__ import annotations

from enum import Enum
from typing import TYPE_CHECKING, Annotated, List, Literal, Optional, Union

from pydantic import BaseModel, Field

if TYPE_CHECKING:
    from routee.powertrain.core.model_config import ModelConfig

__all__ = [
    "TrainingMethod",
    "FastSimSource",
    "RealWorldSource",
    "LegacySource",
    "TrainingSource",
    "TrainingConfig",
    "Provenance",
]


class TrainingMethod(str, Enum):
    """How a model's training data was produced.

    Doubles as the discriminator for the ``TrainingSource`` union — each source
    variant pins ``method`` to exactly one of these values.
    """

    FASTSIM_SIMULATION = "fastsim_simulation"
    REAL_WORLD = "real_world"
    LEGACY = "legacy"


class FastSimSource(BaseModel):
    """Training data produced by simulating a FASTSim vehicle.

    The standard path: a model pipeline runs drive cycles through FASTSim and
    trains on the resulting energy traces. Recording the vehicle, the simulator
    version, and the pipeline that drove them is what makes a published model
    reproducible.

    The pipeline keeps its own provenance database, so this records **keys, not
    copies**. ``pipeline_run_id`` and ``dataset_run_ids`` resolve there to the
    full configuration each run used — dataset filters, estimator settings, trip
    caps, the sampling seed, the identity of the assembled training frame, and
    everything else that would have to match to recreate the model. None of that
    is duplicated here: a copy drifts from the source of truth and can't be
    verified against it, while a key can't drift.

    What remains alongside the keys describes *what was simulated* rather than
    how the training data was assembled — the vehicle, the simulator version,
    the pipeline version. Those are cheap, stable, and readable at a glance.

    This assumes the provenance database is reachable whenever a model needs to
    be reproduced. That is a deliberate trade, and a cheap one to revisit:
    provenance is excluded from ``model_digest``, so a field added here later is
    non-breaking and can even be backfilled onto already-published models
    without changing their identity.
    """

    method: Literal[TrainingMethod.FASTSIM_SIMULATION] = (
        TrainingMethod.FASTSIM_SIMULATION
    )

    #: Vehicle identifier in https://github.com/NatLabRockies/fastsim-vehicles
    #: (e.g. ``"v1/fastsim-3/conv/toyota/camry-4cyl-2wd/2016/base/r1"``).
    fastsim_vehicle_id: Optional[str] = None
    #: Git tag / commit sha pinning the ``fastsim-vehicles`` repo the vehicle
    #: definition was read from.
    fastsim_vehicles_ref: Optional[str] = None
    #: Version of the FASTSim package that ran the simulation.
    fastsim_version: Optional[str] = None

    #: Version of the model pipeline that orchestrated simulation and training.
    pipeline_version: Optional[str] = None
    pipeline_run_id: Optional[str] = None
    #: Git commit sha of the pipeline repo at run time.
    pipeline_repo_ref: Optional[str] = None

    dataset_run_ids: List[str] = Field(default_factory=list)
    #: Dataset sources the training data was drawn from (e.g. ``["wm1"]``).
    #: Usually one, but a run can be configured to sample across several
    #: (e.g. ``["wm1", "wm2"]``).
    data_sources: List[str] = Field(default_factory=list)
    notes: Optional[str] = None


class RealWorldSource(BaseModel):
    """Training data collected from instrumented vehicles in the field."""

    method: Literal[TrainingMethod.REAL_WORLD] = TrainingMethod.REAL_WORLD

    #: Name of the data collection program or provider.
    data_source: Optional[str] = None
    #: Fleet the vehicles were drawn from, when the source spans several.
    fleet: Optional[str] = None
    #: Collection window, as ISO ``YYYY-MM-DD`` strings.
    collection_start: Optional[str] = None
    collection_end: Optional[str] = None
    #: Size of the underlying sample.
    n_vehicles: Optional[int] = None
    n_trips: Optional[int] = None

    #: Human-readable identifier of the collected dataset the model was fit to.
    dataset_name: Optional[str] = None
    #: Fingerprint of that data — see ``routee.powertrain.hash_dataframe``.
    dataset_hash: Optional[str] = None

    notes: Optional[str] = None


class LegacySource(BaseModel):
    """Provenance for models that predate this section.

    Converted v1 archives record no simulator, pipeline, or dataset
    information — the honest answer is "we don't know", and this variant says
    so explicitly rather than leaving the source null.
    """

    method: Literal[TrainingMethod.LEGACY] = TrainingMethod.LEGACY

    #: Where the model came from before conversion (e.g. a library name).
    original_source: Optional[str] = None
    #: The format it was converted from (e.g. ``"v1"``).
    converted_from: Optional[str] = None

    #: Human-readable identifier of the training data, when the pre-conversion
    #: format happened to record one.
    dataset_name: Optional[str] = None
    #: Fingerprint of that data — see ``routee.powertrain.hash_dataframe``.
    dataset_hash: Optional[str] = None

    notes: Optional[str] = None


#: Discriminated union of the training-source variants. Pydantic selects the
#: concrete class from the ``method`` field, so a serialized source round-trips
#: back to the type it was written as.
TrainingSource = Annotated[
    Union[FastSimSource, RealWorldSource, LegacySource],
    Field(discriminator="method"),
]


class TrainingConfig(BaseModel):
    """Build-time hyperparameters — needed to reproduce training, not to use the
    model. Safe to drop from a shipped artifact without affecting prediction.
    """

    test_size: Optional[float] = None
    validation_size: Optional[float] = None
    random_seed: int = 42
    trip_column: str = "trip_id"

    #: Calendar date the model was trained, as an ISO ``YYYY-MM-DD`` string.
    #: Stamped at training time by ``Trainer.train``; ``None`` when unknown
    #: (e.g. legacy models converted from the v1 format).
    trained_date: Optional[str] = None

    @classmethod
    def from_config(cls, config: ModelConfig) -> TrainingConfig:
        return cls(
            test_size=config.test_size,
            validation_size=config.validation_size,
            random_seed=config.random_seed,
            trip_column=config.trip_column,
        )


class Provenance(BaseModel):
    """Where a model came from and how it was built.

    Two complementary answers: ``source`` — what produced the training data
    (FASTSim simulation, real-world collection, or an unknown legacy origin),
    including the dataset labels for that data; and ``training`` — the
    hyperparameters the fit ran under.

    Deliberately excluded from ``model_digest`` (see ``core.digest``). The
    estimator binary's sha256 already pins the exact data and hyperparameters a
    model was fit to, so making provenance identity-bearing would only make it
    uncorrectable — backfilling a FASTSim version onto a published model would
    change the model's identity.
    """

    #: What produced the training data. ``None`` when nothing is recorded.
    source: Optional[TrainingSource] = None
    training: TrainingConfig = Field(default_factory=TrainingConfig)

    @property
    def method(self) -> Optional[TrainingMethod]:
        """The recorded training method, or ``None`` when no source is set."""
        return self.source.method if self.source is not None else None

    @property
    def dataset_name(self) -> Optional[str]:
        """The training dataset's label, when the source records one.

        ``None`` for a ``FastSimSource`` — simulated training data is described
        by ``dataset_run_ids`` against the pipeline's provenance database rather
        than by a label in the artifact — and ``None`` when no source is set.
        """
        return getattr(self.source, "dataset_name", None)

    @property
    def dataset_hash(self) -> Optional[str]:
        """The training data's fingerprint, when the source records one.

        ``None`` for a ``FastSimSource``, for the same reason as
        ``dataset_name``, and ``None`` when no source is set.
        """
        return getattr(self.source, "dataset_hash", None)

    @classmethod
    def from_config(cls, config: ModelConfig) -> Provenance:
        return cls(
            source=config.training_source,
            training=TrainingConfig.from_config(config),
        )
