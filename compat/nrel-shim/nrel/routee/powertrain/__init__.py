"""Tombstone for the renamed ``nrel.routee.powertrain`` distribution.

Importing this package always fails, by design. RouteE-Powertrain is now
published as ``routee.powertrain``; installing this shim pulls the real package
in as a dependency, so the only thing left to do is change the import.

The error is raised rather than re-exporting ``routee.powertrain`` because v2 is
a breaking release -- ``Model``, ``load_model``, and the on-disk model format all
changed -- so a silent alias would turn one clear failure into a cascade of
confusing ones downstream.
"""

raise ImportError(
    "'nrel.routee.powertrain' has been renamed to 'routee.powertrain'.\n"
    "\n"
    "    import routee.powertrain as pt\n"
    "\n"
    "routee.powertrain 2.x has been installed alongside this shim, so no further\n"
    "install step is needed. Note that v2 is a breaking release: Model, load_model,\n"
    "and the on-disk model format all changed.\n"
    "\n"
    "Migration guide: "
    "https://natlabrockies.github.io/routee-powertrain/migrating_from_v1.html"
)
