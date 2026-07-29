# Estimator Input/Output Contract

A routee-powertrain model runs its predictions through an underlying estimator
(an ONNX graph, an NGBoost model, …). For the ONNX case the inference engine
consumes a **positional** float tensor — column 0, column 1, … — so a consumer
must feed features in the exact order the model was trained on. Getting the
order wrong does not error; it silently produces wrong energy.

To make that order unambiguous, every estimator ships a **self-describing
input/output contract**. This page documents the contract so downstream
consumers (notably [routee-compass](https://github.com/NatLabRockies/routee-compass))
can read it instead of relying on an out-of-band assumption.

## Where the contract lives

The contract lives in **two artifacts**, each with a different reader:

1. **`metadata.json`** — the ordered contract is stored once, in the `contract`
   section (`feature_set` order, `distance`, `target`, `predict_method`). This is
   the source of truth for any reader that has the metadata. `estimator.input_spec`
   in the same file carries only the estimator _mechanics_ — `lookback`,
   `grouping_column`, `pad_strategy` — not a second copy of the columns.
2. **The estimator binary itself** — for ONNX models the _resolved, positional_
   contract is embedded in the graph's `metadata_props`, so a bare `.onnx` file is
   self-describing **without** the sidecar `metadata.json`. This is the copy a
   consumer that only has the binary (e.g. routee-compass) reads.

The two are not redundant: they serve different consumers. On load, powertrain
**raises** if the binary's embedded order disagrees with the `contract` in
`metadata.json` — a corrupt or hand-edited artifact fails loudly rather than
mispredicting.

## The fields

The **resolved, positional** contract embedded in the ONNX binary (and the
in-memory `InputSpec`) carries:

| Field             | Meaning                                                                                                                                                                                          |
| ----------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| `input_columns`   | Ordered positional columns of the **input** tensor. Each is `{name, units, dtype}`. This is the feature columns in order, with the distance column appended last when `predict_method == "raw"`. |
| `output_columns`  | Ordered positional columns of the **output** tensor. Point estimators emit one column per energy target; NGBoost appends a `<target>_std` column.                                                |
| `predict_method`  | `"rate"` (raw output is energy-per-distance; multiply by distance to get energy) or `"raw"` (raw output is already absolute energy).                                                             |
| `distance_column` | Name of the distance column — the RATE multiplier, and the final RAW input position.                                                                                                             |
| `lookback`        | Rows of prior context per prediction. `0` = pointwise/tabular. `> 0` = a `(N, lookback, F)` windowed sequence model (1D CNN).                                                                    |
| `grouping_column` | Column that buckets rows into independent sequences (e.g. `route_id`). Required when `lookback > 0`.                                                                                             |
| `pad_strategy`    | How the lookback window is padded at a sequence start (`"repeat_first"` or `"zero"`).                                                                                                            |

In `metadata.json`, only the last three (`lookback`/`grouping_column`/`pad_strategy`)
appear under `estimator.input_spec`; the ordered columns come from the `contract`
section (and are embedded in full in the binary).

## ONNX `metadata_props` keys

For ONNX estimators the same fields are embedded under these keys (all values
are JSON where noted):

| Key                      | Value                                                          |
| ------------------------ | -------------------------------------------------------------- |
| `routee_input_columns`   | JSON array of `{name, units, dtype}`, positional input order   |
| `routee_output_columns`  | JSON array of `{name, units, dtype}`, positional output order  |
| `routee_predict_method`  | `"rate"` or `"raw"`                                            |
| `routee_distance_column` | distance column name                                           |
| `routee_lookback`        | integer (written only when `> 0`)                              |
| `routee_grouping_column` | grouping column name (written only when `lookback > 0`)        |
| `routee_pad_strategy`    | `"repeat_first"` / `"zero"` (written only when `lookback > 0`) |

The input tensor node is named `"input"` and the output node `"output"`.

## How a consumer should use it

1. On load, read `routee_input_columns` from the `.onnx` `metadata_props` (via
   the ONNX session's model metadata).
2. Note that the names are **powertrain** feature names (`speed_mph`,
   `grade_percent`). Map each powertrain column to the consumer's
   source value **by the config's declared mapping**, and use `units` to drive
   any unit conversion.
3. Build the feature vector in **contract order** (`routee_input_columns`), not
   in the order the consumer's config happens to list features.
4. Validate at load: every embedded input column must have a mapping and a
   compatible unit; otherwise error.

## Migrating older artifacts

Registry entries minted before the contract existed carry no embedded
`input_columns`. Re-embed them (deriving the contract from each model's own
metadata) with:

```
python scripts/backfill_input_contract.py <registry_root>
```
