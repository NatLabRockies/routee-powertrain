# Developer Scripts

Scripts used by the developers for various tasks.

## train_model_catalog.py

Trains the model catalog from a directory of drive-cycle datasets.
`kestrel_train_model_catalog.sh` runs the same job as a Slurm batch script.

## cnn_training_example.py

End-to-end example of training a 1D CNN model with `CNNTrainer` and exporting it to ONNX.

## Publishing models

Publishing moved to the registry in v2. See `scripts/upload_to_s3.py` to push a local registry
tree to S3 and `scripts/build_s3_index.py` to (re)build the `index.json` used for fast queries,
plus [docs/publishing_a_model.md](../../docs/publishing_a_model.md) for the full workflow.

The old Box-hosted download-link workflow (`build_box_shared_links.py`) was removed along with
the v1 `external_model_links.json` catalog.
