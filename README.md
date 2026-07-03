# BrainYears2026

Runnable code package for the BrainYears manuscript.

## Files

- model.py: runs the BrainYears model inference workflow.
- generate_figures.py: generates manuscript-related figure panels.
- model_bundle.joblib: serialized trained model and preprocessing bundle.
- model_simulated_data.csv: simulated demo dataset for code execution.
- figures/: generated SVG figure panels.

## Demo data

This repository includes model_simulated_data.csv, a fully simulated demo dataset provided only to test that the code runs. It does not contain real Sens.ai participant-level data and should not be used to reproduce the manuscript results or interpret model performance.

The real Sens.ai participant-level dataset is not included in this repository.

## Running the code

Install dependencies:

pip install numpy pandas scikit-learn scipy joblib matplotlib

Run the model:

python model.py

Generate figures:

python generate_figures.py

## Notes

The simulated CSV preserves the expected input structure for the code but contains synthetic values only. Manuscript performance metrics were generated using the original analysis dataset, not the simulated demo data.

Questions: Sierra Lore, sierralore@alumni.stanford.edu
