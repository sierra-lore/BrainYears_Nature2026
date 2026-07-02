# BrainYears2026

This repository contains the runnable analysis and model files for **BrainYears**, an ERP–EEG-based brain age clock developed to estimate chronological age from functional electrophysiological features.

BrainYears uses features derived from EEG and event-related potential recordings collected with the Sens.ai platform. In the manuscript, the model predicts chronological age across adulthood with strong held-out performance, including Pearson r = 0.92 and MAE = 4.42 years. The model uses ERP-, EEG-, behavioral-, demographic-, spectral-, and composite-derived features.

## Repository contents

```text
generate_figures.py      # Regenerates SVG figure panels from model.csv and model_bundle.joblib
model.py                 # Trains and evaluates the BrainYears model
model_bundle.joblib      # Trained BrainYears model bundle
model.csv                # Processed feature-level input table used for model training  
```

## System requirements

### Software

The code was written in Python and requires the following packages:

```text
Python 3.12
numpy 2.5.0
pandas 3.0.3
scipy 1.18.0
scikit-learn 1.9.0
joblib 1.5.3
matplotlib 3.11.0
seaborn 0.13.2
```

### Operating systems

The code should run on macOS, Linux, or Windows with a standard Python installation.

### Hardware

No non-standard hardware is required. A standard desktop or laptop computer should be sufficient for the included files.

## Installation

Clone the repository:

```bash
git clone git@github.com:sierra-lore/BrainYears2026.git
cd BrainYears2026
```

Install dependencies:

```bash
pip install numpy pandas scipy scikit-learn joblib matplotlib seaborn
```

Typical installation time on a standard desktop computer is less than 5 minutes, excluding time required to install Python if it is not already available.

## Demo

To train and evaluate the BrainYears model using the included processed feature table:

```bash
python model.py
```

This script loads `model.csv`, filters samples to ages 18.0–89.99 years, performs an 80:20 train-test split, fits all preprocessing steps on the training set only, trains the two-stage BrainYears model, applies polynomial bias correction, prints train/test performance metrics, and saves the trained model bundle to:

```text
model_bundle.joblib
```

Expected terminal output includes the number of samples and features used, the number of features retained after variance filtering, train/test MAE and Pearson correlation before bias correction, and final train/test MAE and Pearson correlation after bias correction.

Typical runtime for `model.py` on a standard desktop computer is expected to be a few minutes.

To regenerate figure panels:

```bash
python generate_figures.py
```

This script creates a `figures/` directory and writes SVG figure panels to that folder.

Typical runtime for `generate_figures.py` on a standard desktop computer is expected to be a few minutes.

## Model overview

BrainYears uses a two-stage machine-learning architecture. First, ElasticNet regression captures sparse linear age-associated structure from scaled features. Second, a gradient-boosted regressor is trained on residuals using clipped raw features and the ElasticNet prediction as inputs. A polynomial bias-correction model is then fit on training-set prediction error as a function of predicted age and applied to final predictions.

All preprocessing steps are fit only on the training set and then applied unchanged to the held-out test set.

## Reproduction note

This repository supports model training, held-out evaluation, model serialization, and figure generation from processed feature-level inputs. It does not include raw EEG recordings or identifiable participant-level data. Therefore, full reproduction from raw device recordings is not possible from this repository alone.

## Citation

If you use this code or model, please cite:

Lore S., Julihn C., Telfer P., Scheibye-Knudsen M., and Verdin E.
**BrainYears: A functional ERP–EEG brain age clock for scalable assessment of brain aging.**

## Contact

For questions, please contact:

Sierra Lore
[sierralore@alumni.stanford.edu](mailto:sierralore@alumni.stanford.edu)


