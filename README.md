# Magnetoencephalography and Electroencephalography Brainwave Activity Visualization

This repository contains code and methodologies for visualizing brainwave activity using Magnetoencephalography (MEG) and Electroencephalography (EEG) data. The project aims to provide innovative 2D and 3D visualizations of neural activity, along with robust classification and validation techniques.

---

## Overview

Understanding human brain functions is critical for applications in mental health treatments and cognitive science. This project tackles challenges in brainwave data visualization by processing and classifying MEG/EEG datasets, generating 2D and 3D visualizations of brainwave activity, and employing cross-validation techniques to ensure accuracy in classification.

---

## How It Works

This project employs the following methodologies:

1. **Data Processing and Classification**:
   - MEG and EEG datasets are preprocessed to filter noise and categorize brainwave signals.
   - Signals are grouped based on frequencies and mental activities (e.g., auditory vs. visual tasks).

   <img width="541" alt="image" src="https://github.com/user-attachments/assets/a30d6971-a585-41bf-8d2b-451934993c49" />

2. **2D and 3D Visualization**:
   - **Signal Traces**: Time-series plots representing neural activity.
   - **Scalp Topographies**: Heatmaps visualizing brainwave activity over specific scalp areas.
   - **Arrow Maps**: Visuals displaying magnetic field magnitude and direction.
   - **3D Field Maps**: Interactive 3D models showing neural activity distribution.
   
   <img width="803" alt="image" src="https://github.com/user-attachments/assets/fa230b59-fefc-46c9-9969-d751dcbdde18" />

   <img width="1406" alt="image" src="https://github.com/user-attachments/assets/cbd8b3fb-6f45-4bab-a49d-a53cc127c0b9" />

4. **Evaluation**:
   - A Random Forest Classifier is trained on raw evoked data.
   - The classifier is validated using Leave-One-Out Cross-Validation (LOOCV) to measure accuracy between auditory and visual conditions.

5. **Software Environment**:
   - All visualizations and analyses are powered by the `MNE-Python` library, a robust framework for analyzing EEG and MEG data.

---

## Features

- Categorization of brainwave signals based on frequency and mental activities.
- 2D and 3D visualizations including Signal Traces, Scalp Topographies, and 3D Field Maps.
- Leave-One-Out Cross-Validation (LOOCV) for robust evaluation of classification performance.
- Support for two datasets: MNE Sample Dataset and Alcoholic Group Dataset.

---

## Requirements

To use this project, ensure you have the following:

- **Python 3.8+**
- Libraries:
  - `numpy`
  - `pandas`
  - `matplotlib`
  - `mne`
  - `scikit-learn`
  - `tqdm`

Install dependencies via:
```bash
pip install -r requirements.txt
