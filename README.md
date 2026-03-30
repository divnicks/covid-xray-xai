Explainable Deep Learning for COVID-19 Chest X-Ray Diagnosis

This project implements an Explainable Deep Learning framework for detecting COVID-19 from chest X-ray (CXR) images while ensuring model transparency and clinical relevance.

Unlike standard “black-box” CNN models, this system combines:

- Lung-region focused preprocessing

- Multi-channel feature representation

- Grad-CAM visual explanations

- Quantitative attention analysis

Performance

Accuracy: 95.3%

AUC: 0.983

MCC: ~0.83

The model focuses attention primarily within anatomically meaningful lung regions, improving interpretability and trust.

Method Summary

Lung segmentation & bone suppression

Four-channel feature construction

Modified Xception CNN classifier

Grad-CAM explanation + attention validation

This repo includes

- Model training & evaluation pipeline

- Explainability module

- Flask-based prototype interface



