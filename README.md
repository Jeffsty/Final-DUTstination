# Final-DUTstination

*Saving good Devices Under Test (DUTs) from their falsely predicted final destination!*

## Project Overview

This repository contains the research project code for developing and evaluating machine learning models (MLP and Random Forest) to predict the likelihood of a failed Automated Test Equipment (ATE) result being a **False Positive** for a specific type of PCB (DUT). The goal is to provide actionable insights to production operatives, helping them decide the best next step (e.g., retest, send to rework) rather than automatically trusting potentially misleading ATE failure flags.

## Goal & Objectives

The primary goal of this research is to develop a reliable classifier that, given the data from an ATE failure event, predicts the probability that the failure is a False Positive.

Specific Objectives:

1.  Prepare and preprocess ATE test log data, focusing on instances where a DUT failed testing.
2.  Develop, train, and tune two distinct classification models:
    *   A Multi-Layer Perceptron (MLP) neural network.
    *   A Random Forest (RF) classifier.
3.  Address potential class imbalance between True Failures and False Positives in the training data.
4.  Rigorously evaluate and compare the performance of the MLP and RF models using appropriate metrics, benchmarked against ground truth labels established by the quality department.
5.  Provide insights into which model is more suitable for deployment as a decision-support tool for production operators.

**Important Note:** This system is intended as a **decision-support tool** to provide recommendations, *not* to automatically assign a "Pass" state to a DUT.

## 📊 Dataset

The models are trained on historical ATE test logs containing results for multiple DUTs.

**Note:** The raw data used for this research project may contain sensitive manufacturing information and is not included in this public repository.
