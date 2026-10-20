# Capstone Project: Adaptive SOC Alert Prioritization

**VIT Bhopal University** · **DSN4091 — Capstone Project, Phase I**  
**Group G23** · **Supervisor: Dr. Adarsh Patel**

## Overview

Security operations centres receive more alerts than analysts can investigate one by one. This capstone explores whether agreement between multiple machine-learning classifiers can help decide which alerts may be handled automatically and which should be reviewed by an analyst.

The work is inspired by *Learning to Defer with Human Feedback (L2DHF)*, a research framework for adaptive alert prioritisation. This repository currently contains a standalone confidence-threshold experiment using the CICIDS2017 intrusion-detection dataset. It is an academic prototype, not a production SOC tool.

## Repository contents

- **`Experiments_Confidence_Threshold_Reconstructed/`** — the experiment script, dependencies, methodology notes and saved result tables.
- **Original implementation** — the authors' notebooks and model artefacts remain in the [L2DHF_Final repository](https://github.com/cqs-technology/L2DHF_Final). We link to it rather than republishing its contents here.

The raw dataset is not included. Download `MachineLearningCSV.zip` from the [official CICIDS2017 dataset page](https://www.unb.ca/cic/datasets/ids-2017.html). PCAP packet captures are not required for this experiment.

## Experiment summary

The script cleans the CICIDS2017 CSV files, maps attack labels to reconstructed priority groups, samples up to 20,000 examples per group, fits MinMax scaling and a 12-component PCA transform, and trains four fresh classifiers: Random Forest, AdaBoost, XGBoost and Gaussian Naive Bayes. The ensemble uses majority voting. The additional policy routes an alert to analyst review when its winning vote share is below a selected threshold.

Results from the saved run:

| Vote-agreement threshold | Sent for analyst review | Automatic coverage | Accuracy on automatically processed alerts |
| ---: | ---: | ---: | ---: |
| 0.50 | 0.00% | 100.00% | 98.65% |
| 0.75 | 4.16% | 95.84% | 98.85% |
| 1.00 | 29.74% | 70.26% | 99.79% |

These figures come from a capped, class-stratified test sample and do not preserve the natural distribution of CICIDS2017 traffic. They illustrate a coverage-versus-accuracy trade-off in this run; they do not establish how the policy would perform in a live SOC. Vote agreement is not a calibrated probability, and the threshold policy has not been integrated with the original reinforcement-learning agent.

## Running the experiment

Use Python 3.10 or newer. Install the required libraries:

```bash
cd Experiments_Confidence_Threshold_Reconstructed
pip install -r requirements.txt
```

Then pass the path to your downloaded dataset archive:

```bash
python run_reconstructed_experiment.py --dataset-zip "/path/to/MachineLearningCSV.zip" --output-dir results
```

Replace the example path with the actual location of the ZIP file. The script writes metrics, run metadata, row-level predictions, and local model/preprocessing artefacts. Large generated model artefacts and row-level predictions are not tracked in this repository by default.

## Important limitation

This is a **reconstructed experiment**, not an exact reproduction of the original authors' model results. The supplied upstream repository archive contained Git LFS pointer files instead of the five processed CICIDS2017 CSVs, and the fitted PCA/preprocessing objects were unavailable. This experiment therefore retrains models on the public raw CSV files and reconstructs the severity mapping from public descriptions. The sampling approach also changes the natural class distribution. These limitations are documented in the experiment README and run metadata and should be included when the results are cited in reports or presentations.

## References

- Jalalvand, F., Baruwal Chhetri, M., Nepal, S., & Paris, C. *Adaptive alert prioritisation in security operations centres via learning to defer with human feedback*. [DOI](https://doi.org/10.1016/j.ins.2026.124128) · [arXiv preprint](https://arxiv.org/abs/2506.18462)
- Original implementation: [cqs-technology/L2DHF_Final](https://github.com/cqs-technology/L2DHF_Final)
- Dataset: [CICIDS2017, Canadian Institute for Cybersecurity](https://www.unb.ca/cic/datasets/ids-2017.html)

## Team

- Sainihal Somarajupalli
- Pratyush Srivastava
- Priyanshu Raman
- Anjali Kumari
- Sarthak Singh
