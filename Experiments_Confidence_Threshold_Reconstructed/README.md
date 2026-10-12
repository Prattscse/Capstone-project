# Reconstructed CICIDS2017 Confidence-Threshold Experiment

## Purpose

This experiment evaluates whether agreement among four classifiers can be used as a simple routing signal: alerts with lower agreement are flagged for analyst review, while the remaining alerts are eligible for automatic handling.

This is an add-on experiment inspired by the L2DHF project. It is **not** an exact reproduction of the original trained models or the complete reinforcement-learning agent.

## Why a reconstruction was needed

The supplied archive of the upstream repository contained Git LFS pointer files in place of the five processed CICIDS2017 CSVs. It also did not include the fitted PCA/preprocessing objects. The upstream classifiers expect the authors' 12-component PCA representation, so applying those original models directly to raw CICIDS2017 fields would not be a valid evaluation.

Instead, this script rebuilds a documented preprocessing pipeline using the public CICIDS2017 machine-learning CSV archive and trains fresh classifiers. The assumptions are listed below so that the results are interpreted correctly.

## Pipeline

1. Read the eight CSV files inside `MachineLearningCSV.zip` in chunks.
2. Clean header whitespace, omit the duplicate `Fwd Header Length.1` feature, convert input columns to numeric, and discard rows containing unusable numeric values.
3. Map attack labels into four priority groups using an explicit reconstruction of the CVSS family-to-severity mapping described in the project's references:
   - `0`: Normal / benign traffic.
   - `2`: Medium — DoS, PortScan, FTP/SSH Patator, web brute force/XSS, and Infiltration.
   - `3`: High — DDoS, Heartbleed, and SQL injection.
   - `4`: Critical — Bot.
4. Sample up to 20,000 examples per priority group using reservoir sampling. All available Critical/Bot examples were retained.
5. Split the sample into 75% training and 25% test sets, stratified by class. Fit MinMax scaling and 12-component PCA on the training features only.
6. Train Random Forest, AdaBoost, XGBoost, and Gaussian Naive Bayes classifiers.
7. Combine the four predictions using majority voting. If the maximum vote count is tied, the implementation uses the first tied prediction in the row, matching the first-seen tie behavior intended by the original notebook.
8. Evaluate vote-agreement thresholds of 0.50, 0.75, and 1.00. Agreement is the winning vote count divided by four; it is **not a calibrated probability**. Predictions below the selected threshold are marked for analyst review.

## Results from the saved run

The saved test sample contains 15,489 records: 5,000 Normal, 5,000 Medium, 5,000 High, and 489 Critical.

| Agreement threshold | Deferred for review | Automatic coverage | Accuracy on automatically processed records |
| ---: | ---: | ---: | ---: |
| 0.50 | 0.00% | 100.00% | 98.65% |
| 0.75 | 4.16% | 95.84% | 98.85% |
| 1.00 | 29.74% | 70.26% | 99.79% |

The overall majority-vote accuracy on this test sample was 98.65%, with macro-F1 of 0.9857. Twelve PCA components retained 96.73% of training-sample variance.

These values are from a capped, class-stratified test sample. Its class proportions are not representative of natural network traffic, so overall accuracy should not be read as expected production accuracy. The metrics demonstrate a coverage-versus-accuracy trade-off in this run; they do not establish an improvement in real analyst outcomes.

## Reproducibility

Requirements are listed in `requirements.txt`. Download `MachineLearningCSV.zip` from the [official CICIDS2017 page](https://www.unb.ca/cic/datasets/ids-2017.html), then run:

```bash
pip install -r requirements.txt
python run_reconstructed_experiment.py --dataset-zip "/path/to/MachineLearningCSV.zip" --output-dir results
```

The path is an example; replace it with the actual location of the ZIP. The script creates result CSVs, run metadata, row-level predictions and local model/preprocessing artefacts. The row-level predictions and generated model files are deliberately not committed by default.

## Limitations

- This is a reconstructed pipeline and is not a faithful reproduction of the authors' original model results.
- The missing fitted PCA transform and exact per-row CVSS annotations mean the reconstructed features/labels may differ from those used in the original paper.
- The capped, class-stratified sampling process changes the natural distribution of priorities.
- Vote agreement is a simple ensemble agreement score, not calibrated confidence.
- The threshold policy is evaluated separately and is not integrated with the L2DHF reinforcement-learning agent.
- The evaluation does not model actual analyst feedback, time cost, or downstream incident outcomes.

## References

- Jalalvand, F., Baruwal Chhetri, M., Nepal, S., & Paris, C. *Adaptive alert prioritisation in security operations centres via learning to defer with human feedback*. [DOI](https://doi.org/10.1016/j.ins.2026.124128) · [arXiv preprint](https://arxiv.org/abs/2506.18462)
- [Original L2DHF implementation](https://github.com/cqs-technology/L2DHF_Final)
- [CICIDS2017 dataset](https://www.unb.ca/cic/datasets/ids-2017.html)
