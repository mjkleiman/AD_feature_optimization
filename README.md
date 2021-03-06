# Screening for early-stage Alzheimer’s disease using optimized feature sets and machine learning

Michael J. Kleiman*, Elan Barenholtz, and James E. Galvin, for the Alzheimer's Disease Neuroimaging Initiative**

\* Contributed to code

\** All data used in the preparation of this code were obtained from the Alzheimer's Disease Neuroimaging Initiative. A list of the features used can be found in the file **List_of_Features.md**

#### Please cite our preprint/code:
[![DOI](https://img.shields.io/badge/medRxiv-10.1101%2F2020.10.28.20212027-blue)](https://doi.org/10.1101/2020.10.28.20212027)

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.4081704.svg)](https://doi.org/10.5281/zenodo.4081704)

## Layout
Code is located in the **code** directory, arranged in the order they are used to generate outputs. The exception is the splitrepeat.py file, which is a custom-created high stochasticity cross-validation procedure, outlined in an upcoming publication.

Generated feature sets and pairs of feature sets can be found in **Selected_Sets.md**

Model outputs are found in the **models** directory, and contain sensitivity, specificity, PPV, NPV, accuracy, and F1 scores for each of the model iterations. The statistical analyses of these outputs are found in the code directory in the "5-Stats" notebooks.

Note that in file names, "Imp" or "Impaired" implies the two-class "Impaired vs non-impaired" classification method, while "multiclass" or "MC" implies the three-class "CDR 0 vs CDR 0.5 vs CDR 1". "MCN" refers to a "MultiClassifier Network", a method of using two or more separate classifiers each with different featuresets or with different hyperparameters and optimized to target a single class or group of classes. More detail can be found in the attached publication, or within a separate upcoming publication.

