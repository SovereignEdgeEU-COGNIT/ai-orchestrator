# ML Model Repository 1.0

This branch contains the first version of the ML Model Repository for the AI-Orchestrator module. It includes foundational models used in the orchestration system.

## Models Included
- **idec**: A clustering model used for unsupervised data categorization.
- **mc2PCA**: A principal component analysis model for dimensionality reduction in large datasets.

## Version Information
- **Version**: 1.0
- **Models**: idec, mc2PCA

These models form the basis of the AI-enabled orchestrator and are intended for initial deployment and resource optimization tasks.


## Test and Deployment
To test and deploy the models, run the following command:
```bash
python test.py
```
torch model accepts torch tensor as input

Requires modifying row 128 in MTS_utils.py to point towards the dataset, e.g. https://www.kaggle.com/datasets/gauravdhamane/gwa-bitbrains (assuming format of bitbrains dataset)

