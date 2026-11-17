# Statistical Feature Selection for Classification

This project uses the Breast Cancer Wisconsin dataset from scikit-learn to compare statistical feature-selection methods and classification pipelines.

## Notebook

Open `Statistical_Feature_Selection_Breast_Cancer_Final.ipynb` in Jupyter or Google Colab. The notebook installs its Python dependencies in the first section and walks through:

- Zero-variance filtering, Fisher scores, and correlation-based redundancy removal
- Logistic Regression, Decision Tree, and K-Nearest Neighbors comparisons
- Repeated cross-validation, with the held-out test set reserved for final evaluation
- A Gradio dashboard for reviewing results and trying individual or CSV predictions

The final notebook section can save a trained pipeline and result files when run. The notebook itself was not run as part of adding it to this repository.

## Responsible use

This is an educational machine-learning demonstration, not a clinically validated diagnostic system. Do not use it for medical decisions.
