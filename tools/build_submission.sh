#!/bin/sh
# Rebuild ML_Assignment2_Submission.pdf (run from the repo root; needs reportlab).
PY=${PY:-python}
S=screenshots
$PY tools/make_submission_pdf.py README.md ML_Assignment2_Submission.pdf \
  --repo https://github.com/hk9454/assignment \
  --app https://assignment-22jabdambijh7uumb6hvsw.streamlit.app/ \
  --bits-id 2026AK05703 \
  --screenshot $S/lab_04.jpg \
  --screenshot-caption "Figure 1. BITS Virtual Lab (user s2026ak05703): <font name='Courier'>python -m src.train</font> completed in VS Code — all five models trained and evaluated on the test split." \
  --extra \
  "$S/lab_01.jpg::Conda environment ml2 (Python 3.11) on the lab machine — dependencies from requirements.txt installed." \
  "$S/lab_02.jpg::Training started on the lab machine (python -m src.train); four models done, SVM in progress." \
  "$S/lab_03.jpg::Re-run of the training script after an earlier interrupted run; project files and saved models visible in the Explorer." \
  "$S/lab_05.jpg::Streamlit app running on the lab machine (localhost:8501) with the bundled test_data.csv." \
  "$S/lab_06.jpg::App sidebar: CSV upload, test-data toggle, classifier dropdown and decision-threshold slider." \
  "$S/lab_07.jpg::Evaluation metrics for the selected model (Logistic Regression): Accuracy, AUC, Precision, Recall, F1, MCC." \
  "$S/lab_08.jpg::Preview of the uploaded test data (features and target)." \
  "$S/lab_09.jpg::Confusion matrix and ROC curve for the selected model." \
  "$S/lab_10.jpg::Classification report for the selected model." \
  "$S/lab_11.jpg::Compare-all-models tab: metrics of all five models, best value per metric highlighted." \
  "$S/lab_12.jpg::Metric comparison chart; best model by MCC is the Decision Tree." \
  "$S/lab_13.jpg::BITS Virtual Lab desktop of user s2026ak05703."
