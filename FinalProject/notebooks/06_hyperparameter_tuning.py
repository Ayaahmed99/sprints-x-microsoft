# 06_hyperparameter_tuning.py
import pandas as pd
import numpy as np
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
import joblib
import os

df = pd.read_csv("data/heart_disease_cleaned.csv")
TARGET_COL = "target"
X = df.drop(columns=[TARGET_COL])
y = df[TARGET_COL]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

# Pipeline with RandomForest
rf_pipeline = Pipeline([('scaler', StandardScaler()), ('clf', RandomForestClassifier(random_state=42))])
rf_param_grid = {
    'clf__n_estimators': [100, 200, 400],
    'clf__max_depth': [None, 5, 10, 20],
    'clf__min_samples_split': [2, 5, 10],
    'clf__min_samples_leaf': [1, 2, 4]
}
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
rf_grid = GridSearchCV(rf_pipeline, rf_param_grid, cv=cv, scoring='roc_auc', n_jobs=-1, verbose=2)
rf_grid.fit(X_train, y_train)
print("RF best:", rf_grid.best_params_, "score:", rf_grid.best_score_)
joblib.dump(rf_grid.best_estimator_, "models/random_forest_best.pkl")

# SVM with randomized search
svm_pipeline = Pipeline([('scaler', StandardScaler()), ('clf', SVC(probability=True, random_state=42))])
svm_param_dist = {
    'clf__C': [0.1, 1, 10, 100],
    'clf__gamma': ['scale', 'auto'],
    'clf__kernel': ['rbf', 'poly']
}
svm_search = RandomizedSearchCV(svm_pipeline, svm_param_dist, n_iter=8, cv=cv, scoring='roc_auc', n_jobs=-1, random_state=42, verbose=2)
svm_search.fit(X_train, y_train)
print("SVM best:", svm_search.best_params_, "score:", svm_search.best_score_)
joblib.dump(svm_search.best_estimator_, "models/svm_best.pkl")
