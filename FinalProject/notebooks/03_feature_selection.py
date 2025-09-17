# 03_feature_selection.py
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import RFE, SelectFromModel, chi2
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

df = pd.read_csv("data/heart_disease_cleaned.csv")
TARGET_COL = "target"
X = df.drop(columns=[TARGET_COL])
y = df[TARGET_COL]

# If any categorical columns exist, convert to numeric representation for chi2; here we MinMax scale numerical features
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X.select_dtypes(include=[np.number]))
X_num_cols = X.select_dtypes(include=[np.number]).columns.tolist()

# 1) Random Forest feature importances
rf = RandomForestClassifier(n_estimators=200, random_state=42)
rf.fit(X, y)
importances = rf.feature_importances_
feat_imp = pd.Series(importances, index=X.columns).sort_values(ascending=False)
print(feat_imp[:20])

plt.figure(figsize=(10,6))
sns.barplot(x=feat_imp.values[:20], y=feat_imp.index[:20])
plt.title("Top 20 Feature Importances (Random Forest)")
plt.show()

# 2) RFE with Logistic Regression wrapper (or RF)
from sklearn.linear_model import LogisticRegression
lr = LogisticRegression(max_iter=1000, solver='liblinear')
# select top 10
rfe = RFE(estimator=lr, n_features_to_select=10, step=1)
rfe.fit(X, y)
rfe_support = pd.Series(rfe.support_, index=X.columns)
print("RFE selected features:")
print(list(X.columns[rfe_support]))

# 3) Chi-square test (works on non-negative integers)
# We'll run chi2 on minmax-scaled numeric features
from sklearn.feature_selection import SelectKBest
chi2_selector = SelectKBest(score_func=chi2, k='all')
chi2_selector.fit(X_scaled, y)
chi2_scores = pd.Series(chi2_selector.scores_, index=X_num_cols).sort_values(ascending=False)
print(chi2_scores.head(20))

# Combine methods: choose features that appear in majority of methods
selected_by_rf = set(feat_imp[feat_imp > np.quantile(feat_imp,0.5)].index)
selected_by_rfe = set(X.columns[rfe_support])
selected_by_chi = set(chi2_scores.index[:10])
final_selected = list((selected_by_rf | selected_by_rfe | selected_by_chi))
print("Final selected features (union approach):", final_selected)

# Save top feature list
pd.Series(final_selected).to_csv("results/selected_features.csv", index=False)
