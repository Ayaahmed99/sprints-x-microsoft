# 01_data_preprocessing.py
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
import joblib
import os

DATA_PATH = "data/heart_disease.csv"
CLEANED_PATH = "data/heart_disease_cleaned.csv"
os.makedirs("data", exist_ok=True)

# Load dataset
column_names = [
    'age','sex','cp','trestbps','chol','fbs','restecg',
    'thalach','exang','oldpeak','slope','ca','thal','num'
]
df = pd.read_csv(DATA_PATH, header=None, names=column_names, na_values='?')
print("Initial shape:", df.shape)
print(df.head())

print(df.info())
print(df.describe(include='all'))
print("Missing counts:\n", df.isnull().sum())

TARGET_COL = "num"
df[TARGET_COL] = df[TARGET_COL].apply(lambda x: 1 if x > 0 else 0)

num_cols = df.select_dtypes(include=["number"]).columns.tolist()
num_cols.remove(TARGET_COL)
cat_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()

num_imputer = SimpleImputer(strategy="median")
cat_imputer = SimpleImputer(strategy="most_frequent")

df[num_cols] = pd.DataFrame(num_imputer.fit_transform(df[num_cols]), columns=num_cols)
if cat_cols:
    df[cat_cols] = pd.DataFrame(cat_imputer.fit_transform(df[cat_cols]), columns=cat_cols)

preprocessor = ColumnTransformer(
    transformers=[
        ("num", StandardScaler(), num_cols),
        ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), cat_cols)
    ],
    remainder="passthrough"
)

X = df.drop(columns=[TARGET_COL])
y = df[TARGET_COL].astype(int)

preprocessor.fit(X)

plt.figure(figsize=(12,10))
sns.heatmap(df[num_cols + [TARGET_COL]].corr(), annot=True, fmt=".2f", cmap="coolwarm")
plt.title("Correlation matrix")
plt.tight_layout()
plt.show()

df[num_cols].hist(figsize=(12,8))
plt.tight_layout()
plt.show()

plt.figure(figsize=(12,6))
sns.boxplot(data=df[num_cols])
plt.xticks(rotation=45)
plt.title("Boxplots of numeric features")
plt.show()

df.to_csv(CLEANED_PATH, index=False)
print("Saved cleaned csv to", CLEANED_PATH)
