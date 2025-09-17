# 01_data_preprocessing.py (or notebook cells)
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, MinMaxScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import joblib
import os

# Paths
DATA_PATH = "data/heart_disease.csv"
CLEANED_PATH = "data/heart_disease_cleaned.csv"
os.makedirs("data", exist_ok=True)

# 1. Load dataset
df = pd.read_csv(DATA_PATH)
print("Initial shape:", df.shape)
display(df.head())

# 2. Quick overview
print(df.info())
print(df.describe(include='all'))

# Inspect missing values
print("Missing counts:\n", df.isnull().sum())

# If target column differs, rename accordingly
# For safety, check for common target names:
for name in ["target", "Target", "disease", "HD", "Diagnosis", "heart_disease"]:
    if name in df.columns:
        TARGET_COL = name
        break
else:
    TARGET_COL = "target"  # fallback - user should adjust if needed

print("Using target column:", TARGET_COL)

# 3. Handle missing values
# Strategy:
# - numerical: median
# - categorical: most frequent
num_cols = df.select_dtypes(include=["number"]).columns.tolist()
if TARGET_COL in num_cols:
    num_cols.remove(TARGET_COL)
cat_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()

num_imputer = SimpleImputer(strategy="median")
cat_imputer = SimpleImputer(strategy="most_frequent")

df[num_cols] = pd.DataFrame(num_imputer.fit_transform(df[num_cols]), columns=num_cols)
df[cat_cols] = pd.DataFrame(cat_imputer.fit_transform(df[cat_cols]), columns=cat_cols)

# 4. Encoding: decide which categorical columns need one-hot encoded
# If small number of categories -> OneHot, else Label encoding or target encoding (not implemented)
# We'll do OneHot for all cat_cols
preprocessor = ColumnTransformer(
    transformers=[
        ("num", StandardScaler(), num_cols),
        ("cat", OneHotEncoder(handle_unknown="ignore", sparse=False), cat_cols)
    ],
    remainder="passthrough"  # keep any other columns
)

# create X and y
X = df.drop(columns=[TARGET_COL])
y = df[TARGET_COL].astype(int)  # ensure integer labels

# Fit the preprocessor to save it for pipeline later (optional)
preprocessor.fit(X)

# For quick EDA: correlation and basic plots
plt.figure(figsize=(12,10))
sns.heatmap(df.corr(), annot=True, fmt=".2f", cmap="coolwarm")
plt.title("Correlation matrix")
plt.tight_layout()
plt.show()

# Histograms for numeric columns
df[num_cols].hist(figsize=(12,8))
plt.tight_layout()
plt.show()

# Boxplots
plt.figure(figsize=(12,6))
sns.boxplot(data=df[num_cols])
plt.xticks(rotation=45)
plt.title("Boxplots of numeric features")
plt.show()

# Save cleaned raw dataframe (non-transformed)
df.to_csv(CLEANED_PATH, index=False)
print("Saved cleaned csv to", CLEANED_PATH)
