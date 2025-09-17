# Save complete pipeline with preprocessing + final model
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
import joblib

final_pipeline = Pipeline([('scaler', StandardScaler()), ('clf', RandomForestClassifier(**rf_grid.best_params_['clf__params_here']))])
# In practice, use rf_grid.best_estimator_ (which is already a pipeline)
joblib.dump(rf_grid.best_estimator_, "models/final_pipeline.pkl")
