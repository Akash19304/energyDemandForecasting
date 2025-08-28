import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import xgboost as xgb
import lightgbm as lgb
from logger import logger # Import the configured logger

# =============================================================================
# 3. MODEL IMPLEMENTATIONS & EVALUATION
# =============================================================================

class EnergyForecaster:
    """A wrapper for various forecasting models."""

    def __init__(self):
        self.models = {}
        self.scalers = {}
        self.feature_importance = {}

    def prepare_data_for_ml(self, df, target_col='demand_mw', test_size=0.2):
        logger.info("Preparing data for machine learning models...")
        feature_cols = [col for col in df.columns if col != target_col]
        X = df[feature_cols]
        y = df[target_col]

        split_idx = int(len(df) * (1 - test_size))
        X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
        y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]
        logger.info(f"Data split into training and testing sets. Train size: {len(X_train)}, Test size: {len(X_test)}")

        scaler = StandardScaler()
        X_train_scaled = pd.DataFrame(scaler.fit_transform(X_train), index=X_train.index, columns=X_train.columns)
        X_test_scaled = pd.DataFrame(scaler.transform(X_test), index=X_test.index, columns=X_test.columns)
        
        self.scalers['feature_scaler'] = scaler
        logger.info("Feature scaling complete.")
        return X_train_scaled, X_test_scaled, y_train, y_test

    def train_model(self, model_name, X_train, y_train):
        if model_name == 'Random Forest':
            model = RandomForestRegressor(n_estimators=100, max_depth=20, random_state=42, n_jobs=-1)
        elif model_name == 'XGBoost':
            model = xgb.XGBRegressor(n_estimators=500, max_depth=8, learning_rate=0.05, subsample=0.8, colsample_bytree=0.8, random_state=42, early_stopping_rounds=30)
        elif model_name == 'LightGBM':
            model = lgb.LGBMRegressor(n_estimators=500, max_depth=8, learning_rate=0.05, subsample=0.8, colsample_bytree=0.8, random_state=42, early_stopping_rounds=30, verbose=-1)
        else:
            logger.error(f"Unsupported model name provided: {model_name}")
            raise ValueError("Unsupported model name")

        logger.info(f"Starting training for {model_name} model...")
        if model_name in ['XGBoost', 'LightGBM']:
            val_size = int(len(X_train) * 0.15)
            X_train_part, X_val_part = X_train.iloc[:-val_size], X_train.iloc[-val_size:]
            y_train_part, y_val_part = y_train.iloc[:-val_size], y_train.iloc[-val_size:]
            model.fit(X_train_part, y_train_part, eval_set=[(X_val_part, y_val_part)])
        else:
            model.fit(X_train, y_train)
        
        logger.info(f"Training for {model_name} complete.")
        self.models[model_name] = model
        self.feature_importance[model_name] = pd.Series(model.feature_importances_, index=X_train.columns).sort_values(ascending=False)
        return model

class ModelEvaluator:
    """Comprehensive model evaluation."""

    @staticmethod
    def calculate_metrics(y_true, y_pred):
        logger.info("Calculating model performance metrics.")
        metrics = {
            'MAE': mean_absolute_error(y_true, y_pred),
            'RMSE': np.sqrt(mean_squared_error(y_true, y_pred)),
            'MAPE': np.mean(np.abs((y_true - y_pred) / y_true)) * 100,
            'R2': r2_score(y_true, y_pred),
        }
        logger.info(f"Metrics calculated: {metrics}")
        return metrics
