import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import xgboost as xgb
import lightgbm as lgb
from logger import logger # Import the configured logger
from feature_engineering import FeatureEngineer

# =============================================================================
# 3. MODEL IMPLEMENTATIONS & EVALUATION
# =============================================================================

class EnergyForecaster:
    """A wrapper for various forecasting models."""

    def __init__(self):
        self.models = {}
        self.scalers = {}
        self.feature_importance = {}
        self.trained_model = None
        self.feature_engineer = FeatureEngineer()
        self.trained_columns = []

    def prepare_data_for_ml(self, df, target_col='demand_mw', test_size=0.2):
        logger.info("Preparing data for machine learning models...")
        self.trained_columns = [col for col in df.columns if col != target_col]
        X = df[self.trained_columns]
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
            model = xgb.XGBRegressor(n_estimators=500, max_depth=8, learning_rate=0.05, subsample=0.8, colsample_bytree=0.8, random_state=42)
        elif model_name == 'LightGBM':
            model = lgb.LGBMRegressor(n_estimators=500, max_depth=8, learning_rate=0.05, subsample=0.8, colsample_bytree=0.8, random_state=42, verbose=-1)
        else:
            logger.error(f"Unsupported model name provided: {model_name}")
            raise ValueError("Unsupported model name")

        logger.info(f"Starting training for {model_name} model...")
        model.fit(X_train, y_train)
        
        logger.info(f"Training for {model_name} complete.")
        self.models[model_name] = model
        self.trained_model = model
        self.feature_importance[model_name] = pd.Series(model.feature_importances_, index=X_train.columns).sort_values(ascending=False)
        return model

    def predict_future(self, historical_df, future_weather_df, latest_economic_df, days_to_forecast):
        """Generates a forecast for future dates."""
        if self.trained_model is None:
            raise Exception("Model has not been trained yet.")

        logger.info(f"Starting future prediction for {days_to_forecast} days.")
        
        # Engineer features for the future period
        future_features_df = self.feature_engineer.engineer_features_for_future(
            historical_df, 
            future_weather_df, 
            latest_economic_df
        )
        
        # Ensure columns are in the same order as during training
        future_features_df = future_features_df[self.trained_columns]

        # Scale the features
        scaler = self.scalers['feature_scaler']
        future_features_scaled = pd.DataFrame(
            scaler.transform(future_features_df), 
            index=future_features_df.index, 
            columns=future_features_df.columns
        )

        # Make predictions
        future_predictions = self.trained_model.predict(future_features_scaled)
        future_forecast_series = pd.Series(future_predictions, index=future_features_df.index, name="forecast")

        logger.info("Future prediction complete.")
        return future_forecast_series

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
