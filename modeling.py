import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import xgboost as xgb
import lightgbm as lgb
from statsmodels.tsa.api import ExponentialSmoothing
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.statespace.sarimax import SARIMAXResults
from statsmodels.tsa.holtwinters import HoltWintersResults
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

    def prepare_data_for_time_series(self, df, target_col='demand_mw', test_size=0.2):
        logger.info("Preparing data for time series models...")
        
        df.index = pd.to_datetime(df.index)
        df = df.asfreq('D')
        df = df.fillna(method='ffill')

        self.trained_columns = [col for col in df.columns if col != target_col]
        X = df[self.trained_columns]
        y = df[target_col]

        split_idx = int(len(df) * (1 - test_size))
        X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
        y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]
        
        logger.info(f"Data prepared for time series models. Train size: {len(y_train)}, Test size: {len(y_test)}")
        return X_train, X_test, y_train, y_test

    def train_model(self, model_name, X_train, y_train):
        if model_name == 'Random Forest':
            model = RandomForestRegressor(n_estimators=100, max_depth=20, random_state=42, n_jobs=-1)
            model.fit(X_train, y_train)
        elif model_name == 'XGBoost':
            model = xgb.XGBRegressor(n_estimators=500, max_depth=8, learning_rate=0.05, subsample=0.8, colsample_bytree=0.8, random_state=42)
            model.fit(X_train, y_train)
        elif model_name == 'LightGBM':
            model = lgb.LGBMRegressor(n_estimators=500, max_depth=8, learning_rate=0.05, subsample=0.8, colsample_bytree=0.8, random_state=42, verbose=-1)
            model.fit(X_train, y_train)
        elif model_name == 'SARIMAX':
            # For SARIMAX, y_train is the endogenous variable, X_train are the exogenous variables
            model = SARIMAX(y_train, exog=X_train, order=(1, 1, 1), seasonal_order=(1, 1, 1, 7)).fit(disp=False)
        elif model_name == 'Exponential Smoothing':
            # Exponential Smoothing is a univariate model, so it only uses y_train
            model = ExponentialSmoothing(y_train, seasonal='add', seasonal_periods=7).fit()
        else:
            logger.error(f"Unsupported model name provided: {model_name}")
            raise ValueError("Unsupported model name")

        logger.info(f"Starting training for {model_name} model...")
        
        logger.info(f"Training for {model_name} complete.")
        self.models[model_name] = model
        self.trained_model = model
        if hasattr(model, 'feature_importances_'):
            self.feature_importance[model_name] = pd.Series(model.feature_importances_, index=X_train.columns).sort_values(ascending=False)
        return model

    def predict_future(self, historical_df, future_weather_df, latest_economic_df, days_to_forecast):
        """Generates a forecast for future dates."""
        if self.trained_model is None:
            raise Exception("Model has not been trained yet.")

        logger.info(f"Starting future prediction for {days_to_forecast} days.")
        
        # Check the type of the trained model to decide the prediction strategy
        model_type = type(self.trained_model)

        if isinstance(self.trained_model, (RandomForestRegressor, xgb.XGBRegressor, lgb.LGBMRegressor)):
            # Machine learning models require feature engineering for future dates
            future_features_df = self.feature_engineer.engineer_features_for_future(
                historical_df, 
                future_weather_df, 
                latest_economic_df
            )
            future_features_df = future_features_df[self.trained_columns]
            scaler = self.scalers['feature_scaler']
            future_features_scaled = pd.DataFrame(
                scaler.transform(future_features_df), 
                index=future_features_df.index, 
                columns=future_features_df.columns
            )
            future_predictions = self.trained_model.predict(future_features_scaled)
            future_forecast_series = pd.Series(future_predictions, index=future_features_df.index, name="forecast")

        elif isinstance(self.trained_model, SARIMAXResults):
            # SARIMAX model prediction
            future_features_df = self.feature_engineer.engineer_features_for_future(
                historical_df, 
                future_weather_df, 
                latest_economic_df
            )
            future_features_df = future_features_df[self.trained_columns]
            future_predictions = self.trained_model.predict(start=future_features_df.index[0], end=future_features_df.index[-1], exog=future_features_df)
            future_forecast_series = pd.Series(future_predictions, name="forecast")

        elif isinstance(self.trained_model, HoltWintersResults):
            # Exponential Smoothing model prediction
            future_predictions = self.trained_model.forecast(steps=days_to_forecast)
            future_forecast_series = pd.Series(future_predictions, name="forecast")
        
        else:
            raise Exception(f"Unsupported model type for future prediction: {model_type}")


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
