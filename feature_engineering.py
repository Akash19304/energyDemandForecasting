import pandas as pd
import numpy as np
from logger import logger # Import the configured logger

# =============================================================================
# 2. FEATURE ENGINEERING
# =============================================================================

class FeatureEngineer:
    """Advanced feature engineering for energy forecasting"""

    def create_temporal_features(self, df):
        df = df.copy()
        df['hour'] = df.index.hour
        df['day_of_week'] = df.index.dayofweek
        df['month'] = df.index.month
        df['day_of_year'] = df.index.dayofyear
        df['week_of_year'] = df.index.isocalendar().week.astype(int)
        df['is_weekend'] = (df.index.dayofweek >= 5).astype(int)
        df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
        df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
        df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
        df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)
        return df

    def create_lag_features(self, df, target_col='demand_mw', lags=[24, 48, 168]):
        df = df.copy()
        for lag in lags:
            df[f'{target_col}_lag_{lag}'] = df[target_col].shift(lag)
        return df

    def create_rolling_features(self, df, target_col='demand_mw', windows=[24, 168]):
        df = df.copy()
        for window in windows:
            df[f'{target_col}_roll_mean_{window}'] = df[target_col].shift(1).rolling(window).mean()
            df[f'{target_col}_roll_std_{window}'] = df[target_col].shift(1).rolling(window).std()
        return df

    def create_weather_features(self, df):
        df = df.copy()
        if 'temperature' in df.columns:
            df['heating_degree_hours'] = np.maximum(18 - df['temperature'], 0)
            df['cooling_degree_hours'] = np.maximum(df['temperature'] - 22, 0)
        if 'humidity' in df.columns and 'temperature' in df.columns:
            T = df['temperature']
            RH = df['humidity']
            df['heat_index'] = -8.78469475556 + 1.61139411 * T + 2.33854883889 * RH - 0.14611605 * T * RH
        return df

    def engineer_all_features(self, energy_df, weather_df, economic_df):
        logger.info("Starting feature engineering process...")
        df = energy_df.copy()
        if weather_df is not None:
            df = df.join(weather_df, how='left')
        if economic_df is not None:
            df = df.join(economic_df, how='left')

        df = df.fillna(method='ffill').fillna(method='bfill')
        
        logger.info("Creating temporal, lag, rolling, and weather features.")
        df = self.create_temporal_features(df)
        df = self.create_lag_features(df, 'demand_mw')
        df = self.create_rolling_features(df, 'demand_mw')
        df = self.create_weather_features(df)
        
        initial_rows = len(df)
        df = df.dropna()
        final_rows = len(df)
        logger.info(f"Dropped {initial_rows - final_rows} rows with NaN values after feature creation.")
        logger.info("Feature engineering complete.")
        return df
