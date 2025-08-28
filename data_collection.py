import pandas as pd
import numpy as np
import requests
import yfinance as yf
from datetime import datetime, timedelta
import time
import warnings
from logger import logger

warnings.filterwarnings('ignore')

# =============================================================================
# 1. DATA COLLECTION (with Real APIs)
# =============================================================================

class EnergyDataCollector:
    """Collect energy demand data from EIA (US Energy Information Administration)"""

    def __init__(self, api_key):
        if not api_key or api_key == "YOUR_EIA_API_KEY":
            raise ValueError("A valid EIA API key is required.")
        self.api_key = api_key
        self.base_url = "https://api.eia.gov/v2/"

    def get_electricity_demand(self, start_date, end_date, region='US48'):
        """
        Get hourly electricity demand data from the EIA API, handling pagination.
        region options: 'US48', 'CAL', 'TEX', 'NY', 'FLA'
        """
        logger.info(f"Fetching real energy demand data for {region} from {start_date} to {end_date}")
        
        all_data = []
        offset = 0
        length = 5000  # Max records per call

        headers = {'Accept': 'application/json'}
        url = f"{self.base_url}electricity/rto/region-data/data/"
        
        while True:
            params = {
                'api_key': self.api_key,
                'frequency': 'hourly',
                'data[0]': 'value',
                'facets[respondent][]': region,
                'facets[type][]': 'D',  # D = Demand
                'start': start_date,
                'end': end_date,
                'sort[0][column]': 'period',
                'sort[0][direction]': 'asc',
                'offset': offset,
                'length': length
            }
            
            try:
                response = requests.get(url, params=params, headers=headers)
                response.raise_for_status()
                data = response.json()
                
                page_data = data.get('response', {}).get('data', [])
                if not page_data:
                    logger.info("No more data found from EIA API. Ending pagination.")
                    break
                
                all_data.extend(page_data)
                logger.info(f"Fetched {len(page_data)} records from EIA. Total so far: {len(all_data)}.")
                
                if len(page_data) < length:
                    break
                
                offset += length
                time.sleep(0.2)

            except requests.exceptions.RequestException as e:
                logger.error(f"Error fetching EIA data: {e}", exc_info=True)
                if e.response:
                    logger.error(f"EIA API Response Text: {e.response.text}")
                raise

        if not all_data:
            logger.warning("No energy data was returned from the EIA API for the selected range.")
            return pd.DataFrame(columns=['demand_mw'])

        df = pd.DataFrame(all_data)
        df['datetime'] = pd.to_datetime(df['period'])
        df = df.set_index('datetime')
        df['demand_mw'] = pd.to_numeric(df['value'])
        logger.info(f"Successfully fetched and processed {len(df)} total records from EIA.")
        return df[['demand_mw']].sort_index()

class WeatherDataCollector:
    """Collect weather data using the free Open-Meteo API (no API key required)."""

    def get_weather_data(self, start_date, end_date, lat=39.8283, lon=-98.5795):
        """Get historical weather data from Open-Meteo."""
        logger.info(f"Fetching real weather data from Open-Meteo from {start_date} to {end_date}")

        url = "https://archive-api.open-meteo.com/v1/archive"
        params = {
            "latitude": lat,
            "longitude": lon,
            "start_date": start_date,
            "end_date": end_date,
            "hourly": "temperature_2m,relativehumidity_2m,windspeed_10m,shortwave_radiation",
            "timezone": "auto"
        }

        try:
            response = requests.get(url, params=params)
            response.raise_for_status()
            data = response.json()

            df = pd.DataFrame(data['hourly'])
            df['datetime'] = pd.to_datetime(df['time'])
            df = df.set_index('datetime')
            df = df.rename(columns={
                'temperature_2m': 'temperature',
                'relativehumidity_2m': 'humidity',
                'windspeed_10m': 'wind_speed',
                'shortwave_radiation': 'solar_radiation'
            })
            logger.info(f"Successfully fetched {len(df)} records from Open-Meteo.")
            return df[['temperature', 'humidity', 'wind_speed', 'solar_radiation']]

        except requests.exceptions.RequestException as e:
            logger.error(f"Error fetching Open-Meteo data: {e}", exc_info=True)
            if e.response:
                logger.error(f"Open-Meteo API Response Text: {e.response.text}")
            raise

class EconomicDataCollector:
    """Collect economic indicators using yfinance"""

    def get_economic_data(self, start_date, end_date):
        """Get economic indicators from yfinance."""
        logger.info(f"Fetching economic data from yfinance for dates {start_date} to {end_date}")
        try:
            # Download data for all tickers at once
            tickers = "CL=F NG=F ^VIX"
            data = yf.download(tickers, start=start_date, end=end_date, interval='1d', progress=False)
            
            # Check if data is empty
            if data.empty:
                logger.warning("yfinance returned no economic data for the selected date range.")
                # Return an empty DataFrame with the correct columns to prevent downstream errors
                return pd.DataFrame(columns=['oil_price', 'gas_price', 'vix'])

            # Select the 'Close' prices and rename columns
            economic_daily = data['Close']
            economic_daily = economic_daily.rename(columns={'CL=F': 'oil_price', 'NG=F': 'gas_price', '^VIX': 'vix'})
            
            # Forward fill missing values and resample to hourly
            economic_daily = economic_daily.fillna(method='ffill')
            economic_hourly = economic_daily.resample('H').ffill()
            
            logger.info("Successfully fetched and processed economic data from yfinance.")
            return economic_hourly
        except Exception as e:
            logger.error(f"An error occurred while fetching data from yfinance: {e}", exc_info=True)
            raise
