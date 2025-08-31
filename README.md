# Energy Demand Forecasting

This project is a comprehensive energy demand forecasting application that utilizes various machine learning and time series models to predict electricity demand. It features a user-friendly web interface built with Streamlit for interactive backtesting and visualization of forecasting models.

## Features

- **Interactive Web Interface:** A Streamlit-based dashboard allows for easy configuration of backtests, model selection, and visualization of results.
- **Multiple Data Sources:** Integrates data from three real-world APIs:
    - **Energy Demand:** U.S. Energy Information Administration (EIA) for historical electricity demand.
    - **Weather Data:** Open-Meteo for historical weather information.
    - **Economic Indicators:** Yahoo Finance for financial data like oil and gas prices.
- **Advanced Feature Engineering:** Creates a rich feature set for models, including:
    - Temporal features (hour, day of week, month, etc.).
    - Lag and rolling window features.
    - Weather-derived features like heating/cooling degree hours.
- **Multiple Forecasting Models:** Implements and compares a suite of popular forecasting models:
    - LightGBM
    - XGBoost
    - Random Forest
    - SARIMAX
    - Exponential Smoothing
- **In-depth Model Evaluation:** Provides detailed performance metrics (MAE, RMSE, MAPE, R²) and visualizations to compare and evaluate models.
- **Robust Logging:** Includes a structured logging system to monitor the application and diagnose issues.

## Getting Started

### Prerequisites

- Python 3.8+
- An EIA API Key (get a free key from [https://www.eia.gov/opendata/register.php](https://www.eia.gov/opendata/register.php))

### Installation

1.  **Clone the repository:**
    ```bash
    git clone <repository-url>
    cd energy-demand-forecasting
    ```

2.  **Create and activate a virtual environment:**
    ```bash
    python -m venv .venv
    source .venv/bin/activate  # On Windows, use `.venv\Scripts\activate`
    ```

3.  **Install the required dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

### Running the Application

1.  **Launch the Streamlit app:**
    ```bash
    streamlit run app.py
    ```

2.  **Open your web browser** and navigate to the local URL provided by Streamlit (usually `http://localhost:8501`).

3.  **Enter your EIA API Key** in the sidebar, configure your desired backtest settings (date range, region, models), and click "Run Backtest".


## Data Sources

- **EIA (U.S. Energy Information Administration):** Provides hourly electricity demand data for various regions in the United States.
- **Open-Meteo:** A free, open-source weather API used to fetch historical hourly weather data (temperature, humidity, wind speed, etc.).
- **yfinance:** A Python library used to download historical market data from Yahoo Finance, providing economic indicators like oil prices (CL=F), natural gas prices (NG=F), and the VIX volatility index (^VIX).

## Models

The application supports the following forecasting models:

- **Machine Learning Models:**
    - **LightGBM:** A fast, distributed, high-performance gradient boosting framework.
    - **XGBoost:** An optimized distributed gradient boosting library designed to be highly efficient, flexible, and portable.
    - **Random Forest:** An ensemble learning method that operates by constructing a multitude of decision trees at training time.

- **Statistical Time Series Models:**
    - **SARIMAX (Seasonal AutoRegressive Integrated Moving Average with eXogenous factors):** A powerful statistical model for time series forecasting that supports seasonality and external variables.
    - **Exponential Smoothing:** A time series forecasting method for univariate data that can be extended to support data with a seasonal component.
