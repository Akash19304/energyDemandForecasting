import streamlit as st
import pandas as pd
from datetime import datetime, timedelta

# Import your refactored modules
from data_collection import EnergyDataCollector, WeatherDataCollector, EconomicDataCollector
from feature_engineering import FeatureEngineer
from modeling import EnergyForecaster, ModelEvaluator
from visualization import EnergyVisualizer
from logger import logger

# --- Page Configuration ---
st.set_page_config(
    page_title="Energy Demand Forecaster",
    page_icon="⚡",
    layout="wide"
)

# --- Caching Functions for Performance ---
@st.cache_data
def load_all_data(start_date, end_date, region, eia_api_key):
    """Loads all data sources using real APIs."""
    logger.info(f"Cache miss. Loading all data for region '{region}' from {start_date} to {end_date}.")
    
    energy_collector = EnergyDataCollector(api_key=eia_api_key)
    energy_df = energy_collector.get_electricity_demand(start_date.strftime('%Y-%m-%d'), end_date.strftime('%Y-%m-%d'), region)
    
    # Weather collector no longer needs an API key
    weather_collector = WeatherDataCollector()
    weather_df = weather_collector.get_weather_data(start_date.strftime('%Y-%m-%d'), end_date.strftime('%Y-%m-%d'))
    
    economic_collector = EconomicDataCollector()
    economic_df = economic_collector.get_economic_data(start_date.strftime('%Y-%m-%d'), end_date.strftime('%Y-%m-%d'))
    
    return energy_df, weather_df, economic_df

@st.cache_data
def engineer_features(_energy_df, _weather_df, _economic_df):
    """Engineers features and caches the result."""
    logger.info("Cache miss. Engineering features.")
    feature_engineer = FeatureEngineer()
    final_df = feature_engineer.engineer_all_features(_energy_df, _weather_df, _economic_df)
    return final_df

# Use st.session_state to store model and results
if 'model_trained' not in st.session_state:
    st.session_state.model_trained = False
    st.session_state.metrics = {}
    logger.info("Initialized Streamlit session state.")

# --- UI ---
st.title("⚡ Real-Data Energy Demand Forecaster")
st.markdown("This dashboard uses live APIs to fetch and forecast electricity demand.")

# --- Sidebar for User Inputs ---
with st.sidebar:
    st.header("🔑 API Key")
    eia_api_key = st.text_input("EIA API Key", type="password", help="Get a free key from https://www.eia.gov/opendata/register.php")
    
    st.header("⚙️ Configuration")
    
    end_date = datetime.now().date()
    start_date = end_date - timedelta(days=365) # Default 1 year for faster API calls
    selected_dates = st.date_input(
        "Select Date Range for Training Data",
        (start_date, end_date),
        min_value=datetime(2018, 1, 1).date(),
        max_value=datetime.now().date(),
        help="Warning: Very large date ranges can be slow."
    )
    start_date, end_date = selected_dates
    
    region = st.selectbox("Select Region", ['US48', 'CAL', 'TEX', 'NY', 'FLA'], help="The geographical region for demand data.")
    
    model_name = st.selectbox("Select Model", ['LightGBM', 'XGBoost', 'Random Forest'], help="LightGBM is recommended for speed and accuracy.")
    
    if st.button("🚀 Generate Forecast", type="primary", use_container_width=True):
        if not eia_api_key:
            st.error("Please enter your EIA API key to proceed.")
        else:
            st.session_state.model_trained = False
            logger.info(f"Forecast generation started with params: region={region}, model={model_name}")
            
            try:
                with st.spinner("Processing... This may take several minutes depending on the date range."):
                    # 1. Load Data
                    st.info("Step 1/4: Fetching data from live APIs...")
                    energy_df, weather_df, economic_df = load_all_data(start_date, end_date, region, eia_api_key)
                    
                    if energy_df.empty:
                        st.error("Failed to fetch energy data. Please check the date range and API key.")
                    else:
                        # 2. Engineer Features
                        st.info("Step 2/4: Engineering features...")
                        final_df = engineer_features(energy_df, weather_df, economic_df)

                        # 3. Prepare data and Train Model
                        st.info(f"Step 3/4: Training {model_name} model...")
                        forecaster = EnergyForecaster()
                        X_train, X_test, y_train, y_test = forecaster.prepare_data_for_ml(final_df)
                        
                        model = forecaster.train_model(model_name, X_train, y_train)
                        
                        # 4. Make predictions
                        st.info("Step 4/4: Generating predictions...")
                        predictions = model.predict(X_test)
                        predictions = pd.Series(predictions, index=y_test.index)

                        # Store results in session state
                        st.session_state.model_trained = True
                        st.session_state.predictions = predictions
                        st.session_state.y_test = y_test
                        st.session_state.feature_importance = forecaster.feature_importance.get(model_name)
                        st.session_state.metrics = ModelEvaluator.calculate_metrics(y_test, predictions)
                        st.session_state.model_name = model_name
                        st.session_state.energy_df = energy_df
                        st.session_state.weather_df = weather_df
                        st.success("Forecast generated successfully!")
                        logger.info("Full forecast pipeline completed successfully.")

            except Exception as e:
                logger.error("An error occurred during the forecast generation process.", exc_info=True)
                st.error(f"An error occurred: {e}")

# --- Main Area for Displaying Results ---
if not st.session_state.model_trained:
    st.info("Please enter your EIA API key and configure the settings in the sidebar, then click 'Generate Forecast'.")
else:
    # Display tabs with results
    tab1, tab2, tab3 = st.tabs(["📊 Forecast Dashboard", "📈 Exploratory Data Analysis", "🤖 Model Performance"])
    with tab1:
        st.header(f"{st.session_state.model_name} Forecast vs. Actuals")
        metrics = st.session_state.metrics
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("MAE", f"{metrics.get('MAE', 0):.2f} MW", help="Mean Absolute Error")
        col2.metric("RMSE", f"{metrics.get('RMSE', 0):.2f} MW", help="Root Mean Squared Error")
        col3.metric("MAPE", f"{metrics.get('MAPE', 0):.2f} %", help="Mean Absolute Percentage Error")
        col4.metric("R² Score", f"{metrics.get('R2', 0):.3f}", help="R-squared")
        fig_pred = EnergyVisualizer.plot_predictions(st.session_state.y_test, st.session_state.predictions, title=f"Test Set Predictions")
        st.plotly_chart(fig_pred, use_container_width=True)

    with tab2:
        st.header("Exploratory Data Analysis")
        st.markdown("Visualizing the raw input data used for training.")
        if not st.session_state.energy_df.empty:
            st.subheader("Energy Demand (MW)")
            fig_demand = EnergyVisualizer.plot_time_series(st.session_state.energy_df, ['demand_mw'], "Energy Demand Over Time")
            st.plotly_chart(fig_demand, use_container_width=True)
        if st.session_state.weather_df is not None and not st.session_state.weather_df.empty:
            st.subheader("Weather Data")
            fig_weather = EnergyVisualizer.plot_time_series(st.session_state.weather_df, st.session_state.weather_df.columns, "Weather Data Over Time")
            st.plotly_chart(fig_weather, use_container_width=True)

    with tab3:
        st.header("Model Performance Deep Dive")
        st.subheader("Feature Importance")
        st.markdown("Top 20 most influential features for the model's predictions.")
        if st.session_state.feature_importance is not None:
            st.bar_chart(st.session_state.feature_importance.head(20))
        else:
            st.warning("Feature importance not available for this model.")
        st.subheader("Actual vs. Predicted Scatter Plot")
        fig_scatter = EnergyVisualizer.plot_scatter(st.session_state.y_test, st.session_state.predictions)
        st.plotly_chart(fig_scatter, use_container_width=True)
