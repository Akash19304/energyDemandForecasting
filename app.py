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
if 'model_results' not in st.session_state:
    st.session_state.model_results = {}
    logger.info("Initialized Streamlit session state.")

# --- UI ---
st.title("⚡ Energy Demand Backtesting Dashboard")
st.markdown("This dashboard trains one or more models on historical data and evaluates their performance on a hold-out test set.")

# --- Sidebar for User Inputs ---
with st.sidebar:
    st.header("🔑 API Key")
    eia_api_key = st.text_input("EIA API Key", type="password", help="Get a free key from https://www.eia.gov/opendata/register.php")
    
    st.header("⚙️ Backtest Configuration")
    
    max_date = datetime.now().date() - timedelta(days=1)
    end_date_default = max_date
    start_date_default = end_date_default - timedelta(days=365 * 2) # Default 2 years for a decent train/test split
    
    selected_dates = st.date_input(
        "Select Date Range for Backtest Data",
        (start_date_default, end_date_default),
        min_value=datetime(2018, 1, 1).date(),
        max_value=max_date,
        help="Select a date range ending yesterday at the latest."
    )
    start_date, end_date = selected_dates
    
    region = st.selectbox("Select Region", ['US48', 'CAL', 'TEX', 'NY', 'FLA'], help="The geographical region for demand data.")
    
    available_models = ['LightGBM', 'XGBoost', 'Random Forest']
    selected_models = st.multiselect("Select Models to Train", available_models, default=['LightGBM'])
    
    if st.button("🚀 Run Backtest", type="primary", use_container_width=True):
        if not eia_api_key:
            st.error("Please enter your EIA API key to proceed.")
        elif not selected_models:
            st.error("Please select at least one model to train.")
        else:
            st.session_state.model_results = {} # Clear previous results
            logger.info(f"Backtest started for models: {selected_models}")
            
            try:
                with st.spinner("Running backtest... This may take several minutes."):
                    st.info("Step 1/4: Fetching historical data...")
                    energy_df, weather_df, economic_df = load_all_data(start_date, end_date, region, eia_api_key)
                    
                    if energy_df.empty:
                        st.error("Failed to fetch energy data. Please check the date range and API key.")
                    else:
                        st.info("Step 2/4: Engineering features...")
                        final_df = engineer_features(energy_df, weather_df, economic_df)

                        st.info(f"Step 3/4: Preparing data and splitting...")
                        forecaster = EnergyForecaster()
                        X_train, X_test, y_train, y_test = forecaster.prepare_data_for_ml(final_df)
                        
                        st.session_state.y_test = y_test # Save for all models
                        st.session_state.energy_df = energy_df
                        st.session_state.weather_df = weather_df

                        for model_name in selected_models:
                            st.info(f"Training {model_name} model...")
                            model = forecaster.train_model(model_name, X_train, y_train)
                            
                            st.info(f"Evaluating {model_name} on test set...")
                            predictions = model.predict(X_test)
                            predictions = pd.Series(predictions, index=y_test.index)

                            # Store results for this model
                            st.session_state.model_results[model_name] = {
                                'predictions': predictions,
                                'metrics': ModelEvaluator.calculate_metrics(y_test, predictions),
                                'feature_importance': forecaster.feature_importance.get(model_name)
                            }
                        
                        st.success("Backtest completed successfully for all selected models!")
                        logger.info("Backtest pipeline completed successfully.")

            except Exception as e:
                logger.error("An error occurred during the backtest process.", exc_info=True)
                st.error(f"An error occurred: {e}")

# --- Main Area for Displaying Results ---
if not st.session_state.model_results:
    st.info("Please configure settings and click 'Run Backtest' in the sidebar.")
else:
    trained_models = list(st.session_state.model_results.keys())
    
    tab1, tab2, tab3, tab4 = st.tabs(["📊 Model Comparison", "📈 Single Model Performance", "📚 Data Explorer", "🤖 Model Internals"])
    
    with tab1:
        st.header("Overall Model Performance Comparison")
        
        # Display metrics in a table
        metrics_data = {model: results['metrics'] for model, results in st.session_state.model_results.items()}
        metrics_df = pd.DataFrame(metrics_data).T
        st.dataframe(metrics_df)

        # Display comparison plot
        st.subheader("Forecast vs. Actuals Comparison")
        predictions_dict = {model: results['predictions'] for model, results in st.session_state.model_results.items()}
        fig_comp = EnergyVisualizer.plot_predictions_comparison(st.session_state.y_test, predictions_dict)
        st.plotly_chart(fig_comp, use_container_width=True)

    with tab2:
        st.header("Detailed Performance of a Single Model")
        model_to_display = st.selectbox("Select a model to inspect", trained_models)
        
        if model_to_display:
            results = st.session_state.model_results[model_to_display]
            metrics = results['metrics']
            
            st.subheader(f"Metrics for {model_to_display}")
            col1, col2, col3, col4 = st.columns(4)
            col1.metric("MAE", f"{metrics.get('MAE', 0):.2f} MW", help="Mean Absolute Error")
            col2.metric("RMSE", f"{metrics.get('RMSE', 0):.2f} MW", help="Root Mean Squared Error")
            col3.metric("MAPE", f"{metrics.get('MAPE', 0):.2f} %", help="Mean Absolute Percentage Error")
            col4.metric("R² Score", f"{metrics.get('R2', 0):.3f}", help="R-squared")
            
            st.subheader("Forecast vs. Actuals on Test Set")
            fig_pred = EnergyVisualizer.plot_predictions(st.session_state.y_test, results['predictions'], title=f"{model_to_display} Test Set Predictions")
            st.plotly_chart(fig_pred, use_container_width=True)

    with tab3:
        st.header("Exploratory Data Analysis")
        st.markdown("Visualizing the raw input data used for the backtest.")
        if 'energy_df' in st.session_state and not st.session_state.energy_df.empty:
            st.subheader("Energy Demand (MW)")
            fig_demand = EnergyVisualizer.plot_time_series(st.session_state.energy_df, ['demand_mw'], "Energy Demand Over Time")
            st.plotly_chart(fig_demand, use_container_width=True)
        if 'weather_df' in st.session_state and st.session_state.weather_df is not None:
            st.subheader("Weather Data")
            fig_weather = EnergyVisualizer.plot_time_series(st.session_state.weather_df, st.session_state.weather_df.columns, "Weather Data Over Time")
            st.plotly_chart(fig_weather, use_container_width=True)

    with tab4:
        st.header("Model Internals")
        model_to_inspect_internals = st.selectbox("Select a model for internals", trained_models, key="internals_selector")

        if model_to_inspect_internals:
            results = st.session_state.model_results[model_to_inspect_internals]
            
            st.subheader(f"Feature Importance for {model_to_inspect_internals}")
            st.markdown("Top 20 most influential features for the model's predictions.")
            if results['feature_importance'] is not None:
                st.bar_chart(results['feature_importance'].head(20))
            else:
                st.warning("Feature importance not available for this model.")
            
            st.subheader("Actual vs. Predicted Scatter Plot")
            fig_scatter = EnergyVisualizer.plot_scatter(st.session_state.y_test, results['predictions'])
            st.plotly_chart(fig_scatter, use_container_width=True)
