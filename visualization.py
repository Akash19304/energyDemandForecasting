import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
from logger import logger # Import the configured logger

# =============================================================================
# 4. VISUALIZATION FUNCTIONS
# =============================================================================

class EnergyVisualizer:
    """Visualization functions for energy forecasting"""

    @staticmethod
    def plot_time_series(df, columns=None, title="Time Series Data"):
        """Plot time series data using Plotly."""
        logger.info(f"Generating time series plot titled: '{title}' for columns: {columns}")
        if columns is None:
            columns = df.columns
        
        fig = make_subplots(rows=len(columns), cols=1, shared_xaxes=True, subplot_titles=columns)
        for i, col in enumerate(columns):
            fig.add_trace(go.Scatter(x=df.index, y=df[col], name=col, mode='lines'), row=i+1, col=1)
        
        fig.update_layout(height=250*len(columns), title_text=title, showlegend=False)
        return fig

    @staticmethod
    def plot_predictions(y_true, y_pred, title="Predictions vs Actual"):
        """Plot predictions against actual values."""
        logger.info(f"Generating predictions vs. actual plot titled: '{title}'")
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=y_true.index, y=y_true, name="Actual", mode='lines', line=dict(color='royalblue')))
        fig.add_trace(go.Scatter(x=y_true.index, y=y_pred, name="Predicted", mode='lines', line=dict(color='crimson', dash='dash')))
        fig.update_layout(
            title_text=title,
            xaxis_title="Date",
            yaxis_title="Demand (MW)",
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
        )
        return fig

    @staticmethod
    def plot_scatter(y_true, y_pred, title="Scatter Plot: Actual vs. Predicted"):
        """Create a scatter plot of actual vs predicted values."""
        logger.info(f"Generating scatter plot titled: '{title}'")
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=y_true, y=y_pred, mode='markers',
            marker=dict(color='rgba(66, 139, 202, 0.6)', size=5),
            name="Prediction"
        ))
        min_val, max_val = min(y_true.min(), y_pred.min()), max(y_true.max(), y_pred.max())
        fig.add_trace(go.Scatter(
            x=[min_val, max_val], y=[min_val, max_val],
            mode='lines', name="Perfect Prediction", line=dict(dash='dash', color='green')
        ))
        fig.update_layout(
            title_text=title,
            xaxis_title="Actual Demand (MW)",
            yaxis_title="Predicted Demand (MW)"
        )
        return fig
