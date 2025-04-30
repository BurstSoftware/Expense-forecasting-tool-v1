import streamlit as st
import pandas as pd
from prophet import Prophet
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta

# Streamlit app configuration
st.set_page_config(page_title="Expense Forecasting Tool", layout="wide")
st.title("Expense Forecasting Tool")
st.write("Predict future business expenses based on historical data and planned activities.")

# Sidebar for user inputs
st.sidebar.header("Settings")
uploaded_file = st.sidebar.file_uploader("Upload Historical Expenses (CSV)", type=["csv"])
forecast_period = st.sidebar.slider("Forecast Period (Months)", 1, 24, 12)
budget_limit = st.sidebar.number_input("Monthly Budget Limit ($)", min_value=0.0, value=10000.0, step=100.0)
growth_scenario = st.sidebar.selectbox("Growth Scenario", ["Baseline", "High Growth (+20%)", "Low Growth (-20%)"])

# Function to load and preprocess data
def load_data(file):
    if file is not None:
        df = pd.read_csv(file)
        # Ensure correct column names and types
        df['date'] = pd.to_datetime(df['date'])
        df['amount'] = df['amount'].astype(float)
        return df
    return None

# Function to prepare data for Prophet
def prepare_prophet_data(df, category):
    df_category = df[df['category'] == category][['date', 'amount']].rename(columns={'date': 'ds', 'amount': 'y'})
    return df_category

# Function to forecast expenses
def forecast_expenses(df, category, periods):
    prophet_df = prepare_prophet_data(df, category)
    if len(prophet_df) < 2:
        return None
    model = Prophet(yearly_seasonality=True, weekly_seasonality=False, daily_seasonality=False)
    model.fit(prophet_df)
    future = model.make_future_dataframe(periods=periods, freq='M')
    forecast = model.predict(future)
    return forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']]

# Function to adjust forecast for growth scenarios
def adjust_forecast(forecast, scenario):
    if scenario == "High Growth (+20%)":
        forecast['yhat'] *= 1.2
        forecast['yhat_lower'] *= 1.2
        forecast['yhat_upper'] *= 1.2
    elif scenario == "Low Growth (-20%)":
        forecast['yhat'] *= 0.8
        forecast['yhat_lower'] *= 0.8
        forecast['yhat_upper'] *= 0.8
    return forecast

# Main app logic
data = load_data(uploaded_file)

if data is not None:
    # Display data preview
    st.subheader("Historical Expenses")
    st.dataframe(data.head())

    # Get unique categories
    categories = data['category'].unique()
    selected_category = st.selectbox("Select Expense Category", categories)

    # Forecast for selected category
    periods = forecast_period * 30  # Convert months to days for Prophet
    forecast = forecast_expenses(data, selected_category, periods)

    if forecast is not None:
        # Adjust forecast based on growth scenario
        forecast = adjust_forecast(forecast, growth_scenario)

        # Plot historical and forecasted expenses
        st.subheader(f"Expense Trend for {selected_category}")
        fig = go.Figure()
        # Historical data
        historical = prepare_prophet_data(data, selected_category)
        fig.add_trace(go.Scatter(x=historical['ds'], y=historical['y'], mode='lines+markers', name='Historical'))
        # Forecast
        fig.add_trace(go.Scatter(x=forecast['ds'], y=forecast['yhat'], mode='lines', name='Forecast'))
        fig.add_trace(go.Scatter(x=forecast['ds'], y=forecast['yhat_upper'], fill='tonexty', mode='none', name='Upper Bound', opacity=0.2))
        fig.add_trace(go.Scatter(x=forecast['ds'], y=forecast['yhat_lower'], fill='tonexty', mode='none', name='Lower Bound', opacity=0.2))
        fig.update_layout(title=f"{selected_category} Expense Forecast", xaxis_title="Date", yaxis_title="Amount ($)")
        st.plotly_chart(fig, use_container_width=True)

        # Budget overrun alerts
        st.subheader("Budget Alerts")
        forecast_monthly = forecast.resample('M', on='ds')['yhat'].sum().reset_index()
        for _, row in forecast_monthly.iterrows():
            if row['yhat'] > budget_limit:
                st.warning(f"Budget overrun detected for {row['ds'].strftime('%B %Y')}: ${row['yhat']:,.2f} exceeds limit of ${budget_limit:,.2f}")

        # Scenario analysis inputs
        st.subheader("Scenario Analysis")
        variable_cost = st.number_input("Adjust Variable Cost ($)", min_value=0.0, value=1000.0, step=100.0)
        if st.button("Apply Variable Cost"):
            forecast['yhat'] += variable_cost
            st.success(f"Added ${variable_cost:,.2f} to forecast for scenario analysis.")
            # Replot with adjusted forecast
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=historical['ds'], y=historical['y'], mode='lines+markers', name='Historical'))
            fig.add_trace(go.Scatter(x=forecast['ds'], y=forecast['yhat'], mode='lines', name='Adjusted Forecast'))
            fig.update_layout(title=f"{selected_category} Expense Forecast (Adjusted)", xaxis_title="Date", yaxis_title="Amount ($)")
            st.plotly_chart(fig, use_container_width=True)

else:
    st.info("Please upload a CSV file with historical expense data (columns: date, category, amount).")

# Footer
st.markdown("---")
st.write("Built with Streamlit, Pandas, Prophet, and Plotly. Enhances financial planning by forecasting expenses.")
