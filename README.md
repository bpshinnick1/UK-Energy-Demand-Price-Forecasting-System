# UK-Energy-Demand-Price-Forecasting-System
UK Energy Demand & Price Forecasting System
Overview

This project builds an end-to-end analytics and forecasting system for the UK electricity market, integrating historical demand, wholesale power prices, and weather data to support market analysis, capacity planning, and scenario-based decision making.

The objective is to replicate how energy analysts, system planners, and infrastructure investors analyse power systems by:

Structuring large time-series datasets in a relational data model

Building forecasting models for electricity demand and price behaviour

Quantifying uncertainty, seasonality, and weather sensitivity

Creating an interactive executive dashboard for planning and risk assessment

The project combines SQL data modelling, Python time-series forecasting, and Power BI visual analytics in a production-style workflow.

Key Questions This Project Answers

How does UK electricity demand vary by season, temperature, and calendar effects?

How accurately can short- and medium-term demand be forecast using historical patterns and weather features?

How volatile are wholesale electricity prices, and how do they respond to demand spikes and weather shocks?

What scenarios (e.g. cold spells, heatwaves, demand surges) pose the highest system stress and price risk?

Data Sources

National Grid / NESO – Daily historical electricity demand

Elexon BMRS – UK wholesale electricity market price series

Met Office – Daily weather observations (temperature, rainfall)

UK Public Holidays & Calendar Data – Seasonality and demand drivers

All data is processed and stored in a structured PostgreSQL star schema to enable scalable analysis and modelling.

Technical Stack

PostgreSQL
Star schema data warehouse (fact tables for demand and prices, dimension tables for time and weather)

Python (Pandas, NumPy, Statsmodels, Prophet / XGBoost)
Data cleaning, feature engineering, time-series forecasting, model evaluation

Power BI (DAX, What-If Parameters, Forecast Visuals)
Executive dashboards, scenario analysis, capacity and price risk visualisation

GitHub
Version control, documentation, and portfolio presentation

Project Architecture
/data_raw        -> Original datasets (demand, price, weather)
/data_processed -> Cleaned & feature-engineered tables
/sql             -> Star schema, views, transformations
/src             -> Python forecasting and evaluation scripts
/notebooks       -> EDA and modelling experiments
/powerbi         -> Power BI dashboard files and screenshots
/docs            -> Data dictionary, assumptions, methodology

Modelling Approach
Demand Forecasting

Baseline: seasonal naive / rolling averages

Statistical: ARIMA / Prophet

Machine learning: XGBoost regression with weather and lag features

Price Analysis

Trend and volatility decomposition

Demand-price sensitivity

Forecasting with confidence intervals

Stress testing under demand and temperature shocks

Scenario & What-If Analysis

Temperature deviation scenarios (cold spells / heatwaves)

Demand surge scenarios

Impact on peak load, price levels, and volatility

Dashboard Outputs

The final Power BI report will include:

Market Overview

Actual demand and price trends

Volatility and seasonality

Forecasting & Accuracy

Forecast vs actual

Error metrics (MAE, MAPE)

Confidence bands

Scenario & Stress Testing

What-if sliders for temperature and demand shocks

Impact on peak demand and price risk

Planning and capacity stress indicators

Learning Objectives

This project demonstrates:

Advanced SQL data modelling for time-series analytics

End-to-end forecasting pipelines in Python

Business-focused scenario analysis

Executive-level dashboard design

Energy market and infrastructure domain knowledge
