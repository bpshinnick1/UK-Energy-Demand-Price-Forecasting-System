⚡ UK Energy Demand & Price Forecasting System
An end-to-end analytics and forecasting system for the UK electricity market, combining SQL data warehousing, Python machine learning, and Power BI dashboards to support strategic planning and risk assessment.

📊 Project Overview
This project replicates the analytical workflows used by energy market analysts, grid operators, and infrastructure investors to forecast electricity demand, predict wholesale price movements, and stress-test system capacity under various scenarios.
By integrating historical demand data, wholesale power prices, and weather observations into a structured data warehouse, this system enables:

Short and medium-term demand forecasting with quantified uncertainty
Price volatility analysis and demand-price sensitivity modelling
Scenario planning for extreme weather events and demand surges
Executive dashboards for strategic decision-making


🎯 Key Business Questions
QuestionAnalytical ApproachHow does UK electricity demand vary seasonally and respond to weather?Time-series decomposition, correlation analysis, feature engineeringCan we accurately forecast demand 7-30 days ahead?ARIMA, Prophet, XGBoost regression with weather and lag featuresHow volatile are wholesale prices, and what drives price spikes?Volatility decomposition, demand-price sensitivity analysisWhat scenarios pose the highest system stress and financial risk?Monte Carlo simulation, stress testing with what-if parameters

🗂️ Data Sources
SourceData TypePurposeNational Grid ESO / NESODaily electricity demand (MW)Historical demand patterns, seasonality analysisElexon BMRSWholesale electricity prices (£/MWh)Price volatility, demand-price relationshipsMet OfficeTemperature, rainfall, wind speedWeather feature engineering, demand driversUK Calendar DataPublic holidays, working daysCalendar effect adjustments
All data is processed into a PostgreSQL star schema for scalable analytics and modelling.

🏗️ Technical Architecture
uk-energy-forecasting/
│
├── data/
│   ├── raw/                    # Original source data
│   ├── processed/              # Cleaned, feature-engineered tables
│   └── data_dictionary.md      # Schema documentation
│
├── sql/
│   ├── schema.sql              # Star schema definition
│   ├── transformations/        # ETL and feature engineering
│   └── views/                  # Analytical views
│
├── src/
│   ├── data_pipeline.py        # Data ingestion and cleaning
│   ├── feature_engineering.py  # Time-series features, weather integration
│   ├── forecasting/
│   │   ├── demand_models.py    # ARIMA, Prophet, XGBoost implementations
│   │   └── price_models.py     # Price forecasting and volatility
│   └── evaluation.py           # Model validation, accuracy metrics
│
├── notebooks/
│   ├── 01_eda.ipynb           # Exploratory data analysis
│   ├── 02_demand_modelling.ipynb
│   └── 03_price_analysis.ipynb
│
├── powerbi/
│   ├── UK_Energy_Dashboard.pbix
│   └── screenshots/            # Dashboard exports for portfolio
│
└── docs/
    ├── methodology.md          # Modelling approach and assumptions
    └── results_summary.md      # Key findings and insights

🔧 Technical Stack
Data Warehousing

PostgreSQL – Star schema design with fact tables (demand, prices) and dimension tables (time, weather)
SQL – Complex transformations, window functions, aggregations

Forecasting & Machine Learning

Python 3.10+ – Core language
Pandas & NumPy – Data manipulation and feature engineering
Statsmodels – ARIMA, seasonal decomposition
Prophet – Time-series forecasting with holidays and seasonality
XGBoost – Gradient boosting regression with weather features

Visualization & Reporting

Power BI Desktop – Interactive dashboards with DAX measures
What-If Parameters – Scenario analysis and stress testing
Matplotlib & Seaborn – Exploratory visualizations


📈 Modelling Approach
1. Demand Forecasting Pipeline
Model TypeTechniqueUse CaseBaselineSeasonal naive, rolling averagesBenchmark performanceStatisticalARIMA, ProphetCapturing trend and seasonalityMachine LearningXGBoost with lag features + weatherHighest accuracy for short-term forecasts
Key Features:

Rolling window validation (walk-forward)
Temperature, humidity, wind speed integration
Calendar effects (holidays, weekends, daylight hours)
Lag features (1, 7, 14, 28 days)

2. Price Analysis & Volatility

Decomposition: Trend, seasonal, and residual components
Demand-Price Elasticity: Correlation analysis and regression modelling
Confidence Intervals: Quantile forecasting for risk assessment

3. Scenario & Stress Testing

Cold Spell Scenario: Temperature 5-10°C below seasonal average
Heatwave Scenario: Temperature 5-10°C above average
Demand Surge: 10-20% increase in baseline load
Output Metrics: Peak load impact, price spike risk, capacity margins


📊 Power BI Dashboard
The final deliverable includes an executive dashboard with three core views:
1. Market Overview

Historical demand and price trends
Volatility indicators (rolling standard deviation)
Seasonality patterns and year-over-year comparisons

2. Forecasting & Accuracy

7-day and 30-day demand forecasts
Forecast vs actual comparison with confidence bands
Model performance metrics (MAE, MAPE, RMSE)

3. Scenario Planning

What-If Sliders: Temperature deviation, demand shock magnitude
Impact Analysis: Peak load, price risk, system stress indicators
Risk Heatmaps: Probability x impact matrices
