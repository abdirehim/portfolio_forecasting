Time Series Forecasting for Portfolio Management Optimization

Business Objective

Guide Me in Finance (GMF) Investments aims to enhance portfolio performance by leveraging time series forecasting to predict market trends, optimize asset allocation, and minimize risks. The goal is to provide clients with tailored investment strategies using data-driven insights, focusing on three assets: Tesla (TSLA), Vanguard Total Bond Market ETF (BND), and S&P 500 ETF (SPY).

Situational Overview (Business Need)

As a Financial Analyst at GMF Investments, your role is to:





Extract historical financial data from YFinance (July 1, 2015, to July 31, 2025).



Preprocess and analyze data to identify trends and patterns.



Develop time series forecasting models (ARIMA/SARIMA and LSTM) to predict market movements.



Recommend portfolio adjustments based on forecasts to optimize returns and manage risks.

Data

Source: YFinance API
Period: July 1, 2015, to July 31, 2025
Assets:





TSLA: High-growth, high-volatility stock (consumer discretionary, automobile manufacturing).



BND: Bond ETF for stability and low risk.



SPY: S&P 500 ETF for diversified, moderate-risk market exposure.
Data Fields: Date, Open, High, Low, Close, Adj Close, Volume.

Expected Outcomes

Skills





API Usage: Fetch data using YFinance.



Data Wrangling: Clean and preprocess data using pandas.



Feature Engineering: Calculate returns, volatility, and other metrics.



Data Scaling: Normalize/standardize data for modeling.



Statistical Modeling: Build and optimize ARIMA/SARIMA models.



Deep Learning: Develop and evaluate LSTM models.



Model Evaluation: Use MAE, RMSE, MAPE for performance comparison.



Optimization & Visualization: Generate and plot the Efficient Frontier.



MPT Implementation: Use PyPortfolioOpt for portfolio optimization.



Simulation: Backtest portfolio performance.



Professional Communication: Present findings clearly.

Knowledge





Characteristics of asset classes (TSLA, BND, SPY).



Efficient Market Hypothesis and its implications.



Stationarity and its role in ARIMA modeling.



Efficient Frontier and portfolio optimization.



Importance of backtesting and benchmarking.

Abilities





Critically evaluate modeling approaches.



Frame business objectives and synthesize insights.



Make data-driven portfolio recommendations.





Instructions

Task 1: Preprocess and Explore the Data





Fetch Data:





Use YFinance to extract historical data for TSLA, BND, and SPY (July 1, 2015, to July 31, 2025).



Ensure data includes Date, Open, High, Low, Close, Adj Close, and Volume.



Data Cleaning:





Check data types and handle missing values (fill, interpolate, or remove).



Normalize/scale data if needed for machine learning models.



Verify data integrity (e.g., no duplicate dates).



Exploratory Data Analysis (EDA):





Visualize closing prices to identify trends.



Calculate and plot daily percentage changes to assess volatility.



Compute rolling means and standard deviations for short-term trends.



Perform outlier detection for significant anomalies.



Analyze days with high/low returns.



Stationarity and Trends:





Conduct Augmented Dickey-Fuller (ADF) test on closing prices and daily returns.



Discuss implications of stationarity for ARIMA modeling (differencing if non-stationary).



Calculate foundational risk metrics (e.g., Value at Risk, Sharpe Ratio).



Volatility Analysis:





Compute rolling means and standard deviations.



Document insights (e.g., TSLA’s price direction, return fluctuations, risk metrics).





Task 2: Develop Time Series Forecasting Models





Model Selection:





Implement ARIMA/SARIMA (using statsmodels/pmdarima) and LSTM (using TensorFlow/Keras).



Compare trade-offs in complexity, performance, and interpretability.



Data Splitting:





Split data chronologically: train (2015–2023), test (2024–2025).



Avoid random shuffling to preserve temporal order.



Model Training:





For ARIMA: Use grid search or auto_arima to optimize (p, d, q) parameters.



For LSTM: Experiment with layers, neurons, epochs, and batch size.



Train models on training data.



Model Evaluation:





Forecast on test set and compare using MAE, RMSE, and MAPE.



Discuss which model performs better and why.

Task 3: Forecast Future Market Trends





Generate Forecast:





Use the best model from Task 2 to forecast TSLA prices for 6–12 months.



Include confidence intervals for predictions.



Forecast Analysis:





Visualize forecast alongside historical data.



Analyze long-term trends (upward, downward, stable) and anomalies.



Evaluate confidence interval width and its implications for reliability.



Market Insights:





Identify opportunities (e.g., price increases) and risks (e.g., high volatility).



Discuss forecast uncertainty and its impact on long-term reliability.

Task 4: Optimize Portfolio Based on Forecast





Expected Returns:





Use TSLA forecast from Task 3 as expected return.



Use historical average daily returns (annualized) for BND and SPY.



Covariance Matrix:





Compute based on historical daily returns of TSLA, BND, and SPY.



Portfolio Optimization:





Use PyPortfolioOpt to generate the Efficient Frontier.



Plot volatility (x-axis) vs. return (y-axis).



Mark the Maximum Sharpe Ratio and Minimum Volatility portfolios.



Portfolio Recommendation:





Select an optimal portfolio and justify the choice (e.g., risk-adjusted return vs. low risk).



Provide optimal weights, expected return, volatility, and Sharpe Ratio.

Task 5: Strategy Backtesting





Backtesting Period:





Use August 1, 2024, to July 31, 2025, for backtesting.



Benchmark:





Define a static 60% SPY / 40% BND portfolio.



Simulate Strategy:





Start with optimal weights from Task 4.



Hold portfolio for one month or rebalance minimally.



Simulate performance over the backtesting period.



Performance Analysis:





Plot cumulative returns of strategy vs. benchmark.



Calculate Sharpe Ratio and total return for both.



Summarize whether the strategy outperformed the benchmark and its implications.

Tutorials Schedule





Case Discussion: Use #all-week11 to ask questions before August 6, 2025.



Interim Solution Review: Feedback session post-submission on August 10, 2025.



Final Submission Support: Office hours with tutors before August 12, 2025.

Submission





Interim Solution: Submit by 20:00 UTC, August 10, 2025 (preliminary analysis and models).



Final Submission: Submit by 20:00 UTC, August 12, 2025 (complete report, code, visualizations, and recommendations).



Include code, visualizations, and a report summarizing findings, model performance, portfolio recommendations, and backtesting results.

References





YFinance Documentation: https://pypi.org/project/yfinance/



Statsilibre: https://www.statsmodels.org/stable/



PyPortfolioOpt: https://pyportfolioopt.readthedocs.io/



TensorFlow/Keras: https://www.tensorflow.org/



Modern Portfolio Theory: Markowitz, H. (1952). "Portfolio Selection." Journal of Finance.