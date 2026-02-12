# Import Packages
import numpy as np
import pandas as pd
import statsmodels.api as sm
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score, KFold
from sklearn.utils import resample
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import LabelEncoder
import itertools
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import matplotlib.pyplot as plt
from scipy.stats import probplot
from sklearn.model_selection import LeaveOneOut
from sklearn.linear_model import Ridge
from sklearn.linear_model import Lasso
from itertools import combinations
# Load Data
df = pd.read_csv("MTA_Subway_Customer_Journey-Focused_Metrics__Beginning_2020.csv")
label_encoder = LabelEncoder()
# Run LabelEncoder() on categorical data
df['period'] = label_encoder.fit_transform(df['period'])
df['line'] = label_encoder.fit_transform(df['line'])
df['division'] = label_encoder.fit_transform(df['division'])
# Sort by month (maybe delete this)
df['month'] = pd.to_datetime(df['month'])
df = df.sort_values(by='month')
df = df.drop(columns=['month'])
# Create train and test data sets
# Created a copy dataframe just for easier testing early on
df2 = df
X = df2.drop(['customer journey time performance', 'over_five_mins', 'over_five_mins_perc'], axis = 1)
y = df['customer journey time performance']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print(X_train.head())
# Function to create correlation plot
def CorrelationPlot(df):
    corr = df.corr()
    return corr.style.background_gradient(cmap='coolwarm')
# Get correlation plot
CorrelationPlot(X)
# Function to run regression on every possible subset of regressors and get model stats
def EverySubset(X_fun, y_fun):
    # Initialize list of results
    results = []
    predictors = X.columns
    # Iterate through each possible regression and add to results
    for k in range(1, len(predictors) + 1):
        for subset in itertools.combinations(predictors, k):
            X_subset = X[list(subset)]
            X_subset = sm.add_constant(X_subset)
            model = sm.OLS(y, X_subset).fit()
            
            # Append to results the info we want
            results.append({
                'Subset': subset,
                'AIC': model.aic,
                'Adjusted R2': model.rsquared_adj,
                'Coefficients': model.params.to_dict()
            })
    return pd.DataFrame(results)
# Function to get all models with Adjusted R2 above defined value
def BestSubsets(df, cutoff):
    return df[df['Adjusted R2'] > cutoff]
# Get all subset regressions with Adjusted R2 greater than 80%
# Isolate model with highest adjusted R2 in possible_models variable
results_df = EverySubset(X_train, y_train)
results_df = results_df.sort_values(by='Adjusted R2', ascending=False)
possible_models = BestSubsets(results_df, cutoff = .8)
# Print best model regressors based on R2
print(possible_models['Subset'].iloc[0])
# Standard Regression
possible_models =  possible_models.sort_values(by='Adjusted R2')
subset_regressors = possible_models['Subset'].iloc[1]
## Function to run regression based on best available subset of regressors
# train: Use X_train
# test: Use X_test
# y_train: use y_train
# x: use subset_regressors
def SubsetRegression(train, test, y_train, x):
    train_subset = train[list(x)]
    test_subset = test[list(x)]
    model_new = LinearRegression()
    # Fit model based on subset of regressors
    model_new.fit(train_subset, y_train)
    r_squared = model_new.score(train_subset, y_train)
    # Run prediction on test data
    y_pred = model_new.predict(test_subset)
    # Get model statistics
    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    AIC = len(y_train) * np.log(mse) + 2 * X_train.shape[1]
    BIC = len(y_train) * np.log(mse) + np.log(len(y_train)) * X_train.shape[1]
    # r_squared_2 = r2_score(y_test, y_pred)
    
    #output = [r_squared, mae, mse, rmse]
    #output = [round(elem, 4) for elem in output]
    output = [model_new, y_pred, {'R_Squared': round(r_squared, 4),
                                  'Mean_Absolute_Error': round(mae, 4),
                                  'Mean_Squared_Error': round(mse, 4), 'Root_MSE': round(float(rmse),4),
                                  'AIC': round(float(AIC), 4),
                                  'BIC': round(float(BIC), 4)
                                  }]
    return output
# Run subset regression
subset_model = SubsetRegression(X_train, X_test, y_train, subset_regressors)
print(subset_model[2])
# Cross Validation
new_model = subset_model[0]
train_subset = X_train[list(subset_regressors)]
cv_score = cross_val_score(new_model, train_subset, y_train, cv=10, scoring='neg_mean_squared_error')
print('Cross-Validation Score: ', np.mean(cv_score))
# Bootstrap .632 estimate
boot_scores = []
for _ in range(100):
    X_boot, y_boot = resample(train_subset, y_train)
    new_model.fit(X_boot, y_boot)
    boot_scores.append(mean_squared_error(y_train, new_model.predict(train_subset)))
boot632 = np.mean(boot_scores)
print('Bootstrap .632 Estimate: ', boot632)
# Residual Analysis
y_pred = subset_model[1]
residuals = y_test - y_pred
# Function for a basic residual plot
def BasicResidualPlot(test, predictions):
    res = test - predictions
    plt.scatter(predictions, res)
    plt.xlabel("Predicted Values")
    plt.ylabel("Residuals")
    plt.title("Residual Plot")
    plt.axhline(y=0, color='r', linestyle='--')
    plt.show()
# Function for a normal probability plot
def NormalProbabilityPlot(test, predictions):
    res = test - predictions
    probplot(res, dist="norm", plot=plt)
    plt.title("Normal Probability Plot of Residuals")
    plt.show()
# Get both residuals plots
BasicResidualPlot(y_test, y_pred)
NormalProbabilityPlot(y_test, y_pred)
# Ridge Regression Analysis
rdg = Ridge(alpha = .1)
ridge_model = rdg.fit(X_train[list(subset_regressors)], y_train)
ridge_pred = ridge_model.predict(X_test[list(subset_regressors)])
ridge_r2 = r2_score(y_test, ridge_pred)
ridge_sigma2 = np.var(y_test - ridge_pred, ddof=1)
ridge_ll = -len(ridge_pred) / 2 * np.log(2 * np.pi * ridge_sigma2) - np.sum(y_test - ridge_pred**2) / (2 * ridge_sigma2)
ridge_k = X_train[list(subset_regressors)].shape[1] + 1
ridge_aic = 2 * ridge_k - 2 * ridge_ll
print(f"Ridge AIC: {ridge_aic:.2f}")
print('Ridge R2: ' + str(ridge_r2))
print(ridge_model.coef_)
# LASSO Regression Analysis
lasso = Lasso(alpha = .01)
lasso_model = lasso.fit(X_train[list(subset_regressors)], y_train)
lasso_pred = lasso_model.predict(X_test[list(subset_regressors)])
lasso_r2 = r2_score(y_test, lasso_pred)
lasso_sigma2 = np.var(y_test - lasso_pred, ddof=1)
lasso_ll = -len(lasso_pred) / 2 * np.log(2 * np.pi * lasso_sigma2) - np.sum(y_test - lasso_pred**2) / (2 * lasso_sigma2)
lasso_k = X_train[list(subset_regressors)].shape[1] + 1
lasso_aic = 2 * lasso_k - 2 * lasso_ll
print(f"LASSO AIC: {lasso_aic:.2f}")
print('LASSO R2: ' + str(lasso_r2))
print(lasso_model.coef_)
# Get Ridge Plots
NormalProbabilityPlot(y_test, ridge_pred)
BasicResidualPlot(y_test, ridge_pred)
# Get LASSO Plots
NormalProbabilityPlot(y_test, lasso_pred)
BasicResidualPlot(y_test, lasso_pred)
# Ridge Cross Validation
train_subset = X_train[list(subset_regressors)]
cv_score = cross_val_score(ridge_model, train_subset, y_train, cv=10, scoring='neg_mean_squared_error')
print('Ridge Cross-Validation Score: ', np.mean(cv_score))
# Ridge Bootstrap .632 estimate
boot_scores = []
for _ in range(100):
    X_boot, y_boot = resample(train_subset, y_train)
    ridge_model.fit(X_boot, y_boot)
    boot_scores.append(mean_squared_error(y_train, ridge_model.predict(train_subset)))
boot632 = np.mean(boot_scores)
print('Ridge Bootstrap .632 Estimate: ', boot632)
# LASSO Cross Validation
train_subset = X_train[list(subset_regressors)]
cv_score = cross_val_score(lasso_model, train_subset, y_train, cv=10, scoring='neg_mean_squared_error')
print('LASSO Cross-Validation Score: ', np.mean(cv_score))
# LASSO Bootstrap .632 estimate
boot_scores = []
for _ in range(100):
    X_boot, y_boot = resample(train_subset, y_train)
    lasso_model.fit(X_boot, y_boot)
    boot_scores.append(mean_squared_error(y_train, lasso_model.predict(train_subset)))
boot632 = np.mean(boot_scores)
print('LASSO Bootstrap .632 Estimate: ', boot632)

