# 1. Data Loading and Preprocessing
# ---------------------------------

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
from sklearn.pipeline import Pipeline
import xgboost as xgb
import shap
from pathlib import Path
import joblib

# Load Data
file_path = Path('MTA_Subway_Customer_Journey-Focused_Metrics__Beginning_2020.csv')
if file_path.exists():
    df = pd.read_csv(file_path)
    print("Data loaded successfully")
else:
    raise FileNotFoundError(f"File not found at: {file_path}")

# Rename columns
column_mapping = {
    'additional platform time': 'additional_platform_time',
    'additional train time': 'additional_train_time',
    'customer journey time performance': 'customer_journey_time_performance'
}
df = df.rename(columns=column_mapping)

# Feature Engineering
df['total_additional_time'] = df['additional_platform_time'] + df['additional_train_time']
df['apt_per_passenger'] = df['total_apt'] / df['num_passengers']
df['att_per_passenger'] = df['total_att'] / df['num_passengers']
df['month_num'] = pd.to_datetime(df['month']).dt.month
df['year'] = pd.to_datetime(df['month']).dt.year
df['high_performance'] = df['customer_journey_time_performance'] >= 0.90

def performance_class(cjtp):
    if cjtp >= 0.9:
        return 0
    elif cjtp >= 0.8:
        return 1
    else:
        return 2

df['performance_class'] = df['customer_journey_time_performance'].apply(performance_class)

# EDA
plt.figure(figsize=(10, 6))
sns.histplot(df['customer_journey_time_performance'], bins=20)
plt.axvline(0.9, color='red', linestyle='--')
plt.title('CJTP Distribution')
plt.savefig('cjtp_distribution.png', dpi=300)
plt.close()

plt.figure(figsize=(12, 6))
sns.boxplot(x='line', y='customer_journey_time_performance', hue='period', data=df)
plt.axhline(0.9, color='red', linestyle='--')
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig('boxplot_performance.png', dpi=300)
plt.close()

numeric_cols = ['num_passengers', 'additional_platform_time', 'additional_train_time', 'total_apt', 'total_att',
                'over_five_mins_perc', 'customer_journey_time_performance', 'total_additional_time',
                'apt_per_passenger', 'att_per_passenger']
plt.figure(figsize=(12, 10))
sns.heatmap(df[numeric_cols].corr(), annot=True, cmap='coolwarm')
plt.title('Correlation Matrix')
plt.savefig('correlation_matrix.png', dpi=300)
plt.close()

# Data Prep
X = df.drop(['month', 'high_performance', 'performance_class', 'customer_journey_time_performance'], axis=1)
y_binary = df['high_performance']
y_multi = df['performance_class']

categorical_cols = ['division', 'line', 'period']
numerical_cols = ['num_passengers', 'additional_platform_time', 'additional_train_time', 'total_apt', 'total_att',
                  'over_five_mins', 'over_five_mins_perc', 'total_additional_time', 'apt_per_passenger',
                  'att_per_passenger', 'month_num', 'year']

X_train, X_test, y_train_binary, y_test_binary = train_test_split(
    X, y_binary, test_size=0.2, random_state=42, stratify=y_binary)

X_train, X_test, y_train_multi, y_test_multi = train_test_split(
    X, y_multi, test_size=0.2, random_state=42, stratify=y_multi)

# Preprocessing
preprocessor = Pipeline([
    ('one_hot', OneHotEncoder(drop='first', sparse_output=False, handle_unknown='ignore'))
])
cat_preprocessor = preprocessor.fit(X_train[categorical_cols])
cat_features_train = cat_preprocessor.transform(X_train[categorical_cols])
cat_features_test = cat_preprocessor.transform(X_test[categorical_cols])

X_train_processed = np.hstack([X_train[numerical_cols].values, cat_features_train])
X_test_processed = np.hstack([X_test[numerical_cols].values, cat_features_test])

# Feature Names
feature_names = numerical_cols + [f"{col}_{cat}" for col, cats in zip(categorical_cols, cat_preprocessor.named_steps['one_hot'].categories_) for cat in cats[1:]]

# Random Forest (Binary)
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train_processed, y_train_binary)
rf_predictions = rf_model.predict(X_test_processed)
print("\nRandom Forest Binary:")
print(f"Accuracy: {accuracy_score(y_test_binary, rf_predictions):.4f}")
print(classification_report(y_test_binary, rf_predictions))

rf_importance = pd.DataFrame({'feature': feature_names, 'importance': rf_model.feature_importances_})
rf_importance = rf_importance.sort_values(by='importance', ascending=False)
sns.barplot(data=rf_importance.head(15), x='importance', y='feature')
plt.title("Top RF Features")
plt.tight_layout()
plt.savefig('rf_feature_importance_binary.png', dpi=300)
plt.close()

# XGBoost (Binary)
xgb_model = xgb.XGBClassifier(n_estimators=100, max_depth=3, learning_rate=0.1, random_state=42, early_stopping_rounds=10)
xgb_model.fit(X_train_processed, y_train_binary, eval_set=[(X_test_processed, y_test_binary)], verbose=0)
xgb_predictions = xgb_model.predict(X_test_processed)
print("\nXGBoost Binary:")
print(f"Accuracy: {accuracy_score(y_test_binary, xgb_predictions):.4f}")
print(classification_report(y_test_binary, xgb_predictions))

xgb_importance = pd.DataFrame({'feature': feature_names, 'importance': xgb_model.feature_importances_})
xgb_importance = xgb_importance.sort_values(by='importance', ascending=False)
sns.barplot(data=xgb_importance.head(15), x='importance', y='feature')
plt.title("Top XGBoost Features")
plt.tight_layout()
plt.savefig('xgb_feature_importance_binary.png', dpi=300)
plt.close()

# Random Forest (Multi-class)
rf_multi_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_multi_model.fit(X_train_processed, y_train_multi)
rf_multi_predictions = rf_multi_model.predict(X_test_processed)
print("\nRandom Forest Multi-class:")
print(f"Accuracy: {accuracy_score(y_test_multi, rf_multi_predictions):.4f}")
print(classification_report(y_test_multi, rf_multi_predictions))

# XGBoost (Multi-class)
xgb_multi_model = xgb.XGBClassifier(n_estimators=100, max_depth=3, learning_rate=0.1, random_state=42, early_stopping_rounds=10)
xgb_multi_model.fit(X_train_processed, y_train_multi, eval_set=[(X_test_processed, y_test_multi)], verbose=0)
xgb_multi_predictions = xgb_multi_model.predict(X_test_processed)
print("\nXGBoost Multi-class:")
print(f"Accuracy: {accuracy_score(y_test_multi, xgb_multi_predictions):.4f}")
print(classification_report(y_test_multi, xgb_multi_predictions))

# SHAP Analysis
explainer = shap.Explainer(xgb_model)
shap_values = explainer(X_test_processed)
shap.summary_plot(shap_values, X_test_processed, feature_names=feature_names)
plt.savefig('shap_summary_plot.png', dpi=300)
plt.close()

# Hyperparameter Tuning
param_grid = {
    'n_estimators': [50, 100],
    'max_depth': [None, 10, 20],
    'min_samples_split': [2, 5],
    'min_samples_leaf': [1, 2]
}
grid_search = GridSearchCV(RandomForestClassifier(random_state=42), param_grid, cv=5, scoring='accuracy')
grid_search.fit(X_train_processed, y_train_binary)
print("\nBest Params:", grid_search.best_params_)
best_rf_model = grid_search.best_estimator_
best_rf_predictions = best_rf_model.predict(X_test_processed)
print(f"Best RF Accuracy: {accuracy_score(y_test_binary, best_rf_predictions):.4f}")
print(classification_report(y_test_binary, best_rf_predictions))

# Line-specific Model
mask_a_div = X_train['division'] == 'A DIVISION'
a_train_idx = np.where(mask_a_div)[0]
X_train_a = X_train_processed[a_train_idx]
y_train_a = y_train_binary.iloc[a_train_idx]

mask_a_div_test = X_test['division'] == 'A DIVISION'
a_test_idx = np.where(mask_a_div_test)[0]
X_test_a = X_test_processed[a_test_idx]
y_test_a = y_test_binary.iloc[a_test_idx]

if len(X_train_a) > 0 and len(X_test_a) > 0:
    rf_a_model = RandomForestClassifier(n_estimators=100, random_state=42)
    rf_a_model.fit(X_train_a, y_train_a)
    rf_a_predictions = rf_a_model.predict(X_test_a)
    print("\nA Division RF:")
    print(f"Accuracy: {accuracy_score(y_test_a, rf_a_predictions):.4f}")
    print(classification_report(y_test_a, rf_a_predictions))
else:
    print("\nNot enough A Division data")

# B Division-specific Random Forest Model
mask_b_div = X_train['division'] == 'B DIVISION'
b_train_idx = np.where(mask_b_div)[0]
X_train_b = X_train_processed[b_train_idx]
y_train_b = y_train_binary.iloc[b_train_idx]

mask_b_div_test = X_test['division'] == 'B DIVISION'
b_test_idx = np.where(mask_b_div_test)[0]
X_test_b = X_test_processed[b_test_idx]
y_test_b = y_test_binary.iloc[b_test_idx]

if len(X_train_b) > 0 and len(X_test_b) > 0:
    rf_b_model = RandomForestClassifier(n_estimators=100, random_state=42)
    rf_b_model.fit(X_train_b, y_train_b)
    rf_b_predictions = rf_b_model.predict(X_test_b)
    print("\nB Division RF:")
    print(f"Accuracy: {accuracy_score(y_test_b, rf_b_predictions):.4f}")
    print(classification_report(y_test_b, rf_b_predictions))
else:
    print("\nNot enough B Division data")

# Save models
joblib.dump(rf_model, 'mta_rf_binary_model.pkl')
joblib.dump(xgb_model, 'mta_xgb_binary_model.pkl')
joblib.dump(rf_multi_model, 'mta_rf_multi_model.pkl')
joblib.dump(xgb_multi_model, 'mta_xgb_multi_model.pkl')
joblib.dump(cat_preprocessor, 'mta_categorical_preprocessor.pkl')
print("\nModels saved!")


