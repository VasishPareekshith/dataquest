import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import numpy as np

# Load the dataset
file_path = 'C:\\Users\\VASISH1211\\Desktop\\DataQuest\\insurance_data.csv'
insurance_data = pd.read_csv(file_path)

# Handle missing values
insurance_data['age'] = insurance_data['age'].fillna(insurance_data['age'].mean())
insurance_data['region'] = insurance_data['region'].fillna(insurance_data['region'].mode()[0])

# Drop unnecessary columns
insurance_data.drop(columns=['index', 'PatientID'], inplace=True)

# Encode categorical variables
label_encoder = LabelEncoder()
for column in ['gender', 'diabetic', 'smoker', 'region']:
    insurance_data[column] = label_encoder.fit_transform(insurance_data[column])

# Split data into features and target
X = insurance_data.drop(columns=['claim'])
y = insurance_data['claim']

# Split into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train XGBoost model
xgb_model = xgb.XGBRegressor(objective='reg:squarederror', n_estimators=150, learning_rate=0.07)
xgb_model.fit(X_train, y_train)

# Save the model
xgb_model.save_model('model.json')

# Predictions
y_pred = xgb_model.predict(X_test)

# Evaluation Metrics
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)

# Print evaluation results
metrics = {
    'Mean Squared Error': mse,
    'Root Mean Squared Error': rmse,
    'R2 Score': r2,
    'Mean Absolute Error': mae
}

print("Model Evaluation Metrics:")
for metric, value in metrics.items():
    print(f"{metric}: {value:.4f}")
