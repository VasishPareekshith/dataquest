import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import xgboost as xgb
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Load the dataset
file_path = 'insurance_data.csv'
 # Replace with actual path
insurance_data = pd.read_csv(file_path)

# Handle missing values by filling them with the mean for numeric and mode for categorical data
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
xgb_model = xgb.XGBRegressor(objective ='reg:squarederror', n_estimators=150, learning_rate=0.07)
xgb_model.fit(X_train, y_train)

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

# Correlation matrix
correlation_matrix = insurance_data.corr()

# Correlation matrix heatmap
plt.figure(figsize=(10,8))
sns.heatmap(correlation_matrix, annot=True, fmt=".2f", cmap="coolwarm")
plt.title("Correlation Matrix")
plt.show()

# Feature Importance Plot
xgb.plot_importance(xgb_model, importance_type="weight")
plt.title('Feature Importance (Weight)')
plt.show()

# Function to get input and make predictions
def make_prediction(model, input_data):
    # Ensure the input data is formatted similarly to the training data
    input_df = pd.DataFrame([input_data])
    
    # Make the prediction
    prediction = model.predict(input_df)
    return prediction[0]



# Example input data for prediction
input_data = {
    'age': 46,
    'gender': 0,  # 0 for female, 1 for male
    'bmi': 25.6,
    'bloodpressure': 100,
    'diabetic': 1,  # 0 for No, 1 for Yes
    'children': 1,
    'smoker': 1,  # 0 for No, 1 for Yes
    'region': 0   # Adjust based on your label encoding for the region
}

# Get prediction for the input data
predicted_claim = make_prediction(xgb_model, input_data)
print(f"Predicted insurance claim: {predicted_claim:.2f}")




