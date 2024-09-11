import streamlit as st
import pandas as pd
import xgboost as xgb
from sklearn.preprocessing import LabelEncoder
import time



# Load the dataset to get the encoder and labels
file_path = 'insurance_data.csv'
insurance_data = pd.read_csv(file_path)

# Encode categorical variables
label_encoder = LabelEncoder()
for column in ['gender', 'diabetic', 'smoker', 'region']:
    insurance_data[column] = label_encoder.fit_transform(insurance_data[column])

# Load the model
xgb_model = xgb.XGBRegressor(objective='reg:squarederror', n_estimators=150, learning_rate=0.07)
xgb_model.load_model('model.json')  # Load your pre-trained model

# Streamlit app
st.title('💼 INSURANCE CLAIM PREDICTION')
st.write("Enter the details for prediction:")

# Define mappings for categorical variables
gender_mapping = {'male': 1, 'female': 0}
diabetic_mapping = {'No': 0, 'Yes': 1}
smoker_mapping = {'No': 0, 'Yes': 1}
region_mapping = {'northeast': 0, 'northwest': 1, 'southeast': 2, 'southwest': 3}


col1, col2 = st.columns(2)

# First column: 4 input fields
with col1:
    age = st.number_input('Age', min_value=0, max_value=120, value=30)
    gender = st.selectbox('Gender', options=['male', 'female'])
    bmi = st.number_input('BMI', min_value=0.0, max_value=50.0, value=25.0)
    bloodpressure = st.number_input('Blood Pressure', min_value=0, max_value=300, value=120)

# Second column: 4 input fields
with col2:
    diabetic = st.selectbox('Diabetic', options=['No', 'Yes'])
    children = st.number_input('Number of Children', min_value=0, max_value=10, value=0)
    smoker = st.selectbox('Smoker', options=['No', 'Yes'])
    region = st.selectbox('Region', options=['northeast', 'northwest', 'southeast', 'southwest'])

# Add a progress bar when calculating the prediction
progress = st.progress(0)

for i in range(100):
    time.sleep(0.02)
    progress.progress(i+1)
    
# Convert categorical inputs to numerical values
gender_encoded = gender_mapping[gender]
diabetic_encoded = diabetic_mapping[diabetic]
smoker_encoded = smoker_mapping[smoker]
region_encoded = region_mapping[region]

# Prepare input data for prediction
input_data = {
    'age': age,
    'gender': gender_encoded,
    'bmi': bmi,
    'bloodpressure': bloodpressure,
    'diabetic': diabetic_encoded,
    'children': children,
    'smoker': smoker_encoded,
    'region': region_encoded
}

input_df = pd.DataFrame([input_data])

# Make prediction
prediction = xgb_model.predict(input_df)
st.write(f"Predicted Insurance Claim: ${prediction[0]:.2f}")
