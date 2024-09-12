import streamlit as st
import pandas as pd
import xgboost as xgb
from sklearn.preprocessing import LabelEncoder
import time
import seaborn as sns
import matplotlib.pyplot as plt

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
    age = st.number_input('Age', min_value=18, max_value=60, value=18)
    gender = st.selectbox('Gender', options=['male', 'female'])
    bmi = st.number_input('BMI', min_value=16, max_value=53.1, value=16)
    bloodpressure = st.number_input('Blood Pressure', min_value=80, max_value=140, value=80)

# Second column: 4 input fields
with col2:
    diabetic = st.selectbox('Diabetic', options=['No', 'Yes'])
    children = st.number_input('Number of Children', min_value=0, max_value=5, value=0)
    smoker = st.selectbox('Smoker', options=['No', 'Yes'])
    region = st.selectbox('Region', options=['northeast', 'northwest', 'southeast', 'southwest'])

# Add a progress bar when calculating the prediction
progress = st.progress(0)

for i in range(100):
    time.sleep(0.02)
    progress.progress(i+1)
    
    
insurance_data = insurance_data.drop(columns=['index', 'PatientID'])

# Define a function to plot the correlation matrix
def plot_correlation_matrix(data):
    # Compute the correlation matrix
    correlation_matrix = data.corr()

    # Set up the matplotlib figure
    plt.figure(figsize=(10, 8))

    # Draw the heatmap
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f', linewidths=0.5)

    # Set titles and labels
    plt.title('Correlation Matrix')
    plt.xlabel('Features')
    plt.ylabel('Features')

    # Return the plot
    return plt

# Streamlit app
st.title('Insurance Data Analysis')

# Button to show/hide the correlation matrix
if 'show_correlation' not in st.session_state:
    st.session_state.show_correlation = False

toggle_button = st.button('Toggle Correlation Matrix')

if toggle_button:
    st.session_state.show_correlation = not st.session_state.show_correlation

if st.session_state.show_correlation:
    st.write("### Feature Correlation Matrix")
    plt = plot_correlation_matrix(insurance_data)
    st.pyplot(plt)

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
