import streamlit as st
from pycaret.classification import load_model, predict_model
import pandas as pd

st.title("Obesity Prediction")

gender = st.selectbox('Gender', ['-', 'Male', 'Female'])
age = st.number_input('Age', min_value=0, max_value=100)
height = st.number_input('Height (meters)', min_value=0.0, max_value=2.5)
weight = st.number_input('Weight (kg)', min_value=0, max_value=200)
family_history = st.selectbox('Family History with Overweight', ['-', 'yes', 'no'])
high_caloric_food = st.selectbox('Frequent Consumption of High Caloric Food', ['-', 'yes', 'no'])
vegetables = st.number_input('How many days a week you consume vegetables? (days/week)', min_value=0, max_value=7)
meals = st.number_input('Number of Main Meals?', min_value=0, max_value=10)
food_between_meals = st.selectbox('Do you consume food between meals?', ['-', 'Sometimes', 'Frequently', 'Always', 'Never'])
water = st.number_input('How many litres of water you drink daily (liters)', min_value=0.0, max_value=20.0)
calories_monitoring = st.selectbox('Do you monitor the amount of calories you intake?', ['-', 'yes', 'no'])
physical_activity = st.number_input('Physical Activity Frequency (days/week)', min_value=0, max_value=7)
technology_time = st.number_input('How long do you use electronic devices? (hours/day)', min_value=0, max_value=24)
alcohol = st.selectbox('Consumption of Alcohol', ['-', 'yes', 'no'])
transportation = st.selectbox('Transportation Used', ['-', 'Automobile', 'Bike', 'Walking', 'Public Transport'])
smoke = st.selectbox('Do you smoke?', ['-', 'yes', 'no'])

# Function to check if all fields are filled
def all_fields_filled():
    return all([
        gender != '-', age != 0, height != 0.0, weight != 0,
        family_history != '-', high_caloric_food != '-',
        meals != 0, food_between_meals != '-', water != 0.0,
        calories_monitoring != '-',
        alcohol != '-', transportation != '-', smoke != '-'
    ])

model = load_model('obesity_pred_model')

input_data = pd.DataFrame({
    'Gender': [gender],
    'Age': [age],
    'Height': [height],
    'Weight': [weight],
    'family_history_with_overweight': [family_history],
    'frequent_consumption_of_high_caloric_food': [high_caloric_food],
    'frequency_of_consumption_of_vegetables': [vegetables],
    'number_of_main_meals': [meals],
    'consumption_of_food_between_meals': [food_between_meals],
    'consumption_of_water_daily': [water],
    'calories_consumption_monitoring': [calories_monitoring],
    'physical_activity_frequency': [physical_activity],
    'time_using_technology_devices': [technology_time],
    'consumption_of_alcohol': [alcohol],
    'transportation_used': [transportation],
    'SMOKE': [smoke],
    'id': 1
})

st.write("Input Data:")
st.write(input_data)

if all_fields_filled():
    if st.button('Predict'):
        result = predict_model(model, data=input_data)
        st.write(f"The predicted obesity class is: {result.iloc[0, -2]}")
else:
    st.error("Please fill in all fields to make a prediction.")
