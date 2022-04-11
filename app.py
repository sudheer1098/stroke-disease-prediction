import streamlit as st
import numpy as np
import pandas as pd
from PIL import Image
import pickle

def load_model():
    with open('stroke_model.pkl', 'rb') as file:
        model_data = pickle.load(file)
    return model_data

model_data = load_model()

model = model_data['model']
scaler = model_data['scaler_obj']

image = Image.open("media/brain-stroke.jpg")


def main():
    st.title("Stroke Disease Prediction")
    st.image(image, width=None)

    st.subheader("Enter the Patient's details")

    age = st.number_input('Enter your age')
    hypertension = st.radio('Do you have hypertension', ["yes", "no"],)
    heart_disease = st.radio('Do you have heart_disease', ["yes", "no"])
    ever_married = st.radio('Have you ever married', ["yes", "no"])
    Residence_type = st.radio('Select your residence type', ["rural", "urban"])
    avg_glucose_level = st.number_input('Enter your average glucose level')
    bmi = st.number_input('Enter your BMI')
    gender = st.radio("Select your gender", ["female", "male", "other"])
    work_type = st.radio("Select your work type", ["govt job", "never worked", "private job", "self employed", "looking after children"])
    smoking_status = st.radio("Select your smoking status", ["unknown", "formerly smoked", "never smoked", "smokes"])

    if hypertension=='yes':
        hypertension=1
    else:
        hypertension=0

    if heart_disease=='yes':
        heart_disease=1
    else:
        heart_disease=0
    
    if ever_married=='yes':
        ever_married=1
    else:
        ever_married=0
    
    if Residence_type=='rural':
        Residence_type=0
    else:
        Residence_type=1

    gender_Female=0
    gender_Male=0
    gender_Other=0

    if gender=='female':
        gender_Female=1
    elif gender=='male':
        gender_Male=1
    else:
        gender_Other=1

    work_type_Govt_job=0
    work_type_Never_worked=0
    work_type_Private=0
    work_type_Self_employed=0
    work_type_children=0

    if work_type=='govt job':
        work_type_Govt_job=1
    elif work_type=='never worked':
        work_type_Never_worked=1
    elif work_type=='private job':
        work_type_Private=1
    elif work_type=='self employed':
        work_type_Self_employed=1
    else:
        work_type_children=1

    smoking_status_Unknown=0
    smoking_status_formerly_smoked=0
    smoking_status_never_smoked=0
    smoking_status_smokes=0

    if smoking_status=="unknown":
        smoking_status_Unknown=1
    elif smoking_status=="formerly smoked":
        smoking_status_formerly_smoked=1
    elif smoking_status=="never smoked":
        smoking_status_never_smoked=1
    else:
        smoking_status_smokes=1

    ok = st.button("Predict")
    if ok:
        X = [age, hypertension, heart_disease, ever_married,
        Residence_type, avg_glucose_level, bmi, gender_Female,
        gender_Male, gender_Other, work_type_Govt_job,
        work_type_Never_worked, work_type_Private,
        work_type_Self_employed, work_type_children,
        smoking_status_Unknown, smoking_status_formerly_smoked,
        smoking_status_never_smoked, smoking_status_smokes]

        X = scaler.transform([X])

        y = model.predict(X)[0]

        if y==0:
            result = '<p style="font-family:Verdana; color:green; font-size: 30px;"><b>Less chance of STROKE!</b></p>'
            st.markdown(result, unsafe_allow_html=True)
        else:
            result = '<p style="font-family:Verdana; color:red; font-size: 30px;"><b>High chance of STROKE!</b></p>'
            st.markdown(result, unsafe_allow_html=True)
        


main()