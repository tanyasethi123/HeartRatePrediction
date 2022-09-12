import pathlib
import sys
import streamlit as st
import pandas as pd
import numpy as np
import pickle
import tensorflow

import mysql.connector

def init_connection():
    return mysql.connector.connect(
        host="127.0.0.1",
        port=3306,
        user="root",
        password="root1234")


# import matplotlib.pyplot as plt 
import seaborn as sns

from sklearn.metrics import log_loss,roc_auc_score,precision_score,f1_score,recall_score,roc_curve,auc
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix,accuracy_score,fbeta_score,matthews_corrcoef
from sklearn import metrics
from sklearn.metrics import log_loss
import warnings
import sklearn
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import StratifiedKFold
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC

from scipy import stats


st.write("""
# Heart Disease Prediction Application

This is application is for **medical practitioner** not for public use.

""")
st.title("Health Form")
st.subheader("Enter Details below")

def userInputData():
    age= st.slider("Enter your age ", min_value=0,max_value=100, value=40)
    sex= st.selectbox('Sex',('Male', 'Female'))
    if sex=="Male":
        sex=1
    else:
        sex=0
    chestPain = st.selectbox('Chest pain', ('Typical', 'Typical angina', 'Non-anginal pain', 'Asymptomatic'))
    if chestPain =="Typical":
        chestPain = 1
    elif chestPain == "Typical angina":
        chestPain = 2
    elif chestPain == "Non-anginal pain":
        chestPain = 3
    elif chestPain == "Asymptomatic":
        chestpain = 4
    bloodPressure = st.slider("Enter Resting Blood Pressure", min_value=50, max_value=200, key="BP", value=120)
    cholestrol = st.slider("Enter Cholestrol", min_value=100, max_value=600, key="cholest", value=200)
    fastingBloodSugar = st.selectbox('Fasting Blood Sugar', ('Blood sugar level > 120mg/dl', 'Blood sugar level <= 120 mg/dl'))
    if fastingBloodSugar == "Blood sugar level > 120mg/dl":
        fastingBloodSugar = 1
    else:
        fastingBloodSugar = 0
    restingECG = st.selectbox('Resting ECG',('Normal', 'Abnormality in ST-T wave', 'Left ventricular hypertrophy'))
    if restingECG == "Normal":
        restingECG = 0
    elif restingECG == "Abnormality in ST-T wave":
        restingECG = 1
    elif restingECG == "Left ventricular hypertrophy":
        restingECG = 2
    maxHeartRate= st.slider('Enter maximum Heart Rate', min_value=50, max_value=200, key="heartRate", value = 120)
    exercise = st.slectibox('Angina induced by excercise',('Yes','No'))
    if exercise == "Yes":
        exercise = 1
    else:
        exercise = 0
    oldPeak = st.slider('Enter the last peak value',min_value = -10, max_value=10,key='peak value',value=0)
    STslope = st.selectbox('ST Slope',('Normal','Unsloping','Flat','Downsloping'))
    if STslope == "Normal":
        STslope =0
    elif STslope =="Unslopping":
        STslope =1
    elif STslope =="Flat":
        STslope =2
    elif STslope == "Downsloping":
        STslope =3

    data = {
        'Age': age,
        'Sex': sex,
        'Chest Pain':chestPain,
        'Blood Pressure': bloodPressure,
        'Cholestrol': cholestrol,
        'Fasting Blood Sugar': fastingBloodSugar,
        'Resting ECG': restingECG,
        'Maximum Heart Rate': maxHeartRate,
        'Exercise': exercise,
        'old Peak': oldPeak,
        'ST Slope': STslope
    }
    features = pd.DataFrame(data, index=[0])
    return features

input_df = userInputData()
st.write(input_df)

#raw_data = pd.read_csv("C:/Users/Owner/Downloads/heart_statlog_cleveland_hungary_final.csv")

# read the already created model
load_model = pickle.load(open('heartp_mod.pkl','rb'))

#apply model on the inputted model
prediction = load_model.predict(input_df)

prediction_prob = load_model.predict_proba(input_df)

st.subheader ('Prediction')
target = np.array (['Yes', 'No'])
st.write(target[prediction])

st.subheader('Prediction Probability')
st.write(prediction_prob)

input_df['target'] = prediction

conn = init_connection()

def run_query(query):
    with conn.cursor() as cur:
        cur.execute(query)
        return cur.fetchall()

# Write an Insert Query to add the data

#rows = run_query("Select * from adt_project.heart_disease_user_input")


# dt  = pd.read_csv("C:/Users/varun/Downloads/heart_statlog_cleveland_hungary_final.csv")
# dt.columns = ['age', 'sex', 'chest_pain_type', 'resting_blood_pressure', 'cholesterol', 'fasting_blood_sugar',
#               'rest_ecg', 'max_heart_rate_achieved','exercise_induced_angina', 'st_depression', 'st_slope','target']
#
# dt['chest_pain_type'][dt['chest_pain_type'] == 1] = 'typical angina'
# dt['chest_pain_type'][dt['chest_pain_type'] == 2] = 'atypical angina'
# dt['chest_pain_type'][dt['chest_pain_type'] == 3] = 'non-anginal pain'
# dt['chest_pain_type'][dt['chest_pain_type'] == 4] = 'asymptomatic'
# dt['rest_ecg'][dt['rest_ecg'] == 0] = 'normal'
# dt['rest_ecg'][dt['rest_ecg'] == 1] = 'ST-T wave abnormality'
# dt['rest_ecg'][dt['rest_ecg'] == 2] = 'left ventricular hypertrophy'
# dt['st_slope'][dt['st_slope'] == 0] = 'normal'
# dt['st_slope'][dt['st_slope'] == 1] = 'upsloping'
# dt['st_slope'][dt['st_slope'] == 2] = 'flat'
# dt['st_slope'][dt['st_slope'] == 3] = 'downsloping'
#
# dt["sex"] = dt.sex.apply(lambda  x:'male' if x==1 else 'female')
#
# dt_numeric = dt[['st_depression','resting_blood_pressure','cholesterol','max_heart_rate_achieved']]
# z = np.abs(stats.zscore(dt_numeric))
# threshold = 3
# dt = dt[(z < 3).all(axis=1)]
# dt = pd.get_dummies(dt, drop_first=True)
# X = dt.drop(['target'],axis=1)
# y = dt['target']
# X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.3,shuffle=True, random_state=5)
#
#
# dtc=DecisionTreeClassifier()
# model_dtc=dtc.fit(X_train,y_train)
# prediction_dtc=model_dtc.predict(X_test)
# #cm_dtc= confusion_matrix(y_test,prediction_dtc)
#
# prediction = model_dtc.predict(feature[:].astype(float))
# prediction_prob = model_dtc.predict_proba(feature[:].astype(float))
#
# # st.subheader('Prediction')
# # st.write(iris.target_names[prediction])
# # #st.write(prediction)
#
# st.subheader('Prediction Probability')
# st.write(prediction_prob)
# # def run_query(query):
# #     with conn.cursor() as cur:
# #         cur.execute(query)
# #         return cur.fetchall()
#
# # rows = run_query("Select * from adt_project.heart_disease_user_input")
#
# # for row in rows:
# #     st.write(f"{row[1]} has a :{row[2]}:")
# #st.line_chart(df)
#
# # df.head()