import pandas as pd
# import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import sklearn
import warnings
warnings.filterwarnings('ignore')
import pickle
from sklearn.linear_model import LinearRegression
import streamlit as st
import joblib
import time



data = pd.read_csv('USA_Housing.csv')

modelling = joblib.load(open('house.pkl', 'rb'))


#to add picture from local computer
import base64
def add_bg_from_local(image_file):
    with open(image_file, "rb") as image_file:
        encoded_string = base64.b64encode(image_file.read())
    st.markdown(
    f"""
    <style>
    .stApp {{
        background-image: url(data:image/{"png"};base64,{encoded_string.decode()});
        background-size: cover
    }}
    </style>
    """,
    unsafe_allow_html=True
    )
add_bg_from_local('house.jpg') 

# to import css file into streamlit
with open('style.css') as f:
    st.markdown(f'<style>{f.read()}</style>',unsafe_allow_html=True)
    

#form
with st.form('my_form', clear_on_submit=True):
    st.header("HOUSE PREDICTION")
    income = st.number_input('Area Income')
    age = st.number_input('Area House Age')
    room = st.number_input('Number of Rooms')
    pop = st.number_input('Area population')
    submitted = st.form_submit_button("PREDICT")
    if (income and age and room and pop):
        if submitted:
            with st.spinner(text='In progress'):
                time.sleep(3)
                st.success('Done')
            st.write("Your Inputted Data:")
            input_var = pd.DataFrame([{'Avg. Area Income' : income,	'Avg. Area House Age' : age,	'Avg. Area Number of Rooms' : room,	'Area Population' : pop}])
            st.write(input_var)
            
            from sklearn.preprocessing import LabelEncoder, StandardScaler
            scaler = StandardScaler()
            for i in input_var:
                input_var[[i]] = scaler.fit_transform(input_var[[i]])
            
            time.sleep(2)
            tab1, tab2 = st.tabs(["Prediction Pane", "Intepretation Pane"])
            with tab1:
                    st.markdown("<br>", unsafe_allow_html= True)
                    prediction = modelling.predict(input_var)
                    st.write("Predicted Price is :", prediction)
            with tab2:
                st.subheader('modelling Interpretation')
                st.write(f"Profit = {modelling.intercept_.round(2)} + {modelling.coef_[0].round(2)}(Avg. Area Income)+ {modelling.coef_[1].round(2)} (Avg. Area House Age) + {modelling.coef_[2].round(2)} (Avg. Area Number of Rooms)")

                st.markdown("<br>", unsafe_allow_html= True)

                st.markdown(f"- The expected Profit for a startup is {modelling.intercept_}")

                st.markdown(f"- For every additional 1 dollar spent on Avg. Area Income, the expected profit is expected to increase by ${modelling.coef_[0].round(2)}")

                st.markdown(f"- For every additional 1 dollar spent on Avg. Area House Age, the expected profit is expected to decrease by ${modelling.coef_[1].round(2)}")

                st.markdown(f"- For every additional 1 dollar spent on Avg. Area Number of Rooms, the expected profit is expected to increase by ${modelling.coef_[2].round(2)}")