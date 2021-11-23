import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st

def app():
    df = pd.read_csv('healthcare-dataset-stroke-data.csv')
    st.title("""Strokes Data Set""")
    url='https://www.kaggle.com/navkanthjyothi/strokes-dataset-eda-prediction-84-recall/data?select=healthcare-dataset-stroke-data.csv'
    st.write(f"""
    * **link**:- {url}
    """)
    nost = df[df['stroke']==0]
    yest = df[df['stroke']==1]
    st.sidebar.title('Feature Input')
    da = st.sidebar.selectbox('Kind of Data',list(['All','No Stroke','Had Stroke']))
    gen = st.sidebar.selectbox('Gender',list(['Both','Male','Female']))
    if da=='All':
        if gen=='Both':
            st.write(df)
        elif gen=='Male':
            st.write(df[df['gender']=='Male'])
        elif gen=='Female':
            st.write(df[df['gender']=='Female'])
    elif da=='No Stroke':
        if gen=='Both':
            st.write(nost)
        elif gen=='Male':
            st.write(nost[nost['gender']=='Male'])
        elif gen=='Female':
            st.write(nost[nost['gender']=='Female'])
    elif da=='Had Stroke':
        if gen=='Both':
            st.write(yest)
        elif gen=='Male':
            st.write(yest[yest['gender']=='Male'])
        elif gen=='Female':
            st.write(yest[yest['gender']=='Female'])






