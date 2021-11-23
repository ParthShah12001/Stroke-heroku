import numpy as np
import pandas as pd
import seaborn as sns
import streamlit as st
import matplotlib.pyplot as plt

def app():
    st.sidebar.title("Type of Plot")
    st.title('Visualization of Data')
    st.write(" ")
    st.write(" ")
    df = pd.read_csv('healthcare-dataset-stroke-data.csv')

    a = st.sidebar.selectbox('Select Plot',list(['Number of strokes','Gender','Stroke rate in Gender','Age vs Stroke','Age vs heart Disease','Job vs tension','Hypertension vs Stroke','Smoking vs heart disease','Married vs Stroke']))

    if a =='Number of strokes':
        fig = plt.figure(figsize=(10,8))
        sns.countplot(x = df['stroke'])
        plt.title('Stroke distribution')
        st.pyplot(fig)
        st.write("""
                ### Observation:-
                """)
        st.write("As we can see that data in dataset has imbalance in it so before we train it we need to Balance it properly")
    elif a =='Stroke as per Gender':
        fig = plt.figure(figsize=(10, 8))
        sns.countplot(x=df['gender'])
        plt.title('Gender count')
        st.pyplot(fig)
        st.write("""
                ### Observation:-
                """)
        st.write('According to data Female has more chance of having a Stroke then Male')

    elif a == 'Stroke rate according to  Gender':
        fig = plt.figure(figsize=(10, 8))
        sns.countplot(x=df['gender'], hue=df['stroke'])
        plt.title('Stroke rate in gender')
        st.pyplot(fig)

    elif a == 'Age vs Stroke':
        fig = plt.figure(figsize=(10, 8))
        sns.lineplot(x=df['age'], y=df['stroke'])
        plt.title('Age vs stroke')
        st.pyplot(fig)
        st.write("""
            ### Observation:-
            """)
        st.write("so as age increases chance of having a stroke increases")

    elif a == 'Age vs heart Disease':
        fig = plt.figure(figsize=(10, 8))
        sns.lineplot(x=df['age'], y=df['heart_disease'])
        plt.title('Age vs heart disease')
        st.pyplot(fig)
        st.write("""
        ### Observation:-
        """)
        st.write('As age increases chances of having heart diseases also increases')

    elif a == 'Job vs tension':
        fig = plt.figure(figsize=(10, 8))
        sns.barplot(x=df['work_type'], y=df['hypertension'])
        plt.title('job with more tension')
        st.pyplot(fig)
        st.write("""
        ### Observation:-
        """)
        st.write("From the chart we can see that Self employeed people have the most amout of tension and as tension and Stroke are directly related they have more chance of having a Stroke")

    elif a == 'Hypertension vs Stroke':
        fig = plt.figure(figsize=(10, 8))
        sns.countplot(x=df['hypertension'], hue=df['stroke'])
        plt.title('Hypertension and stroke')
        st.pyplot(fig)
        st.write("""
        ### Observation:-
        """)
        st.write("""
        Large number of people didn't have hypertension and only few had a stroke where as count of people with hypertension where few but significant number of people had a stroke.
        so we can say that chances of having a stroke increases with hypertension
        """)

    elif a == 'Smoking vs heart disease':
        fig = plt.figure(figsize=(10, 8))
        sns.barplot(x=df['smoking_status'], y=df['heart_disease'])
        plt.title("Smoking vs heart disease")
        st.pyplot(fig)
        st.write("""
        ### Observation:-
        """)
        st.write('We can clearly see that those who have been former smokers or are still smoking have a greater chance of heart diseases')

    elif a == 'Married vs Stroke':
        fig = plt.figure(figsize=(10, 8))
        sns.barplot(x=df['ever_married'], y=df['stroke'])
        plt.title('Married vs stroke')
        st.pyplot(fig)
        st.write('### Observation:-')
        st.write('married people are tent to have a stroke then unmarried people')