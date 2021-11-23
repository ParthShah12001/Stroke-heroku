import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.utils import resample, shuffle
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score,confusion_matrix
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC

global worty1, mar1, hede1, hyten1, gen1, res1, ss1
st.title('Stroke Prediction')
st.write(" ")
st.write('Depending on vitals of human body this model can predict if the people  will have a * **stroke** or not')
st.write(" ")
st.write(" ")
st.write(" ")
st.sidebar.title('Vitals')

df = pd.read_csv("healthcare-dataset-stroke-data.csv")
df.isna().sum()
df.dropna(inplace=True)
le = LabelEncoder()
le.fit(df['smoking_status'])
df['smoking_status'] = le.transform(df['smoking_status'])
le.fit(df['work_type'])
df['work_type'] = le.transform(df['work_type'])
le.fit(df['Residence_type'])
df['Residence_type'] = le.transform(df['Residence_type'])
le.fit(df['ever_married'])
df['ever_married'] = le.transform(df['ever_married'])
le.fit(df['gender'])
df['gender'] = le.transform(df['gender'])

df.drop(['avg_glucose_level', 'id'], axis=1, inplace=True)

zero = df[df['stroke'] == 0]
one = df[df['stroke'] == 1]
upsampled1 = resample(one, replace=True, n_samples=zero.shape[0])
df = pd.concat([zero, upsampled1])
df = shuffle(df)

x = df.drop(['stroke'], axis=1)
y = df['stroke']

gen = st.sidebar.selectbox('Gender', list(['Female', 'Male']))
age = int(st.sidebar.number_input('Age'))
hyten = st.sidebar.selectbox('Hypertension', list(['No', 'yes']))
hede = st.sidebar.selectbox('HeartDisease', list(['No', 'Yes']))
mar = st.sidebar.selectbox('Ever Married', list(['No', 'Yes']))
worty = st.sidebar.selectbox('Work Type', list(['Govt job', 'Never Worked', 'Private Job', 'Self Employed', 'Child']))
res = st.sidebar.selectbox('Residence Type', list(['Urban', 'Rular']))
bmi = float(st.sidebar.number_input('bmi'))
ss = st.sidebar.selectbox('smoking status', list(['Unknown', 'Formerly Smoked', 'Never Smoked', 'Smokes']))

if gen == 'Female':
    gen1 = int(0)
elif gen == 'Male':
    gen1 = int(1)

if hyten == 'No':
    hyten1 = int(0)
elif hyten == 'Yes':
    hyten1 = int(1)

if hede == 'No':
    hede1 = int(0)
elif hede == 'Yes':
    hede1 = int(1)

if mar == 'No':
    mar1 = int(0)
elif mar == 'Yes':
    mar1 = int(1)

if res == 'Rular':
    res1 = int(0)
elif res == 'Urban':
    res1 = int(1)

if ss == 'Unknown':
    ss1 = int(0)
elif ss == 'Formerly Smoked':
    ss1 = int(1)
elif ss == 'Never Smoked':
    ss1 = int(2)
elif ss == 'Smokes':
    ss1 = int(3)

if worty == 'Govt job':
    worty1 = int(0)
elif worty == 'Never Worked':
    worty1 = int(1)
elif worty == 'Private Job':
    worty1 = int(2)
elif worty == 'Self Employed':
    worty1 = int(3)
elif worty == 'Child':
    worty1 = int(4)

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.7, random_state=101)

a = st.sidebar.selectbox('Model',
                         list(['Logistic Regression', 'KNN', 'Decision Tree', 'Random Forest', 'SVM', 'GaussianNB']))

if a == 'Logistic Regression':
    lg = LogisticRegression(max_iter=450)
    lg.fit(x_train, y_train)
    lg_predict = lg.predict(x_test)
    lg_cm = confusion_matrix(lg_predict, y_test)
    fig = plt.figure(figsize=(2, 2))
    sns.heatmap(lg_cm, annot=True)
    st.pyplot(fig)
    sc = accuracy_score(y_test, lg_predict) * 100
    st.write(f""" 
     * **accuracy_score:-** {sc}
     """)
    z = lg.predict(np.array([[gen1, age, hyten1, hede1, mar1, worty1, res1, bmi, ss1]]))
    st.write("""
     * **Output of Custom Input:-**
     """)
    if z[0] == 0:
        st.write("#### No Stroke")
    elif z[0] == 1:
        st.write("#### Stroke")

if a == 'KNN':
    kn = KNeighborsClassifier(n_neighbors=4)
    kn.fit(x_train, y_train)
    kn_predict = kn.predict(x_test)
    kn_cm = confusion_matrix(y_test, kn_predict)
    fig = plt.figure(figsize=(2, 2))
    sns.heatmap(kn_cm, annot=True)
    st.pyplot(fig)
    sc = accuracy_score(y_test, kn_predict) * 100
    st.write(f""" 
         * **accuracy_score:-** {sc}
         """)
    z = kn.predict(np.array([[gen1, age, hyten1, hede1, mar1, worty1, res1, bmi, ss1]]))
    st.write("""
                 * **Output of Custom Input:-**
                 """)
    if z[0] == 0:
        st.write("#### No Stroke")
    elif z[0] == 1:
        st.write("#### Stroke")

if a == 'Decision Tree':
    dt = DecisionTreeClassifier()
    dt.fit(x_train, y_train)
    dt_predict = dt.predict(x_test)
    dt_cm = confusion_matrix(y_test, dt_predict)
    fig = plt.figure(figsize=(2, 2))
    sns.heatmap(dt_cm, annot=True)
    plt.title('Decision Tree')
    st.pyplot(fig)
    sc = accuracy_score(y_test, dt_predict) * 100
    st.write(f""" 
         * **accuracy_score:-** {sc}
         """)
    z = dt.predict(np.array([[gen1, age, hyten1, hede1, mar1, worty1, res1, bmi, ss1]]))
    st.write("""
         * **Output of Custom Input:-**
         """)
    if z[0] == 0:
        st.write("#### No Stroke")
    elif z[0] == 1:
        st.write("#### Stroke")

if a == 'Random Forest':
    rf = RandomForestClassifier()
    rf.fit(x_train, y_train)
    rf_predict = rf.predict(x_test)
    rf_cm = confusion_matrix(y_test, rf_predict)
    fig = plt.figure(figsize=(2, 2))
    sns.heatmap(rf_cm, annot=True)
    plt.title('Random Forest')
    st.pyplot(fig)
    sc = accuracy_score(y_test, rf_predict) * 100
    st.write(f""" 
         * **accuracy_score:-** {sc}
         """)
    z = rf.predict(np.array([[gen1, age, hyten1, hede1, mar1, worty1, res1, bmi, ss1]]))
    st.write("""
         * **Output of Custom Input:-**
         """)
    if z[0] == 0:
        st.write("#### No Stroke")
    elif z[0] == 1:
        st.write("#### Stroke")

if a == 'SVM':
    svm = SVC()
    svm.fit(x_train, y_train)
    svm_predict = svm.predict(x_test)
    svm_cm = confusion_matrix(y_test, svm_predict)
    fig = plt.figure(figsize=(2, 2))
    sns.heatmap(svm_cm, annot=True)
    plt.title('SVM')
    st.pyplot(fig)
    sc = accuracy_score(y_test, svm_predict) * 100
    st.write(f""" 
         * **accuracy_score:-** {sc}
         """)
    z = svm.predict(np.array([[gen1, age, hyten1, hede1, mar1, worty1, res1, bmi, ss1]]))
    st.write("""
         * **Output of Custom Input:-**
         """)
    if z[0] == 0:
        st.write("#### No Stroke")
    elif z[0] == 1:
        st.write("#### Stroke")

if a == 'GaussianNB':
    nb = GaussianNB()
    nb.fit(x_train, y_train)
    nb_predict = nb.predict(x_test)
    fig = plt.figure(figsize=(2, 2))
    sns.heatmap(confusion_matrix(y_test, nb_predict), annot=True)
    plt.title('GaussianNB')
    st.pyplot(fig)
    sc = accuracy_score(y_test, nb_predict) * 100
    st.write(f""" 
         * **accuracy_score:-** {sc}
         """)
    z = nb.predict(np.array([[gen1, age, hyten1, hede1, mar1, worty1, res1, bmi, ss1]]))
    st.write("""
         * **Output of Custom Input:-**
         """)
    if z[0] == 0:
        st.write("#### No Stroke")
    elif z[0] == 1:
        st.write("#### Stroke")
