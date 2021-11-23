import streamlit as st
from mulapp import MultiApp
from apps import data,visualization,model

app = MultiApp()
st.markdown("""
# Multi-Page App

This multi-page app is using the [streamlit-multiapps](https://github.com/upraneelnihar/streamlit-multiapps) framework developed 
""")

app.add_app("Data",data.app)
app.add_app("Visualization",visualization.app)
app.add_app("Model",model.app)

app.run()