import streamlit as st
import requests
import pandas as pd

# Backend API URL
API_URL = "https://sql-dbbot-rajagopalhertzians-projects.vercel.app/"  # Replace with your backend URL

st.title("Natural Language to SQL Chatbot")
st.write("Upload a dataset and ask questions in natural language.")

# Upload Dataset
uploaded_file = st.file_uploader("Upload your dataset (CSV format)", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.write("Dataset Preview:")
    st.write(df.head())

    # Extract schema
    schema = ", ".join([f"{col} ({dtype})" for col, dtype in zip(df.columns, df.dtypes)])

    # Query Input
    user_query = st.text_input("Ask your question in natural language:")

    if user_query:
        st.write("Generating SQL query...")
        payload = {"natural_language": user_query, "dataset": schema}
        response = requests.post(f"{API_URL}/generate_sql/", json=payload)

        if response.status_code == 200:
            sql_query = response.json().get("sql_query")
            st.write("Generated SQL Query:")
            st.code(sql_query)
        else:
            st.error("Error generating SQL query.")
