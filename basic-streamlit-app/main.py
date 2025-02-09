# Import Streamlit library
import streamlit as st
import pandas as pd

#In terminal: streamlit run main.py
## given local host: http://localhost:8501 

# Title
st.title("Welcome to a summary of different organization!")

# Description
st.write("This is a Streamlit App to help sort different organizations' data including organization name, website, country, description, when it was founded, the industry, and the number of employees. Sort through the data as you please!")
st.subheader("Here is the data:")

#Loading csv file
df = pd.read_csv("Powers-Data-Science-Portfolio/basic-streamlit-app/data/organizations-100.csv")

# Interative table
st.dataframe(df)

#Filter by name
organization = st.selectbox("Select an organization", df["Name"].unique())

# Filtering the DataFrame based on user selection
filtered_df = df[df["Name"] == organization]

# Display the filtered results
st.write(f"Information for {organization}:")
st.dataframe(filtered_df)