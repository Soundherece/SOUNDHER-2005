import streamlit as st
import pandas as pd
from surprise import Dataset, Reader, SVD
from surprise.model_selection import train_test_split
import pickle

st.title("Movie Recommendation System")

@st.cache_data
def load_data():
    return pd.read_csv("modified_movies.csv")

movies = load_data()
st.write("Sample Movies Dataset", movies.head())

# Dummy interaction
st.text("Model and full interaction pipeline goes here...")