import streamlit as st
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import LabelEncoder

# Load your dataset with proper encoding
df = pd.read_csv("IMDb-India.csv", encoding='latin1')

# Use correct column names from your Excel
df = df[['Genre', 'Director', 'Actor 1', 'Rating']].dropna()

# Label encoding for categorical features
le_genre = LabelEncoder()
le_director = LabelEncoder()
le_actor = LabelEncoder()

df['genre_enc'] = le_genre.fit_transform(df['Genre'])
df['director_enc'] = le_director.fit_transform(df['Director'])
df['actor_enc'] = le_actor.fit_transform(df['Actor 1'])

# Features and target
X = df[['genre_enc', 'director_enc', 'actor_enc']]
y = df['Rating']

# Train the model
model = LinearRegression()
model.fit(X, y)

# Streamlit UI
st.title("🎬 Movie Rating Predictor (IMDb India 🇮🇳)")
st.write("Predict IMDb rating based on Genre, Director, and Lead Actor")

# Dropdowns for user input
genre = st.selectbox("Select Genre", le_genre.classes_)
director = st.selectbox("Select Director", le_director.classes_)
actor = st.selectbox("Select Lead Actor (Actor 1)", le_actor.classes_)

if st.button("Predict Rating"):
    input_data = [[
        le_genre.transform([genre])[0],
        le_director.transform([director])[0],
        le_actor.transform([actor])[0]
    ]]
    predicted_rating = model.predict(input_data)[0]
    st.success(f"⭐ Predicted IMDb Rating: {predicted_rating:.2f} / 10")
