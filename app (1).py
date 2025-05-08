import streamlit as st
import pandas as pd
from surprise import Dataset, Reader, SVD
from surprise.model_selection import train_test_split

st.title("AI-Powered Movie Recommender System")

@st.cache_data
def load_data():
    movies = pd.read_csv("modified_movies.csv")
    
    # Create sample user ratings for demonstration
    ratings_data = pd.DataFrame({
        'userID': [1, 1, 1, 2, 2, 3],
        'itemID': [1, 2, 3, 2, 3, 1],
        'rating': [4, 5, 3, 4, 2, 5]
    })

    return movies, ratings_data

movies, ratings_data = load_data()

st.subheader("Sample Movies Dataset")
st.dataframe(movies.head())

st.subheader("Training SVD Recommender...")

# Prepare Surprise data
reader = Reader(rating_scale=(0, 5))
data = Dataset.load_from_df(ratings_data[['userID', 'itemID', 'rating']], reader)
trainset, testset = train_test_split(data, test_size=0.25)

model = SVD()
model.fit(trainset)

st.success("Model trained successfully!")

# Recommend top 5 for user 1
user_id = 1
all_movie_ids = movies['id'].unique()
watched = ratings_data[ratings_data['userID'] == user_id]['itemID'].tolist()
unwatched = [mid for mid in all_movie_ids if mid not in watched]

predictions = [(mid, model.predict(user_id, mid).est) for mid in unwatched]
top_5 = sorted(predictions, key=lambda x: x[1], reverse=True)[:5]

st.subheader(f"Top 5 Recommendations for User {user_id}")
recommended = movies[movies['id'].isin([mid for mid, _ in top_5])]
st.table(recommended[['id', 'title', 'year', 'rating']])