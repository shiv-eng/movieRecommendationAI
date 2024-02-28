# app.py
from flask import Flask, request, jsonify
import pandas as pd
from surprise import Dataset, Reader
from surprise.model_selection import train_test_split
from surprise import KNNBasic
from surprise import accuracy
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

# Change these paths to the correct paths on PythonAnywhere
ratings_df = pd.read_csv('ratings.csv')
movies_df = pd.read_csv('movies.csv')

# Preparing the dataset for Surprise
reader = Reader(rating_scale=(0.5, 5))
data = Dataset.load_from_df(ratings_df[['userId', 'movieId', 'rating']], reader)
trainset, testset = train_test_split(data, test_size=0.25)

# Use KNN algorithm for recommendations
algo = KNNBasic()

# Train the algorithm on the trainset
algo.fit(trainset)

# Predict ratings for the testset
predictions = algo.test(testset)

# Calculate and print the RMSE
accuracy.rmse(predictions)

# Define a function to make recommendations for a user based on genre
def get_recommendations_by_genre(genre, num_recommendations=10):
    # Get movie ids based on genre
    movie_ids = movies_df[movies_df['genres'].str.contains(genre, case=False)]['movieId'].tolist()

    # Get a list of all unique user ids
    user_ids = ratings_df['userId'].unique()

    # Predict ratings for movies in the specified genre for all users
    predictions = [algo.predict(user_id, movie_id) for user_id in user_ids for movie_id in movie_ids]

    # Sort the predictions in descending order of predicted rating
    predictions.sort(key=lambda x: x.est, reverse=True)

    # Get the top 'num_recommendations' movie ids with the highest predicted ratings
    top_movie_ids = [pred.iid for pred in predictions[:num_recommendations]]

    # Map the movie ids to movie titles
    top_movie_titles = movies_df[movies_df['movieId'].isin(top_movie_ids)]['title'].tolist()

    return top_movie_titles

@app.route('/')
def home():
    return 'Hello, this is the home page!'

@app.route('/recommend_by_genre', methods=['GET'])
def recommend_movies_by_genre():
    genre = request.args.get('genre')
    recommendations = get_recommendations_by_genre(genre, num_recommendations=10)
    return jsonify({"recommendations": recommendations})

if __name__ == '__main__':
    app.run(debug=True)
