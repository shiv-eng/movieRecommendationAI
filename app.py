from flask import Flask, request, jsonify
from surprise import Dataset, Reader, KNNBasic, accuracy
from flask_cors import CORS
import pandas as pd

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Load dataset and train KNN algorithm
ratings_df = pd.read_csv('ratings.csv')
movies_df = pd.read_csv('movies.csv')

# Specify the rating scale for Surprise
reader = Reader(rating_scale=(0.5, 5))
data = Dataset.load_from_df(ratings_df[['userId', 'movieId', 'rating']], reader)
trainset = data.build_full_trainset()

# Train KNN algorithm
algo = KNNBasic()
algo.fit(trainset)

# Define a function to make recommendations based on genre
def get_recommendations_by_genre(genre, num_recommendations=10):
    movie_ids = movies_df[movies_df['genres'].str.contains(genre, case=False)]['movieId'].tolist()
    user_ids = ratings_df['userId'].unique()
    predictions = [algo.predict(user_id, movie_id) for user_id in user_ids for movie_id in movie_ids]
    predictions.sort(key=lambda x: x.est, reverse=True)
    top_movie_ids = [pred.iid for pred in predictions[:num_recommendations]]
    top_movie_titles = movies_df[movies_df['movieId'].isin(top_movie_ids)]['title'].tolist()
    return top_movie_titles

# Routes
@app.route('/')
def home():
    return 'Hello, this is the home page!'

@app.route('/recommend_by_genre', methods=['GET'])
def recommend_movies_by_genre():
    genre = request.args.get('genre')
    recommendations = get_recommendations_by_genre(genre, num_recommendations=10)
    return jsonify(recommendations)

if __name__ == '__main__':
    # Run the app on 0.0.0.0 (accessible externally) using Ngrok
    app.run(host='0.0.0.0', port=5000, debug=True)
