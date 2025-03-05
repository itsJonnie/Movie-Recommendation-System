import streamlit as st
import gdown
import os
import zipfile
import pandas as pd
import numpy as np
from sklearn.neighbors import NearestNeighbors  # Replace Annoy import

#############################
# 1. CONSTANTS & URL
#############################
FILE_ID = "10HbBgowGOcUpU5Z18V6i5tUqfCo3WZ1w"
ML_100K_ZIP_URL = f"https://drive.google.com/uc?id={FILE_ID}"
ZIP_FILENAME = "ml-100k.zip"
EXTRACT_DIR = "ml-100k/ml-100k"  # Updated path to match the actual directory structure

#############################
# 2. DATA LOADING FUNCTIONS
#############################

def _download_and_extract_zip():
    """
    Download ml-100k.zip from Google Drive (if not present),
    then extract it to the current directory.
    """
    try:
        # 1) Download if zip file not found
        if not os.path.exists(ZIP_FILENAME):
            st.info("Downloading dataset...")
            success = gdown.download(ML_100K_ZIP_URL, ZIP_FILENAME, quiet=False)
            if not success:
                raise Exception("Failed to download the file")
            st.success("Download complete!")

        # 2) Extract if folder not found
        if not os.path.exists(EXTRACT_DIR):
            st.info("Extracting dataset...")
            with zipfile.ZipFile(ZIP_FILENAME, 'r') as zip_ref:
                zip_ref.extractall(".")  # extracts into ./ml-100k
            st.success("Extraction complete!")

    except Exception as e:
        st.error(f"Error during download/extraction: {str(e)}")
        raise

@st.cache
def load_data():
    """
    High-level function that downloads & extracts the zip (if needed),
    and returns the ratings and movies DataFrames.
    """
    try:
        # Check if the data files exist in the correct location
        ratings_path = os.path.join(EXTRACT_DIR, "u.data")
        movies_path = os.path.join(EXTRACT_DIR, "u.item")
        
        if not os.path.exists(ratings_path) or not os.path.exists(movies_path):
            _download_and_extract_zip()
        
        # Read ratings
        ratings = pd.read_csv(
            ratings_path,
            sep="\t",
            names=["userId", "movieId", "rating", "timestamp"]
        )
        
        # Read movies
        movies = pd.read_csv(
            movies_path,
            sep="|",
            encoding="latin-1",
            names=["movieId", "title", "release_date", "video_release_date", "IMDB_url",
                   "unknown", "Action", "Adventure", "Animation", "Children", "Comedy", "Crime",
                   "Documentary", "Drama", "Fantasy", "Film-Noir", "Horror", "Musical", "Mystery",
                   "Romance", "Sci-Fi", "Thriller", "War", "Western"],
            usecols=[0, 1]
        )
        
        return ratings, movies
        
    except Exception as e:
        st.error(f"Error loading data: {str(e)}")
        raise

@st.cache
def build_user_movie_matrix(ratings: pd.DataFrame):
    """
    Create and return a user–movie matrix from the ratings DataFrame.
    """
    matrix = ratings.pivot_table(index="userId", columns="movieId", values="rating").fillna(0)
    return matrix

@st.cache
def prepare_knn_data(user_movie_matrix):
    """
    Prepare data for k-NN based recommendations using scikit-learn.
    """
    # Calculate user means and normalize the matrix
    user_means = user_movie_matrix.mean(axis=1)
    normalized_matrix = user_movie_matrix.sub(user_means, axis=0)
    
    # Build kNN model
    knn = NearestNeighbors(n_neighbors=11, metric='euclidean')  # 11 because we'll remove the user itself
    knn.fit(normalized_matrix)
    
    return knn, normalized_matrix

#############################
# 3. MODEL / RECOMMENDATION
#############################

def recommend_movies_knn(user_id: int, user_movie_matrix: pd.DataFrame, movies_df: pd.DataFrame, num_recommendations=5):
    """
    Get movie recommendations using scikit-learn k-NN approach.
    """
    # Prepare data
    knn, normalized_matrix = prepare_knn_data(user_movie_matrix)
    
    try:
        user_index = normalized_matrix.index.get_loc(user_id)
    except KeyError:
        return []
    
    # Get the target user's normalized vector
    user_vector = normalized_matrix.iloc[user_index].to_numpy().reshape(1, -1)
    
    # Get nearest neighbors
    distances, indices = knn.kneighbors(user_vector)
    
    # Remove the user itself (first result)
    neighbor_indices = indices[0][1:]
    distances = distances[0][1:]
    
    # Calculate recommendations
    recommended_scores = {}
    for neighbor_idx, dist in zip(neighbor_indices, distances):
        similarity = 1 / (1 + dist)
        neighbor_ratings = normalized_matrix.iloc[neighbor_idx]
        for movie, rating in neighbor_ratings.items():
            if rating != 0:
                recommended_scores[movie] = recommended_scores.get(movie, 0) + similarity * rating
    
    # Filter out movies the user has already rated
    already_rated = set(user_movie_matrix.loc[user_id][user_movie_matrix.loc[user_id] > 0].index)
    filtered_scores = {movie: score for movie, score in recommended_scores.items() if movie not in already_rated}
    
    # Get top recommendations
    recommended_movies = sorted(filtered_scores, key=filtered_scores.get, reverse=True)[:num_recommendations]
    
    # Convert movie IDs to titles
    movie_titles = []
    for movie_id in recommended_movies:
        title = movies_df[movies_df['movieId'] == movie_id]['title'].iloc[0]
        movie_titles.append(title)
    
    return movie_titles

def recommend_movies_svd(user_id: int, user_movie_matrix: pd.DataFrame, num_recommendations=5):
    """
    Replace with your actual SVD-based recommendation logic.
    For now, return dummy movie titles.
    """
    return [f"SVDMovie_{i}" for i in range(1, num_recommendations+1)]

#############################
# 4. STREAMLIT UI
#############################

def main():
    st.title("Movie Recommendation System (ML-100k)")

    # 4.1 Load data
    st.write("Loading data from Google Drive if necessary...")
    ratings, movies = load_data()
    
    # 4.2 Build user–movie matrix
    user_movie_matrix = build_user_movie_matrix(ratings)
    
    # 4.3 Quick preview (optional)
    if st.checkbox("Show data samples"):
        st.write("Ratings sample:")
        st.write(ratings.head())
        st.write("Movies sample:")
        st.write(movies.head())
    
    # 4.4 UI for user ID and model selection
    user_id_input = st.text_input("Enter User ID:", "10")
    model_choice = st.selectbox("Choose Recommendation Model", ("KNN-based", "SVD-based"))
    
    if st.button("Get Recommendations"):
        try:
            user_id = int(user_id_input)
            
            if model_choice == "KNN-based":
                recs = recommend_movies_knn(user_id, user_movie_matrix, movies)
            else:
                recs = recommend_movies_svd(user_id, user_movie_matrix)
            
            st.write(f"Recommendations for User {user_id} using {model_choice}:")
            st.write(recs)
        except ValueError:
            st.error("Please enter a valid numeric User ID.")

if __name__ == "__main__":
    main()
