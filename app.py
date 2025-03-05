import streamlit as st
import gdown
import os
import zipfile
import pandas as pd

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

#############################
# 3. MODEL / RECOMMENDATION DUMMIES
#############################

def recommend_movies_annoy(user_id: int, user_movie_matrix: pd.DataFrame, num_recommendations=5):
    """
    Replace with your actual Annoy-based recommendation logic.
    For now, return dummy movie titles.
    """
    # e.g., find neighbors, filter out seen items, etc.
    return [f"AnnoyMovie_{i}" for i in range(1, num_recommendations+1)]

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
    model_choice = st.selectbox("Choose Recommendation Model", ("Annoy-based", "SVD-based"))
    
    if st.button("Get Recommendations"):
        try:
            user_id = int(user_id_input)
            
            if model_choice == "Annoy-based":
                recs = recommend_movies_annoy(user_id, user_movie_matrix)
            else:
                recs = recommend_movies_svd(user_id, user_movie_matrix)
            
            st.write(f"Recommendations for User {user_id} using {model_choice}:")
            st.write(recs)
        except ValueError:
            st.error("Please enter a valid numeric User ID.")

if __name__ == "__main__":
    main()
