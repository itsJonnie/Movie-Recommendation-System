def recommend_movies_svd(user_id, predicted_ratings_df, raw_matrix, num_recommendations=5):
    """
    Recommend movies for a given user based on SVD-predicted ratings.
    
    Parameters:
      - user_id: int, the target user's ID.
      - predicted_ratings_df: DataFrame of predicted ratings from SVD.
      - raw_matrix: DataFrame of raw user ratings (used to filter out already rated movies).
      - num_recommendations: Number of recommendations to return.
      
    Returns:
      - List of recommended movie titles.
    """
    try:
        # Get predicted ratings for the target user
        user_pred = predicted_ratings_df.loc[user_id]
    except KeyError:
        return []
    
    # Identify movies already rated by the user from the raw (filtered) matrix
    already_rated = set(raw_matrix.loc[user_id][raw_matrix.loc[user_id] > 0].index)
    
    # Filter out movies already rated)
    user_pred_filtered = user_pred.drop(already_rated, errors='ignore')
    
    # Sort the predicted ratings in descending order and pick the top recommendations
    recommended_movies = user_pred_filtered.sort_values(ascending=False).head(num_recommendations).index.tolist()
    return recommended_movies

# Example usage:
user_id = 10  # Change this to a valid user ID from your training set
svd_recs = recommend_movies_svd(user_id, predicted_ratings_df, train_user_movie_matrix, num_recommendations=5)
print("SVD-based recommendations for user", user_id, ":", svd_recs)