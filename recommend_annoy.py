def recommend_movies_annoy(user_id, annoy_index, normalized_matrix, raw_matrix, 
                           num_neighbors=5, num_recommendations=5):
    """
    Recommend movies for a given user using an Annoy-based k-NN model.
    
    Parameters:
      - user_id: int, the target user's ID.
      - annoy_index: an AnnoyIndex built on the normalized user-movie matrix.
      - normalized_matrix: DataFrame with mean-centered (normalized) ratings.
      - raw_matrix: DataFrame with original ratings (to filter out already rated movies).
      - num_neighbors: number of nearest neighbors to consider.
      - num_recommendations: number of movie recommendations to return.
    
    Returns:
      - A list of recommended movie titles.
    """
    try:
        user_index = normalized_matrix.index.get_loc(user_id)
    except KeyError:
        return []
    
    # Get the target user's normalized vector
    user_vector = normalized_matrix.iloc[user_index].to_numpy()
    
    # Retrieve nearest neighbors using Annoy
    # Note: Annoy includes the query item itself, so we ask for one extra neighbor.
    neighbor_indices, distances = annoy_index.get_nns_by_vector(user_vector, num_neighbors+1, include_distances=True)
    
    # Exclude the user itself if present
    if neighbor_indices[0] == user_index:
        neighbor_indices = neighbor_indices[1:]
        distances = distances[1:]
    else:
        neighbor_indices = neighbor_indices[:-1]
        distances = distances[:-1]
    
    # Aggregate weighted scores from neighbors
    recommended_scores = {}
    for neighbor_idx, dist in zip(neighbor_indices, distances):
        # Convert distance to a similarity score.
        # One simple way: similarity = 1 / (1 + distance)
        similarity = 1 / (1 + dist)
        neighbor_ratings = normalized_matrix.iloc[neighbor_idx]
        # For each movie that the neighbor has rated (non-zero normalized rating)
        for movie, rating in neighbor_ratings.items():
            if rating != 0:
                recommended_scores[movie] = recommended_scores.get(movie, 0) + similarity * rating
    
    # Exclude movies that the user has already rated (using raw ratings)
    already_rated = set(raw_matrix.loc[user_id][raw_matrix.loc[user_id] > 0].index)
    filtered_scores = {movie: score for movie, score in recommended_scores.items() if movie not in already_rated}
    
    # Sort movies by the aggregated weighted score and select the top recommendations
    recommended_movies = sorted(filtered_scores, key=filtered_scores.get, reverse=True)[:num_recommendations]
    return recommended_movies

# Example usage:
user_id = 10  # Change this to test with a different user ID
annoy_recs = recommend_movies_annoy(user_id, annoy_index, normalized_matrix, train_user_movie_matrix,
                                    num_neighbors=10, num_recommendations=5)
print("Annoy-based recommendations for user", user_id, ":", annoy_recs)

print("original KNN model output for user 10:")
print("['Casablanca (1942)', 'Cool Hand Luke (1967)', 'Boot, Das (1981)', 'Star Wars (1977)', 'Thin Man, The (1934)']")