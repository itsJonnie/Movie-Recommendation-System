# Movie Recommendation System

A **User-Based Collaborative Filtering** movie recommendation system built with Python, Pandas, NumPy, and Scikit-Learn. This project uses the [MovieLens 100k dataset](https://grouplens.org/datasets/movielens/100k/) to suggest movies to users based on similar user preferences. An interactive UI using ipywidgets in Jupyter Notebook/Google Colab allows you to quickly experiment with the model.

---

## Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Installation](#installation)
- [Usage](#usage)
- [Evaluation](#evaluation)
- [Future Enhancements](#future-enhancements)
- [License](#license)
- [Acknowledgments](#acknowledgments)

---

## Overview

This project demonstrates how to build a movie recommendation system using **collaborative filtering**. We:
- Load and preprocess the MovieLens dataset.
- Build a user–movie rating matrix.
- Train a k-Nearest Neighbors (k-NN) model using cosine similarity.
- Generate recommendations for users based on similar users' preferences.
- Evaluate the model using metrics such as Precision@k.
- Provide a simple interactive UI to view recommendations directly in the notebook.

---

## Features

- **Data Preprocessing:**  
  Loads and cleans the MovieLens 100k dataset. Converts Unix timestamps to human-readable dates and extracts useful features like year.

- **User-Based Collaborative Filtering:**  
  Uses a k-NN model to find similar users and recommends movies they liked but the target user hasn’t seen.

- **Interactive UI:**  
  An ipywidgets-based interface lets you input a user ID and dynamically view the top 5 movie recommendations.

- **Evaluation:**  
  Splits data into training and test sets and computes evaluation metrics (e.g., Precision@5) to gauge recommendation quality.

---

## Installation

### Prerequisites

- Python 3.x
- [Jupyter Notebook](https://jupyter.org/) or [Google Colab](https://colab.research.google.com/)
- Required Python packages:
  - pandas
  - numpy
  - scikit-learn
  - matplotlib
  - seaborn
  - ipywidgets

### Setup

1. **Clone the repository:**

   ```bash
   git clone https://github.com/yourusername/Movie-Recommendation-System.git
   cd Movie-Recommendation-System
   ```

2. **Install the required packages:**

   ```bash
   pip install pandas numpy scikit-learn matplotlib seaborn ipywidgets
   ```

3. **Obtain the Dataset:**

   Download the [MovieLens 100k dataset](https://grouplens.org/datasets/movielens/100k/) and place the ZIP file in a designated folder (e.g., `Datasets/zip/`).  
   In Google Colab, you can mount Google Drive and extract the dataset as described in the notebook.

---

## Usage

1. **Open the Notebook:**  
   Open `Movie_Recommendation_System.ipynb` in Jupyter Notebook or Google Colab.

2. **Run Data Preprocessing Cells:**  
   The notebook walks you through:
   - Loading the data.
   - Unzipping and preprocessing the dataset.
   - Creating a user–movie matrix.
   - Converting timestamps and extracting additional features.

3. **Train the Model:**  
   The notebook shows how to initialize and train a k-NN model on the user–movie matrix.

4. **Get Recommendations:**  
   Use the interactive UI provided by ipywidgets to enter a User ID and view the top 5 recommendations.

5. **Evaluate the Model:**  
   The notebook includes code to split data into training and test sets and compute evaluation metrics like Precision@5.

---

## Evaluation

The recommender system is evaluated using a simple **Precision@5** metric. For example, an **Average Precision@5** of `0.15` indicates that, on average, 15% of the recommended movies match the movies rated by the user in the test set. You can adjust the evaluation approach and explore additional metrics such as Recall@k or Mean Average Precision for more robust assessment.

---

## Future Enhancements

- **Item-Based Collaborative Filtering:**  
  Experiment with recommending movies based on similar items rather than similar users.

- **Matrix Factorization:**  
  Implement SVD or other factorization methods to uncover latent factors and potentially improve recommendations.

- **Hybrid Methods:**  
  Combine collaborative filtering with content-based filtering using movie metadata (genres, release dates, etc.) to enhance recommendations.

- **Advanced Evaluation:**  
  Integrate more formal evaluation metrics and user-centric testing methods.

- **Web Deployment:**  
  Build a web application using Flask or Streamlit for a more user-friendly experience.

---

## License

This project is licensed under the [MIT License](LICENSE).

---

## Acknowledgments

- **MovieLens Dataset:** Provided by [GroupLens Research](https://grouplens.org/datasets/movielens/).
- **Inspiration & References:**  
  - Collaborative filtering concepts from various academic and online resources.
  - Example implementations from numerous machine learning tutorials.

---

Feel free to customize this README to better fit your project's specifics. Enjoy building and improving your Movie Recommendation System!
