# Movie Recommendation System

A Streamlit-based web application that provides movie recommendations using the MovieLens 100K dataset. The system implements two different recommendation approaches:
- K-Nearest Neighbors (KNN)
- Singular Value Decomposition (SVD)

## Features
- Interactive user interface with Streamlit
- User statistics and data visualization
- Two different recommendation algorithms
- Automatic dataset download and extraction
- Real-time recommendations

## Installation

1. Clone the repository:
```bash
git clone https://github.com/itsJonnie/Movie-Recommendation-System.git
cd Movie-Recommendation-System
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Run the application:
```bash
streamlit run app.py
```

## Usage

1. Select a user ID using the slider in the sidebar
2. Choose between KNN-based or SVD-based recommendation
3. Click "Get Recommendations" to see personalized movie suggestions
4. Explore user statistics and data previews

## Data Source
The application uses the MovieLens 100K dataset, which is automatically downloaded and extracted when you first run the application.
