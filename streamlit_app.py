
# streamlit_app.py
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import plotly.express as px

# Page config
st.set_page_config(
    page_title="Movie Recommender",
    page_icon="🎬",
    layout="wide"
)

@st.cache_data
def load_data():
    """Create simple movie dataset"""
    np.random.seed(42)

    movies = {
        'title': [
            'The Matrix', 'Inception', 'Interstellar', 'The Dark Knight', 'Pulp Fiction',
            'Fight Club', 'Forrest Gump', 'The Godfather', 'Titanic', 'Avatar',
            'Star Wars', 'Jurassic Park', 'Terminator 2', 'Alien', 'Blade Runner',
            'The Shining', 'Psycho', 'Casablanca', 'The Avengers', 'Iron Man',
            'Spider-Man', 'Batman Begins', 'The Lion King', 'Toy Story', 'Finding Nemo',
            'Shrek', 'Frozen', 'Up', 'WALL-E', 'Inside Out'
        ],
        'genre': [
            'Sci-Fi', 'Sci-Fi', 'Sci-Fi', 'Action', 'Crime',
            'Drama', 'Drama', 'Crime', 'Romance', 'Sci-Fi',
            'Sci-Fi', 'Adventure', 'Action', 'Horror', 'Sci-Fi',
            'Horror', 'Thriller', 'Romance', 'Action', 'Action',
            'Action', 'Action', 'Animation', 'Animation', 'Animation',
            'Animation', 'Animation', 'Animation', 'Animation', 'Animation'
        ],
        'year': [
            1999, 2010, 2014, 2008, 1994,
            1999, 1994, 1972, 1997, 2009,
            1977, 1993, 1991, 1979, 1982,
            1980, 1960, 1942, 2012, 2008,
            2002, 2005, 1994, 1995, 2003,
            2001, 2013, 2009, 2008, 2015
        ],
        'rating': [8.7, 8.8, 8.6, 9.0, 8.9, 8.8, 8.8, 9.2, 7.8, 7.8,
                  8.6, 8.1, 8.5, 8.4, 8.1, 8.4, 8.5, 8.5, 8.0, 7.9,
                  7.3, 8.2, 8.5, 8.3, 8.2, 7.9, 7.4, 8.2, 8.4, 8.1],
        'description': [
            'A hacker discovers reality is a simulation',
            'Thieves enter dreams to steal secrets',
            'Astronauts travel through a wormhole',
            'Batman fights the Joker in Gotham',
            'Interconnected crime stories in LA',
            'An insomniac office worker starts a fight club',
            'Life story of a simple man with low IQ',
            'The aging patriarch of a crime dynasty',
            'A romance aboard the doomed ship',
            'Humans colonize an alien planet',
            'Rebels fight against the evil Empire',
            'Dinosaurs are brought back to life',
            'A cyborg is sent back to protect John Connor',
            'The crew encounters a deadly alien creature',
            'A blade runner hunts replicants',
            'A family stays at a haunted hotel',
            'A secretary embezzles money and flees',
            'A classic wartime romance',
            'Superheroes team up to save Earth',
            'A billionaire becomes a superhero',
            'A teenager gains spider powers',
            'Bruce Wayne becomes Batman',
            'A lion cub becomes king',
            'Toys come to life when humans leave',
            'A fish searches for his lost son',
            'An ogre goes on a quest',
            'A princess with ice powers',
            'An old man travels with a boy scout',
            'A robot left on Earth finds love',
            'Emotions control a young girl'
        ]
    }

    return pd.DataFrame(movies)

def get_recommendations(movie_title, movies_df, num_recommendations=5):
    """Get movie recommendations based on content similarity"""
    # Combine genre and description
    movies_df['content'] = movies_df['genre'] + ' ' + movies_df['description']

    # Create TF-IDF matrix
    tfidf = TfidfVectorizer(stop_words='english')
    tfidf_matrix = tfidf.fit_transform(movies_df['content'])

    # Calculate similarity
    cosine_sim = cosine_similarity(tfidf_matrix)

    # Get movie index
    try:
        idx = movies_df[movies_df['title'] == movie_title].index[0]
    except:
        return pd.DataFrame()

    # Get similarity scores
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)

    # Get top movies (excluding the selected movie)
    movie_indices = [i[0] for i in sim_scores[1:num_recommendations+1]]

    recommendations = movies_df.iloc[movie_indices].copy()
    recommendations['similarity'] = [sim_scores[i+1][1] for i in range(num_recommendations)]

    return recommendations

def main():
    st.title("🎬 Simple Movie Recommender")
    st.markdown("Find movies similar to your favorites!")

    # Load data
    movies_df = load_data()

    # Sidebar
    st.sidebar.title("Options")

    # Movie selection
    selected_movie = st.selectbox(
        "Choose a movie you like:",
        movies_df['title'].tolist()
    )

    num_recs = st.sidebar.slider("Number of recommendations:", 1, 10, 5)

    # Show selected movie info
    selected_info = movies_df[movies_df['title'] == selected_movie].iloc[0]

    st.subheader(f"Selected Movie: {selected_movie}")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.write(f"**Genre:** {selected_info['genre']}")
    with col2:
        st.write(f"**Year:** {selected_info['year']}")
    with col3:
        st.write(f"**Rating:** ⭐ {selected_info['rating']}")

    st.write(f"**Plot:** {selected_info['description']}")

    # Get recommendations
    if st.button("Get Recommendations", type="primary"):
        recommendations = get_recommendations(selected_movie, movies_df, num_recs)

        if not recommendations.empty:
            st.subheader("🎯 Recommended Movies:")

            for idx, movie in recommendations.iterrows():
                with st.expander(f"🎬 {movie['title']} (Similarity: {movie['similarity']:.2f})"):
                    col1, col2 = st.columns([2, 1])
                    with col1:
                        st.write(f"**Genre:** {movie['genre']}")
                        st.write(f"**Year:** {movie['year']}")
                        st.write(f"**Rating:** ⭐ {movie['rating']}")
                        st.write(f"**Plot:** {movie['description']}")
                    with col2:
                        st.metric("Similarity Score", f"{movie['similarity']:.2f}")
        else:
            st.error("No recommendations found!")

    # Show all movies
    if st.sidebar.checkbox("Show all movies"):
        st.subheader("📚 Movie Database")
        st.dataframe(movies_df[['title', 'genre', 'year', 'rating']], use_container_width=True)

    # Simple analytics
    if st.sidebar.checkbox("Show analytics"):
        st.subheader("📊 Movie Analytics")

        # Genre distribution
        genre_counts = movies_df['genre'].value_counts()
        fig = px.pie(values=genre_counts.values, names=genre_counts.index, title="Movies by Genre")
        st.plotly_chart(fig, use_container_width=True)

        # Ratings over time
        fig2 = px.scatter(movies_df, x='year', y='rating', color='genre', 
                         title="Movie Ratings Over Time", hover_data=['title'])
        st.plotly_chart(fig2, use_container_width=True)

if __name__ == "__main__":
    main()
