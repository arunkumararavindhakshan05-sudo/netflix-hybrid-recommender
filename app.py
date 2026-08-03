from html import escape

import pandas as pd
import streamlit as st

from src.data_preparation import prepare_data
from src.hybrid_recommender import HybridRecommender
from src.popularity_recommender import recommend_popular_movies


st.set_page_config(
    page_title="CineMatch",
    page_icon="🎬",
    layout="wide",
)


st.markdown(
    """
    <style>
    .stApp {
        background:
            linear-gradient(
                180deg,
                #080808 0%,
                #101010 45%,
                #080808 100%
            );
        color: #ffffff;
    }

    .main-title {
        color: #e50914;
        font-size: 3rem;
        font-weight: 800;
        letter-spacing: -2px;
        margin-bottom: 0;
    }

    .subtitle {
        color: #b3b3b3;
        font-size: 1.05rem;
        margin-bottom: 1.5rem;
    }

    .section-title {
        color: #ffffff;
        font-size: 1.5rem;
        font-weight: 700;
        margin-top: 1.2rem;
        margin-bottom: 0.8rem;
    }

    .movie-card {
        background: linear-gradient(
            145deg,
            #242424,
            #171717
        );
        border: 1px solid #303030;
        border-radius: 10px;
        min-height: 230px;
        padding: 16px;
        margin-bottom: 18px;
        transition:
            transform 0.2s ease,
            border-color 0.2s ease;
    }

    .movie-card:hover {
        transform: translateY(-5px);
        border-color: #e50914;
    }

    .movie-icon {
        font-size: 2.4rem;
        margin-bottom: 12px;
    }

    .movie-title {
        color: #ffffff;
        font-size: 1.05rem;
        font-weight: 700;
        line-height: 1.25;
        margin-bottom: 8px;
    }

    .movie-year {
        color: #46d369;
        font-size: 0.85rem;
        font-weight: 600;
    }

    .movie-genres {
        color: #b3b3b3;
        font-size: 0.78rem;
        line-height: 1.35;
        margin-top: 8px;
    }

    .movie-score {
        color: #f5c518;
        font-size: 0.83rem;
        font-weight: 600;
        margin-top: 12px;
    }

    .movie-reason {
        color: #dddddd;
        font-size: 0.75rem;
        font-style: italic;
        margin-top: 8px;
    }

    div.stButton > button {
        background-color: #e50914;
        border: none;
        color: white;
        font-weight: 700;
    }

    div.stButton > button:hover {
        background-color: #f6121d;
        color: white;
        border: none;
    }

    [data-testid="stSidebar"] {
        background-color: #111111;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource(show_spinner="Preparing recommendation models...")
def load_recommendation_system():
    """Load the data and train recommendation models once."""

    movies, ratings, tags, links, _ = prepare_data()

    hybrid_model = HybridRecommender(
        movies,
        ratings,
        tags,
        n_factors=50,
    )

    return movies, ratings, links, hybrid_model


def create_movie_options(movies: pd.DataFrame) -> pd.DataFrame:
    """Create readable movie labels for the selection box."""

    options = movies[
        ["movieId", "clean_title", "year"]
    ].copy()

    options["year_label"] = options["year"].apply(
        lambda year: "Unknown"
        if pd.isna(year)
        else str(int(year))
    )

    options["label"] = (
        options["clean_title"]
        + " ("
        + options["year_label"]
        + ")"
    )

    return options.sort_values(
        ["clean_title", "year"],
        na_position="last",
    ).reset_index(drop=True)


def render_movie_cards(
    movies: pd.DataFrame,
    score_column: str | None = None,
    score_label: str = "Score",
    reason_column: str | None = None,
) -> None:
    """Render movie recommendations as responsive cards."""

    if movies.empty:
        st.info("No movies matched the selected criteria.")
        return

    for start in range(0, len(movies), 5):
        row_movies = movies.iloc[start : start + 5]
        columns = st.columns(5)

        for column, movie in zip(
            columns,
            row_movies.itertuples(index=False),
            strict=False,
        ):
            title = escape(str(movie.clean_title))

            year = (
                "Unknown year"
                if pd.isna(movie.year)
                else str(int(movie.year))
            )

            genres = escape(
                str(movie.genres).replace("|", " • ")
            )

            score_html = ""

            if score_column:
                score = getattr(movie, score_column)

                if not pd.isna(score):
                    score_html = (
                        '<div class="movie-score">'
                        f"{escape(score_label)}: {float(score):.3f}"
                        "</div>"
                    )

            reason_html = ""

            if reason_column:
                reason = getattr(movie, reason_column, "")

                if reason:
                    reason_html = (
                        '<div class="movie-reason">'
                        f"{escape(str(reason))}"
                        "</div>"
                    )

            card = f"""
                <div class="movie-card">
                    <div class="movie-icon">🎬</div>
                    <div class="movie-title">{title}</div>
                    <div class="movie-year">{year}</div>
                    <div class="movie-genres">{genres}</div>
                    {score_html}
                    {reason_html}
                </div>
            """

            with column:
                st.markdown(
                    card,
                    unsafe_allow_html=True,
                )


movies, ratings, links, hybrid_model = (
    load_recommendation_system()
)

movie_options = create_movie_options(movies)

st.markdown(
    '<div class="main-title">CineMatch</div>',
    unsafe_allow_html=True,
)

st.markdown(
    """
    <div class="subtitle">
        Hybrid movie recommendations powered by content similarity,
        collaborative filtering and audience popularity.
    </div>
    """,
    unsafe_allow_html=True,
)


with st.sidebar:
    st.header("Recommendation Settings")

    user_id = st.selectbox(
        "MovieLens user",
        options=sorted(ratings["userId"].unique()),
        index=0,
    )

    recommendation_count = st.slider(
        "Number of recommendations",
        min_value=5,
        max_value=20,
        value=10,
    )

    st.subheader("Hybrid weights")

    collaborative_weight = st.slider(
        "Collaborative filtering",
        min_value=0.0,
        max_value=1.0,
        value=0.50,
        step=0.05,
    )

    content_weight = st.slider(
        "Content similarity",
        min_value=0.0,
        max_value=1.0,
        value=0.40,
        step=0.05,
    )

    popularity_weight = st.slider(
        "Popularity",
        min_value=0.0,
        max_value=1.0,
        value=0.10,
        step=0.05,
    )

    total_weight = (
        collaborative_weight
        + content_weight
        + popularity_weight
    )

    st.caption(
        f"Current weight total: {total_weight:.2f}. "
        "Weights are normalized automatically."
    )


hybrid_tab, similar_tab, popular_tab = st.tabs(
    [
        "For You",
        "Similar Movies",
        "Popular",
    ]
)


with hybrid_tab:
    st.markdown(
        '<div class="section-title">Personalized for You</div>',
        unsafe_allow_html=True,
    )

    selected_label = st.selectbox(
        "Choose a movie you enjoyed",
        options=movie_options["label"].tolist(),
        index=0,
        key="hybrid_movie",
    )

    selected_movie = movie_options.loc[
        movie_options["label"] == selected_label
    ].iloc[0]

    selected_title = selected_movie["clean_title"]

    if total_weight == 0:
        st.warning(
            "Increase at least one recommendation weight."
        )

    elif st.button(
        "Generate Hybrid Recommendations",
        use_container_width=True,
    ):
        try:
            with st.spinner(
                "Analysing movies and user preferences..."
            ):
                recommendations = hybrid_model.recommend(
                    user_id=int(user_id),
                    seed_title=selected_title,
                    top_n=recommendation_count,
                    content_weight=content_weight,
                    collaborative_weight=collaborative_weight,
                    popularity_weight=popularity_weight,
                )

            render_movie_cards(
                recommendations,
                score_column="hybrid_score",
                score_label="Hybrid score",
                reason_column="recommendation_reason",
            )

            with st.expander(
                "View this user's highest-rated movies"
            ):
                history = (
                    hybrid_model.collaborative_model
                    .get_user_history(
                        int(user_id),
                        top_n=10,
                    )
                )

                st.dataframe(
                    history,
                    hide_index=True,
                    use_container_width=True,
                )

        except ValueError as error:
            st.error(str(error))


with similar_tab:
    st.markdown(
        '<div class="section-title">Because You Watched...</div>',
        unsafe_allow_html=True,
    )

    content_label = st.selectbox(
        "Select a movie",
        options=movie_options["label"].tolist(),
        index=0,
        key="content_movie",
    )

    content_movie = movie_options.loc[
        movie_options["label"] == content_label
    ].iloc[0]

    if st.button(
        "Find Similar Movies",
        use_container_width=True,
    ):
        recommendations = (
            hybrid_model.content_model.recommend(
                content_movie["clean_title"],
                top_n=recommendation_count,
            )
        )

        render_movie_cards(
            recommendations,
            score_column="similarity_score",
            score_label="Similarity",
        )


with popular_tab:
    st.markdown(
        '<div class="section-title">Popular on CineMatch</div>',
        unsafe_allow_html=True,
    )

    genre_values = sorted(
        {
            genre
            for genres in movies["genres"].dropna()
            for genre in genres.split("|")
            if genre != "(no genres listed)"
        }
    )

    selected_genre = st.selectbox(
        "Filter by genre",
        options=["All Genres", *genre_values],
    )

    genre_filter = (
        None
        if selected_genre == "All Genres"
        else selected_genre
    )

    popular_movies = recommend_popular_movies(
        movies,
        ratings,
        top_n=recommendation_count,
        genre=genre_filter,
    )

    render_movie_cards(
        popular_movies,
        score_column="popularity_score",
        score_label="Popularity",
    )


st.divider()

st.caption(
    "Built with Python, MovieLens, Scikit-learn, SVD, "
    "TF-IDF, cosine similarity and Streamlit."
)