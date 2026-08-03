import pandas as pd

from src.data_preparation import prepare_data
from src.eda import create_movie_statistics


def calculate_weighted_scores(
    movie_statistics: pd.DataFrame,
    quantile: float = 0.75,
) -> pd.DataFrame:
    """Calculate a weighted popularity score for every qualified movie."""

    if not 0 <= quantile <= 1:
        raise ValueError("Quantile must be between 0 and 1.")

    statistics = movie_statistics.dropna(
        subset=["rating_count", "average_rating"]
    ).copy()

    if statistics.empty:
        return statistics.assign(popularity_score=pd.Series(dtype=float))

    total_ratings = statistics["rating_count"].sum()

    global_average = (
        statistics["average_rating"] * statistics["rating_count"]
    ).sum() / total_ratings

    minimum_votes = statistics["rating_count"].quantile(quantile)

    qualified_movies = statistics[
        statistics["rating_count"] >= minimum_votes
    ].copy()

    vote_count = qualified_movies["rating_count"]
    average_rating = qualified_movies["average_rating"]

    qualified_movies["popularity_score"] = (
        vote_count / (vote_count + minimum_votes) * average_rating
        + minimum_votes
        / (vote_count + minimum_votes)
        * global_average
    )

    return qualified_movies.sort_values(
        ["popularity_score", "rating_count"],
        ascending=[False, False],
    )


def recommend_popular_movies(
    movies: pd.DataFrame,
    ratings: pd.DataFrame,
    top_n: int = 10,
    genre: str | None = None,
) -> pd.DataFrame:
    """Return popular movies, optionally filtered by genre."""

    if top_n < 1:
        raise ValueError("top_n must be at least 1.")

    movie_statistics = create_movie_statistics(movies, ratings)

    if genre:
        movie_statistics = movie_statistics[
            movie_statistics["genres"].str.contains(
                genre,
                case=False,
                na=False,
                regex=False,
            )
        ]

    ranked_movies = calculate_weighted_scores(movie_statistics)

    columns = [
        "movieId",
        "clean_title",
        "year",
        "genres",
        "rating_count",
        "average_rating",
        "popularity_score",
    ]

    return ranked_movies[columns].head(top_n).reset_index(drop=True)


def display_recommendations(
    recommendations: pd.DataFrame,
    heading: str,
) -> None:
    """Print formatted movie recommendations."""

    print(f"\n{heading}")
    print("=" * 80)

    display_columns = [
        "clean_title",
        "year",
        "genres",
        "rating_count",
        "average_rating",
        "popularity_score",
    ]

    print(
        recommendations[display_columns].to_string(
            index=False,
            formatters={
                "average_rating": "{:.2f}".format,
                "popularity_score": "{:.3f}".format,
            },
        )
    )


if __name__ == "__main__":
    movies, ratings, _, _, _ = prepare_data()

    overall_recommendations = recommend_popular_movies(
        movies,
        ratings,
        top_n=10,
    )

    action_recommendations = recommend_popular_movies(
        movies,
        ratings,
        top_n=10,
        genre="Action",
    )

    display_recommendations(
        overall_recommendations,
        "TOP 10 POPULAR MOVIES",
    )

    display_recommendations(
        action_recommendations,
        "TOP 10 POPULAR ACTION MOVIES",
    )