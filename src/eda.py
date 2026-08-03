from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from src.data_preparation import prepare_data

OUTPUT_DIR = Path("artifacts") / "eda"


def create_movie_statistics(
    movies: pd.DataFrame,
    ratings: pd.DataFrame,
) -> pd.DataFrame:
    """Calculate rating statistics for each movie."""

    rating_statistics = (
        ratings.groupby("movieId")
        .agg(
            rating_count=("rating", "count"),
            average_rating=("rating", "mean"),
        )
        .reset_index()
    )

    return movies.merge(
        rating_statistics,
        on="movieId",
        how="left",
        validate="one_to_one",
    )


def create_genre_statistics(movies: pd.DataFrame) -> pd.DataFrame:
    """Count the number of movies belonging to each genre."""

    movie_genres = movies[["movieId", "genres"]].copy()
    movie_genres["genre"] = movie_genres["genres"].str.split("|")
    movie_genres = movie_genres.explode("genre")

    movie_genres = movie_genres[
        movie_genres["genre"] != "(no genres listed)"
    ]

    return (
        movie_genres.groupby("genre")
        .size()
        .reset_index(name="movie_count")
        .sort_values("movie_count", ascending=False)
    )


def save_rating_distribution(ratings: pd.DataFrame) -> None:
    """Create a chart showing the distribution of user ratings."""

    plt.figure(figsize=(9, 5))

    sns.countplot(
        data=ratings,
        x="rating",
        color="#e50914",
    )

    plt.title("MovieLens Rating Distribution")
    plt.xlabel("Rating")
    plt.ylabel("Number of Ratings")
    plt.tight_layout()

    plt.savefig(
        OUTPUT_DIR / "rating_distribution.png",
        dpi=150,
    )

    plt.close()


def save_top_genres(genre_statistics: pd.DataFrame) -> None:
    """Create a chart showing the most common movie genres."""

    top_genres = genre_statistics.head(12)

    plt.figure(figsize=(10, 6))

    sns.barplot(
        data=top_genres,
        x="movie_count",
        y="genre",
        color="#e50914",
    )

    plt.title("Most Common Movie Genres")
    plt.xlabel("Number of Movies")
    plt.ylabel("Genre")
    plt.tight_layout()

    plt.savefig(
        OUTPUT_DIR / "top_genres.png",
        dpi=150,
    )

    plt.close()


def save_most_rated_movies(movie_statistics: pd.DataFrame) -> None:
    """Create a chart showing movies with the most ratings."""

    most_rated = movie_statistics.nlargest(
        10,
        "rating_count",
    ).sort_values("rating_count")

    plt.figure(figsize=(10, 6))

    sns.barplot(
        data=most_rated,
        x="rating_count",
        y="clean_title",
        color="#e50914",
    )

    plt.title("Most Rated Movies")
    plt.xlabel("Number of Ratings")
    plt.ylabel("Movie")
    plt.tight_layout()

    plt.savefig(
        OUTPUT_DIR / "most_rated_movies.png",
        dpi=150,
    )

    plt.close()


def display_summary(
    movies: pd.DataFrame,
    ratings: pd.DataFrame,
    movie_statistics: pd.DataFrame,
    genre_statistics: pd.DataFrame,
) -> None:
    """Display useful statistics in the terminal."""

    print("\nDATASET SUMMARY")
    print("=" * 60)
    print(f"Movies: {len(movies):,}")
    print(f"Ratings: {len(ratings):,}")
    print(f"Users: {ratings['userId'].nunique():,}")
    print(f"Average rating: {ratings['rating'].mean():.2f}")

    print("\nMOST RATED MOVIES")
    print("=" * 60)
    print(
        movie_statistics.nlargest(10, "rating_count")[
            ["clean_title", "year", "rating_count", "average_rating"]
        ].to_string(index=False)
    )

    # Require at least 50 ratings to prevent a movie with one
    # five-star rating from being treated as the best movie.
    qualified_movies = movie_statistics[
        movie_statistics["rating_count"] >= 50
    ]

    print("\nHIGHEST-RATED MOVIES (MINIMUM 50 RATINGS)")
    print("=" * 60)
    print(
        qualified_movies.nlargest(10, "average_rating")[
            ["clean_title", "year", "rating_count", "average_rating"]
        ].to_string(index=False)
    )

    print("\nMOST COMMON GENRES")
    print("=" * 60)
    print(genre_statistics.head(10).to_string(index=False))


def run_eda() -> None:
    """Run the complete exploratory data analysis."""

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    movies, ratings, _, _, _ = prepare_data()

    movie_statistics = create_movie_statistics(
        movies,
        ratings,
    )

    genre_statistics = create_genre_statistics(movies)

    display_summary(
        movies,
        ratings,
        movie_statistics,
        genre_statistics,
    )

    save_rating_distribution(ratings)
    save_top_genres(genre_statistics)
    save_most_rated_movies(movie_statistics)

    print(f"\nCharts saved in: {OUTPUT_DIR.resolve()}")


if __name__ == "__main__":
    run_eda()