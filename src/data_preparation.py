import pandas as pd

from src.data_loader import load_data


def prepare_movies(movies: pd.DataFrame) -> pd.DataFrame:
    """Clean movie titles and prepare genre information."""

    cleaned_movies = movies.copy()

    # Extract the four-digit release year from the end of each title.
    cleaned_movies["year"] = (
        cleaned_movies["title"]
        .str.extract(r"\((\d{4})\)$", expand=False)
        .astype("Int64")
    )

    # Remove the release year to produce a searchable movie title.
    cleaned_movies["clean_title"] = (
        cleaned_movies["title"]
        .str.replace(r"\s*\(\d{4}\)$", "", regex=True)
        .str.strip()
    )

    # Convert pipe-separated genres into text suitable for ML vectorization.
    cleaned_movies["genres_text"] = (
        cleaned_movies["genres"]
        .replace("(no genres listed)", "")
        .str.replace("|", " ", regex=False)
    )

    return cleaned_movies


def prepare_ratings(ratings: pd.DataFrame) -> pd.DataFrame:
    """Convert rating timestamps into readable dates."""

    cleaned_ratings = ratings.copy()

    cleaned_ratings["rated_at"] = pd.to_datetime(
        cleaned_ratings["timestamp"],
        unit="s",
    )

    return cleaned_ratings


def create_movie_ratings_dataset(
    movies: pd.DataFrame,
    ratings: pd.DataFrame,
) -> pd.DataFrame:
    """Combine movie information with user ratings."""

    return ratings.merge(
        movies,
        on="movieId",
        how="inner",
        validate="many_to_one",
    )


def prepare_data():
    """Load and prepare all datasets used by the recommender."""

    movies, ratings, tags, links = load_data()

    movies = prepare_movies(movies)
    ratings = prepare_ratings(ratings)

    movie_ratings = create_movie_ratings_dataset(movies, ratings)

    return movies, ratings, tags, links, movie_ratings


if __name__ == "__main__":
    movies, ratings, tags, links, movie_ratings = prepare_data()

    print("Data preparation successful")
    print(f"Movies: {len(movies):,}")
    print(f"Ratings: {len(ratings):,}")
    print(f"Users: {ratings['userId'].nunique():,}")
    print(f"Rated movies: {ratings['movieId'].nunique():,}")
    print(f"Combined records: {len(movie_ratings):,}")

    print("\nPrepared movie columns:")
    print(movies.head())

    print("\nRating date range:")
    print(f"From: {ratings['rated_at'].min()}")
    print(f"To:   {ratings['rated_at'].max()}")