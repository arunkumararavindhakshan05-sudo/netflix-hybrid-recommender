import pandas as pd

from src.data_preparation import (
    create_movie_ratings_dataset,
    prepare_movies,
    prepare_ratings,
)


def test_prepare_movies_extracts_title_year_and_genres():
    movies = pd.DataFrame(
        {
            "movieId": [1, 2],
            "title": ["Toy Story (1995)", "Jumanji (1995)"],
            "genres": [
                "Adventure|Animation|Comedy",
                "Adventure|Children|Fantasy",
            ],
        }
    )

    result = prepare_movies(movies)

    assert result["year"].tolist() == [1995, 1995]
    assert result["clean_title"].tolist() == ["Toy Story", "Jumanji"]
    assert result["genres_text"].tolist() == [
        "Adventure Animation Comedy",
        "Adventure Children Fantasy",
    ]


def test_prepare_ratings_converts_timestamp():
    ratings = pd.DataFrame(
        {
            "userId": [1],
            "movieId": [1],
            "rating": [4.0],
            "timestamp": [0],
        }
    )

    result = prepare_ratings(ratings)

    assert "rated_at" in result.columns
    assert result.loc[0, "rated_at"] == pd.Timestamp("1970-01-01")


def test_create_movie_ratings_dataset_merges_records():
    movies = pd.DataFrame(
        {
            "movieId": [1],
            "title": ["Toy Story (1995)"],
            "genres": ["Adventure|Animation|Comedy"],
        }
    )

    ratings = pd.DataFrame(
        {
            "userId": [1, 2],
            "movieId": [1, 1],
            "rating": [4.0, 5.0],
            "timestamp": [0, 1],
        }
    )

    result = create_movie_ratings_dataset(movies, ratings)

    assert len(result) == 2
    assert result["title"].tolist() == [
        "Toy Story (1995)",
        "Toy Story (1995)",
    ]