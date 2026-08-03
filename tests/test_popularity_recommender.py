import pandas as pd
import pytest

from src.popularity_recommender import (
    calculate_weighted_scores,
    recommend_popular_movies,
)


def test_weighted_score_excludes_movies_with_few_ratings():
    statistics = pd.DataFrame(
        {
            "movieId": [1, 2, 3],
            "clean_title": ["Movie A", "Movie B", "Movie C"],
            "year": [2000, 2001, 2002],
            "genres": ["Drama", "Action", "Comedy"],
            "rating_count": [100, 80, 2],
            "average_rating": [4.5, 4.6, 5.0],
        }
    )

    result = calculate_weighted_scores(
        statistics,
        quantile=0.5,
    )

    assert set(result["movieId"]) == {1, 2}
    assert 3 not in result["movieId"].tolist()
    assert "popularity_score" in result.columns


def test_recommendations_can_be_filtered_by_genre():
    movies = pd.DataFrame(
        {
            "movieId": [1, 2, 3],
            "clean_title": ["Action One", "Drama One", "Action Two"],
            "year": [2000, 2001, 2002],
            "genres": ["Action|Drama", "Drama", "Action|Comedy"],
        }
    )

    ratings = pd.DataFrame(
        {
            "userId": [1, 2, 3, 1, 2, 3],
            "movieId": [1, 1, 1, 2, 3, 3],
            "rating": [5.0, 4.0, 4.5, 5.0, 4.0, 4.5],
        }
    )

    result = recommend_popular_movies(
        movies,
        ratings,
        top_n=10,
        genre="Action",
    )

    assert not result.empty
    assert result["genres"].str.contains("Action").all()


def test_invalid_quantile_raises_error():
    statistics = pd.DataFrame()

    with pytest.raises(ValueError):
        calculate_weighted_scores(
            statistics,
            quantile=1.5,
        )


def test_invalid_top_n_raises_error():
    movies = pd.DataFrame()
    ratings = pd.DataFrame()

    with pytest.raises(ValueError):
        recommend_popular_movies(
            movies,
            ratings,
            top_n=0,
        )