import pandas as pd
import pytest

from src.collaborative_recommender import CollaborativeRecommender


@pytest.fixture
def recommender():
    movies = pd.DataFrame(
        {
            "movieId": [1, 2, 3, 4, 5],
            "clean_title": [
                "Action One",
                "Action Two",
                "Drama One",
                "Drama Two",
                "Shared Favourite",
            ],
            "year": [2000, 2001, 2002, 2003, 2004],
            "genres": [
                "Action",
                "Action|Thriller",
                "Drama",
                "Drama|Romance",
                "Action|Drama",
            ],
        }
    )

    ratings = pd.DataFrame(
        {
            "userId": [
                1,
                1,
                2,
                2,
                2,
                3,
                3,
                3,
                4,
                4,
                4,
            ],
            "movieId": [
                1,
                2,
                1,
                2,
                5,
                3,
                4,
                5,
                3,
                4,
                5,
            ],
            "rating": [
                5.0,
                4.0,
                4.5,
                5.0,
                4.0,
                5.0,
                4.5,
                3.0,
                4.0,
                5.0,
                3.5,
            ],
        }
    )

    return CollaborativeRecommender(
        movies,
        ratings,
        n_factors=2,
    )


def test_recommendations_exclude_already_rated_movies(recommender):
    results = recommender.recommend(
        user_id=1,
        top_n=3,
    )

    rated_movie_ids = {1, 2}

    assert not set(results["movieId"]).intersection(rated_movie_ids)
    assert "predicted_rating" in results.columns


def test_predicted_ratings_are_in_valid_range(recommender):
    results = recommender.recommend(
        user_id=1,
        top_n=3,
    )

    assert results["predicted_rating"].between(0.5, 5.0).all()


def test_unknown_user_raises_error(recommender):
    with pytest.raises(ValueError, match="Unknown user"):
        recommender.recommend(
            user_id=999,
            top_n=3,
        )


def test_invalid_top_n_raises_error(recommender):
    with pytest.raises(ValueError, match="top_n"):
        recommender.recommend(
            user_id=1,
            top_n=0,
        )


def test_user_history_contains_only_selected_user_movies(recommender):
    history = recommender.get_user_history(
        user_id=1,
        top_n=10,
    )

    assert set(history["movieId"]) == {1, 2}
    assert history.iloc[0]["rating"] >= history.iloc[-1]["rating"]