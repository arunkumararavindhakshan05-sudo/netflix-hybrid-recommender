import pandas as pd
import pytest

from src.hybrid_recommender import HybridRecommender


@pytest.fixture
def recommender():
    movies = pd.DataFrame(
        {
            "movieId": [1, 2, 3, 4, 5],
            "clean_title": [
                "The Matrix",
                "The Terminator",
                "The Notebook",
                "Toy Story",
                "Blade Runner",
            ],
            "year": [1999, 1984, 2004, 1995, 1982],
            "genres": [
                "Action|Sci-Fi|Thriller",
                "Action|Sci-Fi|Thriller",
                "Drama|Romance",
                "Adventure|Animation|Comedy",
                "Action|Sci-Fi",
            ],
            "genres_text": [
                "Action Sci-Fi Thriller",
                "Action Sci-Fi Thriller",
                "Drama Romance",
                "Adventure Animation Comedy",
                "Action Sci-Fi",
            ],
        }
    )

    tags = pd.DataFrame(
        {
            "movieId": [1, 2, 3, 4, 5],
            "tag": [
                "artificial intelligence dystopian",
                "cyborg dystopian",
                "romantic",
                "family animation",
                "artificial intelligence dystopian",
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
                4,
                1,
                2,
                5,
                2,
                3,
                5,
                2,
                3,
                5,
            ],
            "rating": [
                5.0,
                3.0,
                4.5,
                5.0,
                4.5,
                4.0,
                2.5,
                5.0,
                4.5,
                2.0,
                5.0,
            ],
        }
    )

    return HybridRecommender(
        movies,
        ratings,
        tags,
        n_factors=2,
    )


def test_hybrid_recommendations_exclude_watched_movies(recommender):
    results = recommender.recommend(
        user_id=1,
        seed_title="The Matrix",
        top_n=3,
    )

    watched_movie_ids = {1, 4}

    assert not set(results["movieId"]).intersection(
        watched_movie_ids
    )


def test_hybrid_results_are_sorted(recommender):
    results = recommender.recommend(
        user_id=1,
        seed_title="The Matrix",
        top_n=3,
    )

    scores = results["hybrid_score"].tolist()

    assert scores == sorted(scores, reverse=True)


def test_hybrid_results_include_explanations(recommender):
    results = recommender.recommend(
        user_id=1,
        seed_title="The Matrix",
        top_n=3,
    )

    assert results["recommendation_reason"].notna().all()


def test_zero_weights_raise_error(recommender):
    with pytest.raises(ValueError, match="At least one"):
        recommender.recommend(
            user_id=1,
            seed_title="The Matrix",
            content_weight=0,
            collaborative_weight=0,
            popularity_weight=0,
        )