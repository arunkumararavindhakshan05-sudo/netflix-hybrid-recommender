import pandas as pd
import pytest

from src.content_recommender import ContentBasedRecommender


@pytest.fixture
def recommender():
    movies = pd.DataFrame(
        {
            "movieId": [1, 2, 3, 4],
            "clean_title": [
                "The Matrix",
                "The Terminator",
                "The Notebook",
                "Toy Story",
            ],
            "year": [1999, 1984, 2004, 1995],
            "genres": [
                "Action|Sci-Fi|Thriller",
                "Action|Sci-Fi|Thriller",
                "Drama|Romance",
                "Adventure|Animation|Comedy",
            ],
            "genres_text": [
                "Action Sci-Fi Thriller",
                "Action Sci-Fi Thriller",
                "Drama Romance",
                "Adventure Animation Comedy",
            ],
        }
    )

    tags = pd.DataFrame(
        {
            "movieId": [1, 1, 2, 3, 4],
            "tag": [
                "artificial intelligence",
                "dystopian",
                "cyborg dystopian",
                "romantic",
                "family animation",
            ],
        }
    )

    return ContentBasedRecommender(movies, tags)


def test_similar_movie_is_recommended(recommender):
    result = recommender.recommend("The Matrix", top_n=1)

    assert result.iloc[0]["clean_title"] == "The Terminator"


def test_selected_movie_is_excluded(recommender):
    result = recommender.recommend("The Matrix", top_n=3)

    assert "The Matrix" not in result["clean_title"].tolist()


def test_partial_title_search(recommender):
    result = recommender.search_movies("Matrix")

    assert len(result) == 1
    assert result.iloc[0]["clean_title"] == "The Matrix"


def test_unknown_movie_raises_error(recommender):
    with pytest.raises(ValueError, match="No movie found"):
        recommender.recommend("Unknown Movie")


def test_invalid_top_n_raises_error(recommender):
    with pytest.raises(ValueError, match="top_n"):
        recommender.recommend(
            "The Matrix",
            top_n=0,
        )