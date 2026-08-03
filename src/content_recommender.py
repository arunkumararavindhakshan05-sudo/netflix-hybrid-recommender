import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from src.data_preparation import prepare_data


class ContentBasedRecommender:
    """Recommend movies using genre and tag similarity."""

    def __init__(
        self,
        movies: pd.DataFrame,
        tags: pd.DataFrame,
    ) -> None:
        self.movies = movies.reset_index(drop=True).copy()
        self.tags = tags.copy()

        self._create_movie_features()
        self._train_model()

    def _create_movie_features(self) -> None:
        """Combine each movie's genres and user-generated tags."""

        tag_text = (
            self.tags.dropna(subset=["tag"])
            .groupby("movieId")["tag"]
            .apply(lambda values: " ".join(values.astype(str)))
            .reset_index(name="tag_text")
        )

        self.movies = self.movies.merge(
            tag_text,
            on="movieId",
            how="left",
            validate="one_to_one",
        )

        self.movies["tag_text"] = self.movies["tag_text"].fillna("")

        self.movies["content_features"] = (
            self.movies["genres_text"].fillna("")
            + " "
            + self.movies["tag_text"]
        ).str.lower()

    def _train_model(self) -> None:
        """Convert movie information into TF-IDF vectors."""

        self.vectorizer = TfidfVectorizer(
            stop_words="english",
            ngram_range=(1, 2),
            min_df=1,
        )

        self.feature_matrix = self.vectorizer.fit_transform(
            self.movies["content_features"]
        )

    def search_movies(
        self,
        query: str,
        limit: int = 10,
    ) -> pd.DataFrame:
        """Search movies using a complete or partial title."""

        if not query.strip():
            return self.movies.iloc[0:0][
                ["movieId", "clean_title", "year", "genres"]
            ]

        matches = self.movies[
            self.movies["clean_title"].str.contains(
                query.strip(),
                case=False,
                na=False,
                regex=False,
            )
        ]

        columns = [
            "movieId",
            "clean_title",
            "year",
            "genres",
        ]

        return matches[columns].head(limit).reset_index(drop=True)

    def _find_movie_index(self, title: str) -> int:
        """Find the best movie matching the supplied title."""

        normalized_title = title.strip().casefold()

        exact_matches = self.movies[
            self.movies["clean_title"].str.casefold()
            == normalized_title
        ]

        if not exact_matches.empty:
            return int(exact_matches.index[0])

        partial_matches = self.movies[
            self.movies["clean_title"].str.contains(
                title.strip(),
                case=False,
                na=False,
                regex=False,
            )
        ]

        if partial_matches.empty:
            raise ValueError(
                f"No movie found matching '{title}'. "
                "Try a different or shorter title."
            )

        best_match = partial_matches.sort_values(
            "year",
            ascending=False,
            na_position="last",
        ).iloc[0]

        return int(best_match.name)

    def recommend(
        self,
        title: str,
        top_n: int = 10,
    ) -> pd.DataFrame:
        """Return movies similar to the selected movie."""

        if top_n < 1:
            raise ValueError("top_n must be at least 1.")

        movie_index = self._find_movie_index(title)

        similarity_scores = cosine_similarity(
            self.feature_matrix[movie_index],
            self.feature_matrix,
        ).flatten()

        ranked_indices = similarity_scores.argsort()[::-1]

        ranked_indices = [
            index
            for index in ranked_indices
            if index != movie_index
        ][:top_n]

        recommendations = self.movies.iloc[ranked_indices].copy()

        recommendations["similarity_score"] = similarity_scores[
            ranked_indices
        ]

        columns = [
            "movieId",
            "clean_title",
            "year",
            "genres",
            "similarity_score",
        ]

        return recommendations[columns].reset_index(drop=True)


def display_recommendations(
    selected_title: str,
    recommendations: pd.DataFrame,
) -> None:
    """Display content-based recommendations."""

    print(f"\nBECAUSE YOU WATCHED: {selected_title}")
    print("=" * 90)

    print(
        recommendations.to_string(
            index=False,
            formatters={
                "similarity_score": "{:.3f}".format,
            },
        )
    )


if __name__ == "__main__":
    movies, _, tags, _, _ = prepare_data()

    recommender = ContentBasedRecommender(
        movies,
        tags,
    )

    movie_query = "Matrix"

    matches = recommender.search_movies(movie_query)

    print("\nMOVIE SEARCH RESULTS")
    print("=" * 90)
    print(matches.to_string(index=False))

    recommendations = recommender.recommend(
        movie_query,
        top_n=10,
    )

    display_recommendations(
        movie_query,
        recommendations,
    )