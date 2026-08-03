import pandas as pd

from src.collaborative_recommender import CollaborativeRecommender
from src.content_recommender import ContentBasedRecommender
from src.data_preparation import prepare_data
from src.popularity_recommender import recommend_popular_movies


class HybridRecommender:
    """Combine content, collaborative and popularity recommendations."""

    def __init__(
        self,
        movies: pd.DataFrame,
        ratings: pd.DataFrame,
        tags: pd.DataFrame,
        n_factors: int = 50,
    ) -> None:
        self.movies = movies.copy()
        self.ratings = ratings.copy()

        self.content_model = ContentBasedRecommender(
            movies,
            tags,
        )

        self.collaborative_model = CollaborativeRecommender(
            movies,
            ratings,
            n_factors=n_factors,
        )

    @staticmethod
    def _normalize_popularity(
        scores: pd.Series,
    ) -> pd.Series:
        """Normalize popularity scores to a range between zero and one."""

        normalized = pd.Series(
            0.0,
            index=scores.index,
            dtype=float,
        )

        valid_scores = scores.dropna()

        if valid_scores.empty:
            return normalized

        minimum = valid_scores.min()
        maximum = valid_scores.max()

        if maximum == minimum:
            normalized.loc[valid_scores.index] = 1.0
            return normalized

        normalized.loc[valid_scores.index] = (
            valid_scores - minimum
        ) / (maximum - minimum)

        return normalized

    def _find_seed_movie_id(self, title: str) -> int:
        """Resolve the selected movie title into a MovieLens movie ID."""

        normalized_title = title.strip().casefold()

        exact_matches = self.movies[
            self.movies["clean_title"].str.casefold()
            == normalized_title
        ]

        if not exact_matches.empty:
            return int(exact_matches.iloc[0]["movieId"])

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
                f"No movie found matching '{title}'."
            )

        best_match = partial_matches.sort_values(
            "year",
            ascending=False,
            na_position="last",
        ).iloc[0]

        return int(best_match["movieId"])

    @staticmethod
    def _create_reason(
        row: pd.Series,
        seed_title: str,
    ) -> str:
        """Create a simple explanation for a recommendation."""

        content_score = row["content_score"]
        collaborative_score = row["collaborative_score"]
        popularity_score = row["normalized_popularity_score"]

        if content_score >= collaborative_score and content_score > 0:
            return f"Similar to {seed_title}"

        if collaborative_score > 0:
            return "Matches your rating preferences"

        if popularity_score > 0:
            return "Popular with MovieLens users"

        return "Recommended for you"

    def recommend(
        self,
        user_id: int,
        seed_title: str,
        top_n: int = 10,
        content_weight: float = 0.40,
        collaborative_weight: float = 0.50,
        popularity_weight: float = 0.10,
        candidate_pool: int = 100,
    ) -> pd.DataFrame:
        """Return weighted hybrid recommendations."""

        if top_n < 1:
            raise ValueError("top_n must be at least 1.")

        weights = [
            content_weight,
            collaborative_weight,
            popularity_weight,
        ]

        if any(weight < 0 for weight in weights):
            raise ValueError("Recommendation weights cannot be negative.")

        total_weight = sum(weights)

        if total_weight == 0:
            raise ValueError(
                "At least one recommendation weight must be positive."
            )

        content_weight /= total_weight
        collaborative_weight /= total_weight
        popularity_weight /= total_weight

        candidate_pool = max(candidate_pool, top_n)

        seed_movie_id = self._find_seed_movie_id(seed_title)

        content_results = self.content_model.recommend(
            seed_title,
            top_n=candidate_pool,
        )[
            ["movieId", "similarity_score"]
        ].rename(
            columns={"similarity_score": "content_score"}
        )

        collaborative_results = self.collaborative_model.recommend(
            user_id,
            top_n=candidate_pool,
        )[
            ["movieId", "predicted_rating"]
        ]

        popularity_results = recommend_popular_movies(
            self.movies,
            self.ratings,
            top_n=candidate_pool,
        )[
            ["movieId", "popularity_score"]
        ]

        candidates = content_results.merge(
            collaborative_results,
            on="movieId",
            how="outer",
        )

        candidates = candidates.merge(
            popularity_results,
            on="movieId",
            how="outer",
        )

        watched_movie_ids = set(
            self.ratings.loc[
                self.ratings["userId"] == user_id,
                "movieId",
            ]
        )

        candidates = candidates[
            ~candidates["movieId"].isin(watched_movie_ids)
        ]

        candidates = candidates[
            candidates["movieId"] != seed_movie_id
        ].copy()

        candidates["content_score"] = (
            candidates["content_score"]
            .fillna(0.0)
            .clip(0.0, 1.0)
        )

        candidates["collaborative_score"] = (
            (
                candidates["predicted_rating"].fillna(0.5)
                - 0.5
            )
            / 4.5
        ).clip(0.0, 1.0)

        candidates["normalized_popularity_score"] = (
            self._normalize_popularity(
                candidates["popularity_score"]
            )
        )

        candidates["hybrid_score"] = (
            content_weight * candidates["content_score"]
            + collaborative_weight
            * candidates["collaborative_score"]
            + popularity_weight
            * candidates["normalized_popularity_score"]
        )

        candidates["recommendation_reason"] = candidates.apply(
            self._create_reason,
            axis=1,
            seed_title=seed_title,
        )

        candidates = candidates.merge(
            self.movies[
                [
                    "movieId",
                    "clean_title",
                    "year",
                    "genres",
                ]
            ],
            on="movieId",
            how="left",
            validate="one_to_one",
        )

        columns = [
            "movieId",
            "clean_title",
            "year",
            "genres",
            "content_score",
            "predicted_rating",
            "popularity_score",
            "hybrid_score",
            "recommendation_reason",
        ]

        return (
            candidates.sort_values(
                "hybrid_score",
                ascending=False,
            )[columns]
            .head(top_n)
            .reset_index(drop=True)
        )


def display_recommendations(
    user_id: int,
    seed_title: str,
    recommendations: pd.DataFrame,
) -> None:
    """Display hybrid recommendations."""

    print(
        f"\nHYBRID RECOMMENDATIONS FOR USER {user_id} "
        f"BASED ON '{seed_title}'"
    )
    print("=" * 120)

    print(
        recommendations.to_string(
            index=False,
            formatters={
                "content_score": "{:.3f}".format,
                "predicted_rating": (
                    lambda value: ""
                    if pd.isna(value)
                    else f"{value:.3f}"
                ),
                "popularity_score": (
                    lambda value: ""
                    if pd.isna(value)
                    else f"{value:.3f}"
                ),
                "hybrid_score": "{:.3f}".format,
            },
        )
    )


if __name__ == "__main__":
    movies, ratings, tags, _, _ = prepare_data()

    recommender = HybridRecommender(
        movies,
        ratings,
        tags,
        n_factors=50,
    )

    selected_user = 1
    selected_movie = "Matrix"

    results = recommender.recommend(
        user_id=selected_user,
        seed_title=selected_movie,
        top_n=10,
    )

    display_recommendations(
        selected_user,
        selected_movie,
        results,
    )