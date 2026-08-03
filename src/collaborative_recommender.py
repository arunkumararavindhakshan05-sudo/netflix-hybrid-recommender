import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from sklearn.decomposition import TruncatedSVD

from src.data_preparation import prepare_data


class CollaborativeRecommender:
    """Recommend movies using user-rating behaviour and SVD."""

    def __init__(
        self,
        movies: pd.DataFrame,
        ratings: pd.DataFrame,
        n_factors: int = 50,
    ) -> None:
        if ratings.empty:
            raise ValueError("Ratings data cannot be empty.")

        if n_factors < 1:
            raise ValueError("n_factors must be at least 1.")

        self.movies = movies.copy()
        self.ratings = ratings.copy()

        self._create_user_movie_matrix()
        self._train_model(n_factors)

    def _create_user_movie_matrix(self) -> None:
        """Create and mean-centre the user-movie rating matrix."""

        self.user_movie_matrix = self.ratings.pivot_table(
            index="userId",
            columns="movieId",
            values="rating",
            aggfunc="mean",
            fill_value=0.0,
        )

        self.user_ids = self.user_movie_matrix.index.to_numpy()
        self.movie_ids = self.user_movie_matrix.columns.to_numpy()

        self.user_to_index = {
            user_id: index
            for index, user_id in enumerate(self.user_ids)
        }

        rating_matrix = self.user_movie_matrix.to_numpy(dtype=float)

        self.rated_mask = rating_matrix > 0

        rating_sums = rating_matrix.sum(axis=1)
        rating_counts = self.rated_mask.sum(axis=1)

        self.user_means = rating_sums / rating_counts

        # Subtract each user's average only from movies they rated.
        self.centered_matrix = np.where(
            self.rated_mask,
            rating_matrix - self.user_means[:, np.newaxis],
            0.0,
        )

    def _train_model(self, n_factors: int) -> None:
        """Train an SVD model on the centred ratings matrix."""

        maximum_factors = min(self.centered_matrix.shape) - 1

        if maximum_factors < 1:
            raise ValueError(
                "At least two users and two rated movies are required."
            )

        self.n_factors = min(n_factors, maximum_factors)

        self.model = TruncatedSVD(
            n_components=self.n_factors,
            random_state=42,
        )

        sparse_matrix = csr_matrix(self.centered_matrix)

        self.user_factors = self.model.fit_transform(sparse_matrix)
        self.item_factors = self.model.components_.T

    def _predict_user_ratings(self, user_id: int) -> np.ndarray:
        """Predict the selected user's ratings for every movie."""

        if user_id not in self.user_to_index:
            raise ValueError(
                f"Unknown user ID: {user_id}. "
                "Choose an existing MovieLens user."
            )

        user_index = self.user_to_index[user_id]

        centred_predictions = (
            self.user_factors[user_index] @ self.model.components_
        )

        predictions = centred_predictions + self.user_means[user_index]

        return np.clip(predictions, 0.5, 5.0)

    def recommend(
        self,
        user_id: int,
        top_n: int = 10,
    ) -> pd.DataFrame:
        """Recommend unseen movies for a user."""

        if top_n < 1:
            raise ValueError("top_n must be at least 1.")

        predictions = self._predict_user_ratings(user_id)
        user_index = self.user_to_index[user_id]

        prediction_data = pd.DataFrame(
            {
                "movieId": self.movie_ids,
                "predicted_rating": predictions,
                "already_rated": self.rated_mask[user_index],
            }
        )

        unseen_movies = prediction_data[
            ~prediction_data["already_rated"]
        ].copy()

        recommendations = unseen_movies.nlargest(
            top_n,
            "predicted_rating",
        )

        recommendations = recommendations.merge(
            self.movies,
            on="movieId",
            how="left",
            validate="one_to_one",
        )

        columns = [
            "movieId",
            "clean_title",
            "year",
            "genres",
            "predicted_rating",
        ]

        return recommendations[columns].reset_index(drop=True)

    def get_user_history(
        self,
        user_id: int,
        top_n: int = 10,
    ) -> pd.DataFrame:
        """Return the user's highest-rated movies."""

        if user_id not in self.user_to_index:
            raise ValueError(f"Unknown user ID: {user_id}.")

        user_ratings = self.ratings[
            self.ratings["userId"] == user_id
        ]

        history = user_ratings.merge(
            self.movies,
            on="movieId",
            how="left",
            validate="many_to_one",
        )

        columns = [
            "movieId",
            "clean_title",
            "year",
            "genres",
            "rating",
        ]

        return (
            history.sort_values(
                ["rating", "movieId"],
                ascending=[False, True],
            )[columns]
            .head(top_n)
            .reset_index(drop=True)
        )


def display_results(
    heading: str,
    results: pd.DataFrame,
) -> None:
    """Display movie results in the terminal."""

    print(f"\n{heading}")
    print("=" * 90)

    formatters = {}

    if "predicted_rating" in results.columns:
        formatters["predicted_rating"] = "{:.3f}".format

    print(
        results.to_string(
            index=False,
            formatters=formatters,
        )
    )


if __name__ == "__main__":
    movies, ratings, _, _, _ = prepare_data()

    recommender = CollaborativeRecommender(
        movies,
        ratings,
        n_factors=50,
    )

    selected_user = 1

    user_history = recommender.get_user_history(
        selected_user,
        top_n=10,
    )

    recommendations = recommender.recommend(
        selected_user,
        top_n=10,
    )

    display_results(
        f"USER {selected_user} - HIGHEST-RATED MOVIES",
        user_history,
    )

    display_results(
        f"USER {selected_user} - COLLABORATIVE RECOMMENDATIONS",
        recommendations,
    )