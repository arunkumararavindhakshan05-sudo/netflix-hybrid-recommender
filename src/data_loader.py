from pathlib import Path

import pandas as pd

# Find the dataset folder relative to this project.
DATA_DIR = (
    Path(__file__).resolve().parents[1]
    / "data"
    / "ml-latest-small"
)


def load_data():
    """Load all MovieLens CSV files."""

    movies = pd.read_csv(DATA_DIR / "movies.csv")
    ratings = pd.read_csv(DATA_DIR / "ratings.csv")
    tags = pd.read_csv(DATA_DIR / "tags.csv")
    links = pd.read_csv(DATA_DIR / "links.csv")

    return movies, ratings, tags, links


def inspect_data(name, dataframe):
    """Display basic information about a dataset."""

    print(f"\n{'=' * 50}")
    print(f"{name.upper()} DATASET")
    print(f"{'=' * 50}")

    print(f"Rows: {dataframe.shape[0]}")
    print(f"Columns: {dataframe.shape[1]}")
    print(f"Column names: {list(dataframe.columns)}")

    print("\nFirst five records:")
    print(dataframe.head())

    print("\nMissing values:")
    print(dataframe.isnull().sum())


if __name__ == "__main__":
    movies, ratings, tags, links = load_data()

    inspect_data("Movies", movies)
    inspect_data("Ratings", ratings)
    inspect_data("Tags", tags)
    inspect_data("Links", links)