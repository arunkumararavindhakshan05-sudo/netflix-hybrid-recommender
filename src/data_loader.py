import pandas as pd

from src.download_data import ensure_dataset


def load_data():
    """Load all MovieLens CSV files."""

    # Verify that the dataset exists.
    # If it is missing, download and extract it automatically.
    data_dir = ensure_dataset()

    movies = pd.read_csv(
        data_dir / "movies.csv"
    )

    ratings = pd.read_csv(
        data_dir / "ratings.csv"
    )

    tags = pd.read_csv(
        data_dir / "tags.csv"
    )

    links = pd.read_csv(
        data_dir / "links.csv"
    )

    return movies, ratings, tags, links


def inspect_data(
    name: str,
    dataframe: pd.DataFrame,
) -> None:
    """Display basic information about a dataset."""

    print(f"\n{'=' * 50}")
    print(f"{name.upper()} DATASET")
    print(f"{'=' * 50}")

    print(f"Rows: {dataframe.shape[0]}")
    print(f"Columns: {dataframe.shape[1]}")
    print(
        f"Column names: {list(dataframe.columns)}"
    )

    print("\nFirst five records:")
    print(dataframe.head())

    print("\nMissing values:")
    print(dataframe.isnull().sum())


if __name__ == "__main__":
    movies, ratings, tags, links = load_data()

    inspect_data(
        "Movies",
        movies,
    )

    inspect_data(
        "Ratings",
        ratings,
    )

    inspect_data(
        "Tags",
        tags,
    )

    inspect_data(
        "Links",
        links,
    )