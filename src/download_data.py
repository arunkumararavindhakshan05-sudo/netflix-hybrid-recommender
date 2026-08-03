from pathlib import Path
from shutil import copyfileobj
from urllib.request import Request, urlopen
from zipfile import ZipFile

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = PROJECT_ROOT / "data"
DATA_DIR = DATA_ROOT / "ml-latest-small"
ARCHIVE_PATH = DATA_ROOT / "ml-latest-small.zip"

DATASET_URL = (
    "https://files.grouplens.org/datasets/"
    "movielens/ml-latest-small.zip"
)

REQUIRED_FILES = {
    "movies.csv",
    "ratings.csv",
    "tags.csv",
    "links.csv",
}


def dataset_is_available(
    dataset_dir: Path = DATA_DIR,
) -> bool:
    """Check whether all required MovieLens files exist."""

    return all(
        (dataset_dir / filename).is_file()
        for filename in REQUIRED_FILES
    )


def download_archive(
    url: str = DATASET_URL,
    destination: Path = ARCHIVE_PATH,
) -> None:
    """Download the MovieLens ZIP archive."""

    destination.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    request = Request(
        url,
        headers={
            "User-Agent": "CineMatch-Recommender/1.0",
        },
    )

    with (
        urlopen(request, timeout=60) as response,
        destination.open("wb") as output_file,
    ):
        copyfileobj(response, output_file)


def safely_extract_archive(
    archive_path: Path,
    destination: Path,
) -> None:
    """Extract a ZIP file while preventing unsafe paths."""

    destination.mkdir(
        parents=True,
        exist_ok=True,
    )

    destination_root = destination.resolve()

    with ZipFile(archive_path) as archive:
        for member in archive.infolist():
            target_path = (
                destination / member.filename
            ).resolve()

            if not target_path.is_relative_to(destination_root):
                raise ValueError(
                    f"Unsafe ZIP member: {member.filename}"
                )

        archive.extractall(destination)


def ensure_dataset() -> Path:
    """Download and extract MovieLens when it is unavailable."""

    if dataset_is_available():
        return DATA_DIR

    print("MovieLens dataset not found. Downloading...")

    download_archive()
    safely_extract_archive(
        ARCHIVE_PATH,
        DATA_ROOT,
    )

    ARCHIVE_PATH.unlink(missing_ok=True)

    if not dataset_is_available():
        raise RuntimeError(
            "MovieLens download completed, but required files "
            "are missing."
        )

    print("MovieLens dataset downloaded successfully.")

    return DATA_DIR


if __name__ == "__main__":
    dataset_path = ensure_dataset()
    print(f"Dataset available at: {dataset_path}")