from zipfile import ZipFile

import pytest

from src.download_data import (
    dataset_is_available,
    safely_extract_archive,
)


def test_dataset_availability_requires_all_files(tmp_path):
    """A dataset is available when every required file exists."""

    dataset_dir = tmp_path / "ml-latest-small"
    dataset_dir.mkdir()

    required_files = [
        "movies.csv",
        "ratings.csv",
        "tags.csv",
        "links.csv",
    ]

    for filename in required_files:
        (dataset_dir / filename).write_text(
            "test",
            encoding="utf-8",
        )

    assert dataset_is_available(dataset_dir)


def test_incomplete_dataset_is_not_available(tmp_path):
    """An incomplete dataset should not be accepted."""

    dataset_dir = tmp_path / "ml-latest-small"
    dataset_dir.mkdir()

    (dataset_dir / "movies.csv").write_text(
        "test",
        encoding="utf-8",
    )

    assert not dataset_is_available(dataset_dir)


def test_safe_archive_is_extracted(tmp_path):
    """A valid ZIP archive should be extracted successfully."""

    archive_path = tmp_path / "safe.zip"
    destination = tmp_path / "output"

    with ZipFile(archive_path, "w") as archive:
        archive.writestr(
            "ml-latest-small/movies.csv",
            "movieId,title,genres",
        )

    safely_extract_archive(
        archive_path,
        destination,
    )

    extracted_file = (
        destination
        / "ml-latest-small"
        / "movies.csv"
    )

    assert extracted_file.is_file()


def test_unsafe_archive_path_is_rejected(tmp_path):
    """A ZIP file attempting path traversal should be rejected."""

    archive_path = tmp_path / "unsafe.zip"
    destination = tmp_path / "output"

    with ZipFile(archive_path, "w") as archive:
        archive.writestr(
            "../unsafe.txt",
            "unsafe content",
        )

    with pytest.raises(
        ValueError,
        match="Unsafe ZIP member",
    ):
        safely_extract_archive(
            archive_path,
            destination,
        )