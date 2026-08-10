from pathlib import Path
from tempfile import TemporaryDirectory

import pytest

from backend.index_snapshot import (
    build_fingerprint,
    create_index_snapshot,
    extract_index_snapshot,
    snapshot_is_compatible,
)


@pytest.mark.parametrize("kind", ["file", "directory"])
def test_index_snapshot_round_trip(kind):
    with TemporaryDirectory() as temp_dir:
        root = Path(temp_dir)
        source = root / ("index.json" if kind == "file" else "index")
        target = root / ("restored.json" if kind == "file" else "restored")
        if kind == "file":
            source.write_text('{"documents":[]}', encoding="utf-8")
        else:
            source.mkdir()
            (source / "chroma.sqlite3").write_bytes(b"index")
            (source / "segments").mkdir()
            (source / "segments" / "data.bin").write_bytes(b"vectors")

        metadata, bundle = create_index_snapshot(source, "light")
        extract_index_snapshot(
            bundle,
            metadata,
            target,
            max_total_bytes=1024,
        )

        assert metadata["kind"] == kind
        assert snapshot_is_compatible(metadata, "light")
        if kind == "file":
            assert target.read_bytes() == source.read_bytes()
        else:
            assert (target / "chroma.sqlite3").read_bytes() == b"index"
            assert (target / "segments" / "data.bin").read_bytes() == b"vectors"


def test_snapshot_rejects_corruption_and_incompatible_fingerprint():
    with TemporaryDirectory() as temp_dir:
        root = Path(temp_dir)
        source = root / "index.json"
        source.write_text("[]", encoding="utf-8")
        metadata, bundle = create_index_snapshot(source, "light")

        with pytest.raises(ValueError, match="完整性"):
            extract_index_snapshot(
                bundle + b"corrupt",
                metadata,
                root / "bad.json",
                max_total_bytes=1024,
            )

        incompatible = {**metadata, "fingerprint": build_fingerprint("full")}
        assert not snapshot_is_compatible(incompatible, "light")
