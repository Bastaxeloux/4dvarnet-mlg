"""Regression tests for mean/uncertainty ASCII file selection."""

from pathlib import Path
import sys
import tempfile

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from data.ascii_files import (
    resolve_satellite_ascii,
    satellite_ascii_candidates,
)


DAY = "2024122112"


def touch(directory, name):
    path = Path(directory) / name
    path.touch()
    return path


def main():
    with tempfile.TemporaryDirectory() as tmp:
        mean = touch(tmp, f"{DAY}_pmw_cci_l2p_av.asc")
        uncertainty = touch(tmp, f"{DAY}_pmw_cci_l2p_std_av.asc")
        touch(tmp, f"{DAY}_pmw_cci_l2p_min_av.asc")
        touch(tmp, f"{DAY}_pmw_cci_l2p_max_av.asc")

        assert resolve_satellite_ascii(tmp, DAY, "pmw", "av") == mean
        assert resolve_satellite_ascii(tmp, DAY, "pmw", "std") == uncertainty
        assert satellite_ascii_candidates(tmp, DAY, "pmw", "av") == [mean]

        mean.unlink()
        try:
            resolve_satellite_ascii(tmp, DAY, "pmw", "av")
        except FileNotFoundError:
            pass
        else:
            raise AssertionError("The uncertainty file was accepted as a mean")

        touch(tmp, f"{DAY}_pmw_c3s_l2p_av.asc")
        touch(tmp, f"{DAY}_pmw_cci_l2p_av.asc")
        try:
            resolve_satellite_ascii(tmp, DAY, "pmw", "av")
        except RuntimeError as exc:
            assert "Ambiguous" in str(exc)
        else:
            raise AssertionError("Ambiguous mean files were accepted")

    print("ASCII file selection tests passed")


if __name__ == "__main__":
    main()
