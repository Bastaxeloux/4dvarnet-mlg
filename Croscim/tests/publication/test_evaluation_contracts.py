#!/usr/bin/env python3
from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from contrib.SST.evaluation.assembly import PatchAccumulator
from contrib.SST.evaluation.coast import build_coastal_mask
from contrib.SST.evaluation.metrics import (
    WeightedSufficientStats,
    circular_block_bootstrap_indices,
    weighted_sufficient_stats,
)
from contrib.SST.evaluation.masking import downsample_mask
from contrib.SST.evaluation.protocol import build_publication_manifests, load_manifest


class FakeDataset:
    da_dims = {"lat": 4, "lon": 5}

    def _slices_from_flat_index(self, index):
        slices = (
            (slice(0, 3), slice(0, 3)),
            (slice(0, 3), slice(2, 5)),
            (slice(1, 4), slice(0, 3)),
            (slice(1, 4), slice(2, 5)),
        )
        lat_slice, lon_slice = slices[index]
        return {"time": slice(0, 3), "lat": lat_slice, "lon": lon_slice}


class ProtocolTests(unittest.TestCase):
    def test_publication_dates_are_deterministic(self):
        with tempfile.TemporaryDirectory() as directory:
            paths = build_publication_manifests(directory)
            pilot = load_manifest(paths["pilot"])
            final = load_manifest(paths["test"])
            protocol = load_manifest(paths["protocol"])
            self.assertEqual(len(pilot["records"]), 24)
            self.assertEqual(pilot["records"][0]["index"], 7)
            self.assertEqual(pilot["records"][-1]["index"], 357)
            self.assertEqual([record["index"] for record in final["records"]], list(range(7, 359)))
            self.assertEqual(len(final["records"]), 352)
            self.assertEqual(protocol["coarse_mask_rule"], "any_x1_withheld_pixel")
            for record in pilot["records"] + final["records"]:
                self.assertAlmostEqual(record["longitude_shift_degrees"] % 1.5, 0.0)
                self.assertIn(record["donor_year"], range(2017, 2023))

            second_directory = Path(directory) / "second"
            second_paths = build_publication_manifests(second_directory)
            self.assertEqual(Path(paths["test"]).read_bytes(), Path(second_paths["test"]).read_bytes())

class AssemblyTests(unittest.TestCase):
    def test_patch_assembly_covers_edges_and_tracks_disagreement(self):
        accumulator = PatchAccumulator(FakeDataset(), 3)
        predictions = np.stack([
            np.full((3, 3, 3), value, dtype=np.float32)
            for value in (1.0, 2.0, 3.0, 4.0)
        ])
        accumulator.add(predictions, [0, 1, 2, 3])
        result = accumulator.finalize()
        self.assertFalse(np.any(result.geometric_coverage == 0))
        self.assertEqual(result.mean.shape, (3, 4, 5))
        self.assertAlmostEqual(float(result.mean[1, 1, 2]), 2.5)
        self.assertGreater(float(result.overlap_std_central[1, 2]), 0.0)

    def test_central_only_assembly_keeps_the_middle_day(self):
        accumulator = PatchAccumulator(FakeDataset(), 5, central_only=True)
        prediction = np.stack([
            np.full((3, 3), value, dtype=np.float32)
            for value in range(5)
        ])
        accumulator.add(np.stack([prediction] * 4), [0, 1, 2, 3])
        result = accumulator.finalize()
        self.assertEqual(result.mean.shape, (1, 4, 5))
        np.testing.assert_array_equal(result.mean[0, :3, :3], 2.0)


class MaskTests(unittest.TestCase):
    def test_coarse_mask_prevents_fine_scale_withholding_leakage(self):
        mask = np.zeros((1, 6, 6), dtype=bool)
        mask[0, 1, 1] = True
        coarse = downsample_mask(mask, 3)
        np.testing.assert_array_equal(coarse, [[[True, False], [False, False]]])

    def test_coastline_detection_wraps_at_antimeridian(self):
        latitude = np.array([0.0])
        longitude = np.array([-179.0, -90.0, 0.0, 179.0])
        surfmask = np.array([[0, 0, 0, 1]], dtype=np.uint8)
        coastal = build_coastal_mask(latitude, longitude, surfmask, threshold_km=300.0)
        self.assertTrue(coastal[0, 3])


class MetricTests(unittest.TestCase):
    def test_empty_support_keeps_the_metric_schema(self):
        metrics = WeightedSufficientStats().metrics()
        self.assertEqual(metrics["n_pixels"], 0)
        self.assertTrue(np.isnan(metrics["rmse_c"]))
        self.assertEqual(
            set(metrics),
            {"n_pixels", "sum_w", "rmse_c", "mae_c", "bias_c", "correlation", "target_std_c", "nrmse"},
        )

    def test_weighted_metrics_and_nrmse_use_same_support(self):
        target = np.array([[1.0, 2.0]])
        prediction = np.array([[1.0, 3.0]])
        stats = weighted_sufficient_stats(target, prediction, np.ones_like(target, dtype=bool), np.array([0.0]))
        metrics = stats.metrics()
        self.assertAlmostEqual(metrics["rmse_c"], np.sqrt(0.5))
        self.assertAlmostEqual(metrics["mae_c"], 0.5)
        self.assertAlmostEqual(metrics["bias_c"], 0.5)
        self.assertAlmostEqual(metrics["target_std_c"], 0.5)
        self.assertAlmostEqual(metrics["nrmse"], np.sqrt(2.0))

        merged = WeightedSufficientStats.from_dict(stats.as_dict()).merge(
            WeightedSufficientStats.from_dict(stats.as_dict())
        )
        self.assertAlmostEqual(merged.metrics()["rmse_c"], metrics["rmse_c"])

    def test_circular_bootstrap_is_deterministic_and_in_bounds(self):
        first = circular_block_bootstrap_indices(352)
        second = circular_block_bootstrap_indices(352)
        np.testing.assert_array_equal(first, second)
        self.assertEqual(first.shape, (2000, 352))
        self.assertGreaterEqual(int(first.min()), 0)
        self.assertLess(int(first.max()), 352)

if __name__ == "__main__":
    unittest.main()
