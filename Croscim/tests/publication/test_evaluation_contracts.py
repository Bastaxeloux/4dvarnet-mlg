#!/usr/bin/env python3
from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from contrib.SST.evaluation.assembly import PatchAccumulator
from contrib.SST.evaluation.checkpoint import select_best_checkpoint
from contrib.SST.evaluation.coast import build_coastal_mask
from contrib.SST.evaluation.metrics import (
    WeightedSufficientStats,
    circular_block_bootstrap_indices,
    weighted_sufficient_stats,
)
from contrib.SST.evaluation.masking import downsample_mask
from contrib.SST.evaluation.protocol import build_publication_manifests, load_manifest
from contrib.SST.evaluation.io import sha256_file
from scripts.publication.select_checkpoint import select_candidate_records


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
    def test_publication_dates_and_hashes_are_frozen(self):
        with tempfile.TemporaryDirectory() as directory:
            paths = build_publication_manifests(directory)
            pilot = load_manifest(paths["pilot"])
            final = load_manifest(paths["test"])
            donors = load_manifest(paths["donors"])
            protocol = load_manifest(paths["protocol"])
            self.assertEqual(len(pilot["records"]), 24)
            self.assertEqual(pilot["records"][0]["index"], 7)
            self.assertEqual(pilot["records"][-1]["index"], 357)
            self.assertEqual([record["index"] for record in final["records"]], list(range(7, 359)))
            self.assertEqual(len(final["records"]), 352)
            self.assertEqual(len(donors["records"]), 2107)
            self.assertEqual(protocol["coarse_mask_rule"], "any_x1_withheld_pixel")
            for record in pilot["records"] + final["records"]:
                self.assertAlmostEqual(record["longitude_shift_degrees"] % 1.5, 0.0)
                self.assertIn(record["donor_year"], range(2017, 2023))

            second_directory = Path(directory) / "second"
            second_paths = build_publication_manifests(second_directory)
            self.assertEqual(Path(paths["test"]).read_bytes(), Path(second_paths["test"]).read_bytes())


class CheckpointTests(unittest.TestCase):
    def test_selects_native_best_cycle_checkpoint(self):
        with tempfile.TemporaryDirectory() as directory:
            checkpoint_dir = Path(directory)
            last = checkpoint_dir / "last.ckpt"
            first = checkpoint_dir / "cycle_end_epoch=023.ckpt"
            best = checkpoint_dir / "cycle_end_epoch=047.ckpt"
            for path in (last, first, best):
                path.write_bytes(path.name.encode("ascii"))
            callback_key = "ModelCheckpoint{'monitor': 'val/x1/loss', 'mode': 'min'}"
            last_payload = {
                "state_dict": {},
                "callbacks": {
                    callback_key: {
                        "best_k_models": {str(first): 0.3, str(best): 0.2},
                    }
                },
            }

            def fake_load(path):
                if Path(path).name == last.name:
                    return last_payload
                if Path(path).name == best.name:
                    return {"state_dict": {}, "epoch": 47, "global_step": 1200}
                return {"state_dict": {}, "epoch": 23, "global_step": 600}

            with patch("contrib.SST.evaluation.checkpoint._load_checkpoint", side_effect=fake_load):
                selected = select_best_checkpoint(checkpoint_dir)
            self.assertEqual(Path(selected.path), best.resolve())
            self.assertEqual(selected.epoch, 47)
            self.assertEqual(selected.score, 0.2)

    def test_controlled_pilot_selects_lowest_hidden_x1_rmse(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest_hash = "a" * 64
            pairs = []
            for epoch, score in ((23, 0.8), (47, 0.6)):
                checkpoint = root / f"cycle_end_epoch={epoch:03d}.ckpt"
                checkpoint.write_bytes(f"checkpoint-{epoch}".encode("ascii"))
                evaluation = root / f"pilot_{epoch}"
                results = evaluation / "results"
                results.mkdir(parents=True)
                validation = {
                    "accepted": True,
                    "mode": "controlled",
                    "n_dates": 24,
                    "frozen_protocol_sha256": None,
                    "checkpoint_sha256": sha256_file(checkpoint),
                    "manifest_sha256": manifest_hash,
                }
                (evaluation / "pilot_validation.json").write_text(json.dumps(validation))
                metrics = results / "metrics_summary.csv"
                metrics.write_text(
                    "period_type,period,method,support,regime,rmse_c\n"
                    f"annual,2023,croscim_x1,hidden,global,{score}\n"
                )
                aggregation = {
                    "mode": "controlled",
                    "n_dates": 24,
                    "checkpoint_sha256": sha256_file(checkpoint),
                    "manifest": {"sha256": manifest_hash},
                    "artifacts": {metrics.name: sha256_file(metrics)},
                }
                (results / "aggregation_complete.json").write_text(json.dumps(aggregation))
                pairs.append((str(checkpoint), str(evaluation)))

            with patch(
                "scripts.publication.select_checkpoint.load_checkpoint_metadata",
                side_effect=[(23, 100), (47, 200)],
            ):
                candidates, selected = select_candidate_records(pairs)
            self.assertEqual(len(candidates), 2)
            self.assertEqual(selected["epoch"], 47)
            self.assertEqual(selected["hidden_global_x1_rmse_c"], 0.6)


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
