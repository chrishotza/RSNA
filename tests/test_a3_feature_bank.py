import unittest

from src.a3_feature_bank import (
    SliceRecord,
    acquisition_type_id,
    adjacent_triplet,
    build_series_window_manifest,
    deterministic_anchor_centers,
    physical_slice_coordinate,
    sort_slice_records,
)


class TestA3FeatureBank(unittest.TestCase):
    def test_physical_coordinate_uses_slice_normal(self):
        ori = (1, 0, 0, 0, 1, 0)
        self.assertEqual(physical_slice_coordinate((0, 0, 7), ori), 7.0)

    def test_geometry_sort_beats_instance_number(self):
        ori = (1, 0, 0, 0, 1, 0)
        records = [
            SliceRecord("c", (0, 0, 3), ori, 1),
            SliceRecord("a", (0, 0, 1), ori, 3),
            SliceRecord("b", (0, 0, 2), ori, 2),
        ]
        ordered = sort_slice_records(records)
        self.assertEqual([r.sop_instance_uid for r in ordered], ["a", "b", "c"])

    def test_fallback_is_series_wide(self):
        records = [
            SliceRecord("b", None, None, 2),
            SliceRecord("a", (0, 0, 99), (1, 0, 0, 0, 1, 0), 1),
        ]
        ordered = sort_slice_records(records)
        self.assertEqual([r.sop_instance_uid for r in ordered], ["a", "b"])

    def test_triplet_replication_at_edges(self):
        self.assertEqual(adjacent_triplet(0, 5), (0, 0, 1))
        self.assertEqual(adjacent_triplet(4, 5), (3, 4, 4))

    def test_anchors_are_deterministic_unique_and_bounded(self):
        a = deterministic_anchor_centers(7, n_anchors=12)
        b = deterministic_anchor_centers(7, n_anchors=12)
        self.assertEqual(a, b)
        self.assertEqual(len(a), len(set(a)))
        self.assertTrue(all(0 <= x < 7 for x in a))

    def test_acquisition_ids_are_stable(self):
        self.assertEqual(acquisition_type_id("sagittal", 0, 0), 4)
        self.assertEqual(acquisition_type_id("cor", 1, 0), 10)
        self.assertEqual(acquisition_type_id("axial", 1, 1), 15)

    def test_manifest_is_adjacent_25d(self):
        sops = [f"s{i}" for i in range(10)]
        m = build_series_window_manifest(sops, n_anchors=4)
        self.assertEqual(len(m), 4)
        for row in m:
            self.assertEqual(len(row["sop_triplet"]), 3)


if __name__ == "__main__":
    unittest.main()
