from __future__ import annotations

import sys
import unittest
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


class RepositoryTest(unittest.TestCase):
    def test_configs_reference_existing_assets(self):
        for name in ("binary", "multiclass"):
            config = yaml.safe_load((ROOT / "configs" / f"{name}.yaml").read_text(encoding="utf-8"))
            self.assertTrue((ROOT / config["checkpoint"]).is_file())
            self.assertTrue(config["data"]["test_dir"].startswith("data/test/"))

    def test_test_dataset_counts(self):
        if not (ROOT / "data" / "test").is_dir():
            self.skipTest("Private test dataset is intentionally absent")
        expected = {
            "binary/Hazard": 17,
            "binary/No_Hazard": 8,
            "multiclass/Confused_Wiring": 9,
            "multiclass/Fire_Lane_Blocked": 10,
        }
        for relative, count in expected.items():
            files = [path for path in (ROOT / "data" / "test" / relative).iterdir() if path.is_file()]
            self.assertEqual(len(files), count, relative)


if __name__ == "__main__":
    unittest.main()
