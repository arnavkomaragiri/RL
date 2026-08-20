from __future__ import annotations

import hashlib
import json
import tempfile
import unittest
import zipfile
from pathlib import Path

import prepare_clean_split


TASK_ID = "015ab7f4-b069-4912-ac0b-a3f8bd9c869d"


class PrepareCleanSplitTest(unittest.TestCase):
    def test_quarantine_preserves_deleted_file(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            split = root / "split"
            data_dir = prepare_clean_split.task_data_dir(split, TASK_ID)
            data_dir.mkdir(parents=True)
            leaked = data_dir / "DE.csv"
            leaked.write_text("answer\n")
            archive = root / "privileged"

            prepare_clean_split.quarantine_patch_targets(split, archive)
            prepare_clean_split.apply_patches(split)

            archived = archive / f"capsule_{TASK_ID}" / "DE.csv"
            self.assertFalse(leaked.exists())
            self.assertEqual(archived.read_text(), "answer\n")
            manifest = json.loads((archive / "manifest.json").read_text())
            entry = manifest["files"][0]
            self.assertEqual(entry["capsule_uuid"], TASK_ID)
            self.assertEqual(entry["sha256"], hashlib.sha256(b"answer\n").hexdigest())

    def test_replaces_expert_only_capsule_with_raw_input(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            split = root / "split"
            target_id = "a8350a95-b0d8-4e40-aeed-2bd297043086"
            source_id = "4ee38ef1-ea48-4c6a-856a-ff7c7166413d"
            target_dir = prepare_clean_split.task_data_dir(split, target_id)
            source_dir = prepare_clean_split.task_data_dir(split, source_id)
            target_dir.mkdir(parents=True)
            source_dir.mkdir(parents=True)
            expert = target_dir / "mtb_hyp2_drug_sensitivity.py"
            expert.write_text("print('answer')\n")
            raw = source_dir / "GSE222412_rawCountMatrix.csv.gz"
            raw.write_bytes(b"raw-counts")
            archive = root / "privileged"

            prepare_clean_split.quarantine_patch_targets(split, archive)
            prepare_clean_split.apply_patches(split)

            self.assertFalse(expert.exists())
            self.assertEqual(
                (target_dir / "GSE222412_rawCountMatrix.csv.gz").read_bytes(),
                b"raw-counts",
            )
            self.assertTrue(
                (
                    archive / f"capsule_{target_id}" / "mtb_hyp2_drug_sensitivity.py"
                ).is_file()
            )

    def test_soapberry_zip_keeps_data_only(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            archive = Path(temporary) / "data_and_code.zip"
            with zipfile.ZipFile(archive, "w") as handle:
                handle.writestr("data_and_code/data/survey.csv", "raw\n")
                handle.writestr("data_and_code/code/analysis.R", "answer\n")
                handle.writestr("data_and_code/Figures/result.pdf", "answer\n")
                handle.writestr("__MACOSX/._survey.csv", "metadata\n")

            prepare_clean_split.patch_soapberry_zip(archive)

            with zipfile.ZipFile(archive) as handle:
                self.assertEqual(handle.namelist(), ["data_and_code/data/survey.csv"])
                self.assertEqual(handle.read("data_and_code/data/survey.csv"), b"raw\n")

    def test_validation_expert_analysis_is_quarantined(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            split = root / "split"
            task_id = "33b801bb-9b47-4a0a-9314-05325c82fde7"
            data_dir = prepare_clean_split.task_data_dir(split, task_id)
            data_dir.mkdir(parents=True)
            notebook = data_dir / "FHAC_Issy-Paper1_2E_v2.ipynb"
            notebook.write_text('{"expert": "analysis"}\n')
            raw_counts = data_dir / "Issy_ASXL1_blood_featureCounts_GeneTable_final.txt"
            raw_counts.write_text("gene\tsample\n")
            archive = root / "privileged"

            prepare_clean_split.quarantine_patch_targets(split, archive)
            prepare_clean_split.apply_patches(split)

            self.assertFalse(notebook.exists())
            self.assertTrue(raw_counts.is_file())
            self.assertEqual(
                (
                    archive / f"capsule_{task_id}" / "FHAC_Issy-Paper1_2E_v2.ipynb"
                ).read_text(),
                '{"expert": "analysis"}\n',
            )

    def test_validation_comparison_task_receives_fibroblast_inputs(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            split = root / "split"
            source_id = "2a8a40d4-05b0-4eec-8bd2-825f61fc9f5d"
            target_id = "4ef3fcd8-1c35-466f-9d93-49b92f4ea760"
            source_dir = prepare_clean_split.task_data_dir(split, source_id)
            target_dir = prepare_clean_split.task_data_dir(split, target_id)
            source_dir.mkdir(parents=True)
            target_dir.mkdir(parents=True)
            (
                source_dir / "Issy_ASXL1_fibro_featureCounts_GeneTable_final.txt"
            ).write_bytes(b"raw-fibroblast-counts")
            (source_dir / "Issy_ASXL1_fibro_coldata_gender.xlsx").write_bytes(
                b"fibroblast-metadata"
            )

            prepare_clean_split.apply_patches(split)

            self.assertEqual(
                (
                    target_dir / "Issy_ASXL1_fibro_featureCounts_GeneTable_final.txt"
                ).read_bytes(),
                b"raw-fibroblast-counts",
            )
            self.assertEqual(
                (target_dir / "Issy_ASXL1_fibro_coldata_gender.xlsx").read_bytes(),
                b"fibroblast-metadata",
            )


if __name__ == "__main__":
    unittest.main()
