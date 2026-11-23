import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from task_skills.gateway.contracts import Catalog
from task_skills.models import load_scene_config
from task_skills.run_guidance_preview import main, preview

ROOT = Path(__file__).resolve().parents[1]
EXAMPLES = ROOT / "examples" / "guidance"


class GuidancePreviewTests(unittest.TestCase):
    """The preview must report what the gateway would do, not a parallel opinion."""

    def setUp(self):
        self.catalog = Catalog(load_scene_config(ROOT / "task_skills" / "scene_config.json"))

    def test_shipped_accepted_examples_expand_to_the_gateway_plan(self):
        single = preview(EXAMPLES / "01_single_pour.json", self.catalog)
        chained = preview(EXAMPLES / "02_pour_wait_pour.json", self.catalog)
        self.assertEqual(len(single["rows"]), 20)
        self.assertEqual(len(chained["rows"]), 41)
        # The second pour may not start before the settle wait, which follows both homes.
        depth = chained["depth"]
        self.assertGreater(depth["pour_2.left_tube_approach"], depth["settle"])
        self.assertGreater(depth["settle"], depth["pour_1.right_arm_home"])
        self.assertIn("tilt=45°", next(r["detail"] for r in chained["rows"] if r["id"] == "pour_2.pour_into_tube"))

    def test_shipped_rejected_examples_carry_the_gateway_error(self):
        for name, fragment in (("03_rejected_unordered_pours.json", "unordered"),
                               ("04_rejected_unknown_operation.json", "embodied.pipette"),
                               ("05_rejected_untaught_objects.json", "no taught pour")):
            with self.subTest(name=name):
                result = preview(EXAMPLES / name, self.catalog)
                self.assertEqual(result["error"]["code"], "INVALID_GRAPH")
                self.assertIn(fragment, result["error"]["message"])

    def test_cli_writes_one_page_for_a_folder(self):
        with TemporaryDirectory() as directory:
            out = Path(directory) / "preview.html"
            self.assertEqual(main([str(EXAMPLES), "--out", str(out)]), 0)
            page = out.read_text(encoding="utf-8")
            for name in ("01_single_pour.json", "05_rejected_untaught_objects.json"):
                self.assertIn(name, page)
            self.assertNotIn("http-equiv", page)


if __name__ == "__main__":
    unittest.main()
