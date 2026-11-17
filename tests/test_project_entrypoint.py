"""The source entry point must not start inference or training implicitly."""
import contextlib
import io
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import project


class ProjectEntryPoint(unittest.TestCase):
    def test_default_lists_commands_without_importing_an_execution_module(self):
        with patch.object(project.importlib, "import_module") as imported:
            with contextlib.redirect_stdout(io.StringIO()) as output:
                project.main([])
        imported.assert_not_called()
        self.assertIn("src/README.md", output.getvalue())

    def test_selected_prompt_matches_the_preserved_experiment(self):
        selected, text = project.load_working_prompt()
        original = project.ROOT / "reports/part2/parts_prompt_replay_20261029/USER_PROMPT.txt"
        self.assertEqual(text, original.read_text())
        self.assertEqual(selected["observed_development_results"]["derived_correct"], 7)

    def test_modified_prompt_is_rejected(self):
        selected = json.loads(project.SELECTION.read_text())
        with tempfile.TemporaryDirectory() as temp:
            folder = Path(temp)
            changed = folder / "changed_prompt.txt"
            changed.write_text("A different instruction")
            selected["prompt_file"] = str(changed)
            selection = folder / "selection.json"
            selection.write_text(json.dumps(selected))
            with self.assertRaisesRegex(ValueError, "differs from the frozen selection"):
                project.load_working_prompt(selection)

    def test_viewer_receives_only_the_requested_port_and_restores_argv(self):
        received = []
        viewer = SimpleNamespace(__file__="viewer.py", main=lambda: received.append(sys.argv[:]))
        previous = sys.argv
        with patch.object(project.importlib, "import_module", return_value=viewer):
            project.main(["serve", "--port", "8766"])
        self.assertEqual(received, [["viewer.py", "--port", "8766"]])
        self.assertIs(sys.argv, previous)

    def test_report_alias_calls_the_saved_report_builder(self):
        module = SimpleNamespace(main=Mock())
        with patch.object(project.importlib, "import_module", return_value=module) as imported:
            project.main(["report", "parts"])
        imported.assert_called_once_with("reporting.build_claim_parts_report")
        module.main.assert_called_once_with()


if __name__ == "__main__":
    unittest.main()
