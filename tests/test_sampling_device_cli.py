import ast
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def _source(path):
    return (ROOT / path).read_text()


def _parser_options(path):
    tree = ast.parse(_source(path))
    options = set()
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        if not isinstance(node.func, ast.Attribute):
            continue
        if node.func.attr != "add_argument":
            continue
        for arg in node.args:
            if isinstance(arg, ast.Constant) and isinstance(arg.value, str):
                options.add(arg.value)
    return options


class SamplingDeviceCliTest(unittest.TestCase):
    def test_sampling_scripts_expose_device_option(self):
        self.assertIn("--device", _parser_options("scripts/sample_tlddpm.py"))
        self.assertIn("--device", _parser_options("scripts/sample_bridge.py"))

    def test_sampling_scripts_move_loaded_model_to_resolved_device(self):
        for path in ("scripts/sample_tlddpm.py", "scripts/sample_bridge.py"):
            with self.subTest(path=path):
                source = _source(path)
                self.assertIn("resolve_sample_device(args.device)", source)
                self.assertIn("model.to(device)", source)


if __name__ == "__main__":
    unittest.main()
