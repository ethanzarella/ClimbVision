"""Regression test: Streamlit magic must not echo DeltaGenerator from a bare expression."""
import ast
from pathlib import Path
import unittest


class TestYoloCaption(unittest.TestCase):
    def test_caption_is_in_if_statement(self):
        tree = ast.parse((Path(__file__).parent / "app.py").read_text())
        found = False
        for node in ast.walk(tree):
            if not isinstance(node, ast.If):
                continue
            if not isinstance(node.test, ast.Compare):
                continue
            if any(isinstance(n, ast.Name) and n.id == "YOLO_NAMES" for n in ast.walk(node.test)):
                for sub in node.body:
                    if isinstance(sub, ast.Expr) and isinstance(sub.value, ast.Call):
                        call = sub.value.func
                        if isinstance(call, ast.Attribute) and call.attr == "caption":
                            found = True
        self.assertTrue(found, "YOLO caption should be guarded by an if statement")
        for node in tree.body:
            if isinstance(node, ast.Expr) and isinstance(node.value, ast.IfExp):
                self.fail("Do not use top-level conditional expressions for Streamlit UI elements")

if __name__ == "__main__":
    unittest.main()
