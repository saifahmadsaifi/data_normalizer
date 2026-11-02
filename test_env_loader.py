"""Offline tests for env_loader.py.

Run:  python -m unittest test_env_loader -v
"""

import os
import tempfile
import unittest
from unittest import mock

from env_loader import load_env, parse_env_line


class ParseLineTests(unittest.TestCase):
    def test_valid_lines(self):
        self.assertEqual(parse_env_line("A=1"), ("A", "1"))
        self.assertEqual(parse_env_line("  A = hello  "), ("A", "hello"))
        self.assertEqual(parse_env_line('A="quoted value"'), ("A", "quoted value"))
        self.assertEqual(parse_env_line("A='single'"), ("A", "single"))
        self.assertEqual(parse_env_line("export A=1"), ("A", "1"))
        self.assertEqual(parse_env_line("A=val # comment"), ("A", "val"))
        self.assertEqual(parse_env_line("A=x=y"), ("A", "x=y"))  # '=' inside value kept
        self.assertEqual(parse_env_line("\ufeffA=1"), ("A", "1"))  # Windows BOM

    def test_ignored_lines(self):
        for line in ["", "   ", "# comment", "no equals sign", "=novalue", "BAD KEY=1"]:
            self.assertIsNone(parse_env_line(line), repr(line))

    def test_hash_inside_quotes_is_kept(self):
        self.assertEqual(parse_env_line('A="has # inside"'), ("A", "has # inside"))


class LoadEnvTests(unittest.TestCase):
    def write(self, text):
        handle = tempfile.NamedTemporaryFile("w", suffix=".env", delete=False, encoding="utf-8")
        handle.write(text)
        handle.close()
        self.addCleanup(os.remove, handle.name)
        return handle.name

    def test_loads_values(self):
        path = self.write("# c\nGEMINI_API_KEY=abc123\nGEMINI_MODEL=m-test\n\n")
        with mock.patch.dict(os.environ, {}, clear=True):
            applied = load_env(path)
            self.assertEqual(os.environ["GEMINI_API_KEY"], "abc123")
            self.assertEqual(os.environ["GEMINI_MODEL"], "m-test")
        self.assertEqual(set(applied), {"GEMINI_API_KEY", "GEMINI_MODEL"})

    def test_existing_environment_wins(self):
        path = self.write("GEMINI_API_KEY=from_file\n")
        with mock.patch.dict(os.environ, {"GEMINI_API_KEY": "from_terminal"}, clear=True):
            applied = load_env(path)
            self.assertEqual(os.environ["GEMINI_API_KEY"], "from_terminal")
        self.assertEqual(applied, {})

    def test_override_flag(self):
        path = self.write("GEMINI_API_KEY=from_file\n")
        with mock.patch.dict(os.environ, {"GEMINI_API_KEY": "from_terminal"}, clear=True):
            load_env(path, override=True)
            self.assertEqual(os.environ["GEMINI_API_KEY"], "from_file")

    def test_missing_file_is_fine(self):
        self.assertEqual(load_env("definitely_not_here.env"), {})

    def test_works_with_client_config(self):
        import llm_client as lc
        path = self.write('GEMINI_API_KEY="k-from-file"\nGEMINI_MODEL=model-from-file\n')
        with mock.patch.dict(os.environ, {}, clear=True):
            load_env(path)
            client = lc.GeminiClient()
            self.assertEqual(client.model, "model-from-file")


if __name__ == "__main__":
    unittest.main()
    