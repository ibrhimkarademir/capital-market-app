"""Modül importları ve sayfa sözdizimi kontrolleri."""

from __future__ import annotations

import ast
import importlib
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]


class ImportTests(unittest.TestCase):
    """Yardımcı modüllerin yüklenebilir olduğunu ve beklenen fonksiyonları test eder."""

    def test_utility_modules_import(self) -> None:
        """Temel yardımcı modüller doğrudan import edilebilmelidir."""
        for module_name in (
            "utils.data_loader",
            "utils.chart_utils",
            "utils.results_loader",
            "utils.ui_components",
            "utils.presentation_mode",
        ):
            with self.subTest(module=module_name):
                importlib.import_module(module_name)

    def test_results_loader_functions_exist(self) -> None:
        """Sonuç yükleyicide beklenen geriye uyumlu fonksiyonlar bulunmalıdır."""
        module = importlib.import_module("utils.results_loader")
        for function_name in (
            "load_prediction_data",
            "load_future_forecast_data",
            "load_interval_prediction_data",
            "load_interval_future_forecast_data",
        ):
            with self.subTest(function=function_name):
                self.assertTrue(callable(getattr(module, function_name, None)))

    def test_python_files_have_valid_syntax(self) -> None:
        """Sayfa ve yardımcı dosyaların sözdizimi geçerli olmalıdır."""
        folders = ("pages", "utils", "scripts", "tests")
        python_files = [PROJECT_ROOT / "app.py"]
        for folder in folders:
            folder_path = PROJECT_ROOT / folder
            if folder_path.exists():
                python_files.extend(folder_path.rglob("*.py"))

        for file_path in python_files:
            with self.subTest(file=file_path.relative_to(PROJECT_ROOT).as_posix()):
                ast.parse(file_path.read_text(encoding="utf-8"))


if __name__ == "__main__":
    unittest.main()
