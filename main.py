"""Запуск настольного приложения B² PolyCrystal."""
from pathlib import Path
import argparse
import sys


def main() -> int:
    parser = argparse.ArgumentParser(description="B² PolyCrystal — анализ SEM-изображений")
    parser.add_argument("--result", type=Path, help="Открыть сохранённую папку результата")
    parser.add_argument("--screenshot", type=Path, help="Сохранить снимок окна и завершить приложение")
    args = parser.parse_args()
    sys.path.insert(0, str(Path(__file__).resolve().parent / "vendor"))
    from bb_polycrystal.gui import launch
    return launch(result_folder=args.result, screenshot_path=args.screenshot)


if __name__ == "__main__":
    raise SystemExit(main())
