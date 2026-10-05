"""Buyruqlar qatori interfeysi.

Ikkita rejim mavjud:
  * ``python run.py``           - saqlangan ma'lumotdan hisobot tayyorlaydi
  * ``python run.py --update``  - OLX va Uybor saytlaridan yangi ma'lumot
                                   yig'adi, keyin hisobot tayyorlaydi
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from .config import Settings


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="uzhousing",
        description="O'zbekiston uy-joy bozori sharhini tayyorlaydi.",
    )
    parser.add_argument(
        "--update",
        action="store_true",
        help="OLX.uz va Uybor.uz dan yangi ma'lumot yig'adi va hisobot "
             "tayyorlaydi. Berilmasa, saqlangan oxirgi ma'lumotdan foydalanadi.",
    )
    parser.add_argument("--verbose", "-v", action="store_true", help="to'liq diagnostika")
    return parser


def _latest_snapshot(output_dir: Path) -> Path | None:
    folder = output_dir / "olx_snapshots"
    if not folder.exists():
        return None
    snapshots = sorted(folder.glob("*.json"))
    return snapshots[-1] if snapshots else None


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.WARNING,
        format="%(levelname)s %(name)s: %(message)s",
    )

    settings = Settings.from_env()
    settings.ensure_dirs()

    def progress(message: str) -> None:
        print(f"  {message}", flush=True)

    print("\n" + "=" * 72)
    print("  O'zbekiston uy-joy bozori sharhi")
    print("=" * 72)

    snapshot_path: Path | None = None
    if not args.update:
        snapshot_path = _latest_snapshot(settings.output_dir)
        if snapshot_path is None:
            print(
                "Saqlangan ma'lumot topilmadi. Avval 'python run.py --update' "
                "buyrug'ini ishga tushiring.",
                file=sys.stderr,
            )
            return 2
        progress(f"Saqlangan ma'lumot ishlatilmoqda: {snapshot_path.name}")

    from .report.olx_bulletin import run_olx

    try:
        result = run_olx(
            settings,
            progress=progress,
            snapshot_path=snapshot_path,
            browser=args.update,
            archive=settings.olx_archive or None,
        )
    except Exception as exc:
        print(f"\nXATOLIK: {exc}", file=sys.stderr)
        if args.verbose:
            raise
        print("To'liq diagnostika uchun --verbose bilan qayta ishga tushiring.", file=sys.stderr)
        return 1

    print("-" * 72)
    print(f"  PDF:   {result.pdf_path}")
    print(f"  Word:  {result.report_path}")
    print(f"  Log:   {result.run_log}")
    print(f"  Vaqt:  {result.duration_seconds:.1f}s\n")
    return 0 if result.ok else 1


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
