from __future__ import annotations

import argparse
import csv
from pathlib import Path


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def pct(value: str) -> str:
    return f"{float(value) * 100:.1f}%"


def write_table(path: Path, header: list[str], rows: list[list[str]]) -> None:
    widths = [len(item) for item in header]
    for row in rows:
        widths = [max(width, len(cell)) for width, cell in zip(widths, row)]

    def fmt(row: list[str]) -> str:
        return "| " + " | ".join(cell.ljust(width) for cell, width in zip(row, widths)) + " |"

    lines = [
        fmt(header),
        "| " + " | ".join("-" * width for width in widths) + " |",
    ]
    lines.extend(fmt(row) for row in rows)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def build_measurement(data_dir: Path, out_dir: Path) -> None:
    rows = read_rows(data_dir / "pfe_measurement_summary.csv")
    table_rows = []
    for row in rows:
        ci = f"[{row['ci_low']}, {row['ci_high']}]"
        table_rows.append([row["metric"], row["baseline_value"], row["current_value"], ci, row["note"]])
    write_table(
        out_dir / "table_measurement_summary.md",
        ["Metric", "Baseline", "Current", "95% CI", "Note"],
        table_rows,
    )


def build_icc(data_dir: Path, out_dir: Path) -> None:
    rows = read_rows(data_dir / "pfe_country_icc.csv")
    table_rows = []
    for row in rows:
        ci = f"[{row['ci_low']}, {row['ci_high']}]"
        table_rows.append([row["model_component"], row["value"], ci, row["p_value"], row["note"]])
    write_table(out_dir / "table_country_icc.md", ["Component", "Value", "95% CI", "p", "Note"], table_rows)


def build_regional(data_dir: Path, out_dir: Path) -> None:
    rows = read_rows(data_dir / "pfe_regional_contrasts.csv")
    table_rows = []
    for row in rows:
        a = f"{row['events_a']}/{row['n_a']} ({pct(row['prevalence_a'])})"
        b = f"{row['events_b']}/{row['n_b']} ({pct(row['prevalence_b'])})"
        table_rows.append([row["theme"], row["contrast"], a, b, row["p_value"], row["note"]])
    write_table(
        out_dir / "table_regional_contrasts.md",
        ["Theme", "Contrast", "Region A", "Region B", "p", "Note"],
        table_rows,
    )


def build_temporal(data_dir: Path, out_dir: Path) -> None:
    rows = read_rows(data_dir / "pfe_temporal_drift.csv")
    table_rows = []
    for row in rows:
        mid = f"{pct(row['mid_2023_prevalence'])} ({row['mid_2023_events']}/{row['mid_2023_n']})"
        late = f"{pct(row['late_2025_prevalence'])} ({row['late_2025_events']}/{row['late_2025_n']})"
        table_rows.append([row["theme"], mid, late, f"+{row['shift_pp']} pp", row["p_value"], row["note"]])
    write_table(
        out_dir / "table_temporal_drift.md",
        ["Theme", "Mid-2023", "Late-2025", "Shift", "p", "Note"],
        table_rows,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--out-dir", type=Path, default=Path("tables"))
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    build_measurement(args.data_dir, args.out_dir)
    build_icc(args.data_dir, args.out_dir)
    build_regional(args.data_dir, args.out_dir)
    build_temporal(args.data_dir, args.out_dir)

    print(f"Wrote locked PFE result tables to {args.out_dir}")


if __name__ == "__main__":
    main()

