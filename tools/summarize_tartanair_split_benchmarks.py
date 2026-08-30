#!/usr/bin/env python3
import argparse
import csv
import json
import os


def _fmt_gaussians(n):
    n = int(n)
    if n >= 1_000_000:
        return f"{n / 1_000_000:.2f}M"
    if n >= 1_000:
        return f"{n / 1_000:.1f}k"
    return str(n)


def _load_rows(sequences, root, allow_missing=False):
    rows = []
    missing = []
    for seq in sequences:
        path = os.path.join(
            root, f"{seq}_full_split5", "benchmark_summary_split.json"
        )
        if not os.path.exists(path):
            missing.append((seq, path))
            if allow_missing:
                continue
            raise FileNotFoundError(
                f"Missing benchmark result for {seq}: {path}\n"
                "Run the full split benchmark first, or pass --allow-missing."
            )

        with open(path, "r", encoding="utf-8") as f:
            x = json.load(f)

        rows.append(
            {
                "Sequence": seq,
                "Method": "LSG-SLAM",
                "MaxMap": float(x["maxmap_ratio"]),
                "Train PSNR": float(x["train_psnr"]),
                "Train SSIM": float(x["train_ssim"]),
                "Test PSNR": float(x["test_psnr"]),
                "Test SSIM": float(x["test_ssim"]),
                "ATE(m)": float(x["ate_rmse_se3_m"]),
                "FPS": float(x["online_fps"]),
                "Gaussians": int(x["gaussians"]),
            }
        )
    return rows, missing


def _write_csv(rows, csv_path):
    headers = [
        "Sequence",
        "Method",
        "MaxMap",
        "ATE(m)",
        "Train PSNR",
        "Train SSIM",
        "Test PSNR",
        "Test SSIM",
        "FPS",
        "Gaussians",
    ]
    os.makedirs(os.path.dirname(csv_path) or ".", exist_ok=True)
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=headers)
        writer.writeheader()
        writer.writerows(rows)


def _compact_table(rows):
    print(
        f"{'Sequence':<10} {'MaxMap':>9} {'ATE RMSE(m)':>12} "
        f"{'Train PSNR/SSIM':>21} {'Test PSNR/SSIM':>21} "
        f"{'FPS':>10} {'Gaussians':>12}"
    )
    print("-" * 116)
    for r in rows:
        print(
            f"{r['Sequence']:<10} "
            f"{100.0*r['MaxMap']:>8.2f}% "
            f"{r['ATE(m)']:>12.4f} "
            f"{r['Train PSNR']:>7.2f}/{r['Train SSIM']:<12.4f} "
            f"{r['Test PSNR']:>7.2f}/{r['Test SSIM']:<12.4f} "
            f"{r['FPS']:>10.2f} "
            f"{_fmt_gaussians(r['Gaussians']):>12}"
        )


def _paper_block(row):
    return "\n".join(
        [
            f"## {row['Sequence']}",
            "",
            "| Method | MaxMap | ATE RMSE ↓ | Train PSNR/SSIM | Test PSNR/SSIM | FPS | Gaussians |",
            "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
            (
                f"| {row['Method']} | {100.0 * row['MaxMap']:.2f}% | "
                f"{row['ATE(m)']:.4f} m | "
                f"{row['Train PSNR']:.2f}/{row['Train SSIM']:.4f} | "
                f"{row['Test PSNR']:.2f}/{row['Test SSIM']:.4f} | "
                f"{row['FPS']:.2f} | {_fmt_gaussians(row['Gaussians'])} |"
            ),
            "",
        ]
    )


def _paper_terminal(rows):
    for row in rows:
        print()
        print(row["Sequence"])
        print(
            f"{'Method':<20} {'MaxMap':>10} {'ATE RMSE ↓':>14} "
            f"{'Train PSNR/SSIM':>19} {'Test PSNR/SSIM':>19} "
            f"{'FPS':>9} {'Gaussians':>12}"
        )
        print("-" * 110)
        print(
            f"{row['Method']:<20} "
            f"{100.0 * row['MaxMap']:>9.2f}% "
            f"{row['ATE(m)']:>11.4f} m "
            f"{row['Train PSNR']:>7.2f}/{row['Train SSIM']:<10.4f} "
            f"{row['Test PSNR']:>7.2f}/{row['Test SSIM']:<10.4f} "
            f"{row['FPS']:>9.2f} "
            f"{_fmt_gaussians(row['Gaussians']):>12}"
        )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "sequences",
        nargs="*",
        default=["SH000", "SH001", "SH002", "SH003"],
        help="Sequences to summarize. Default: SH000 SH001 SH002 SH003",
    )
    parser.add_argument(
        "--root",
        default="experiments/tartanair_split",
        help="Root containing <SEQ>_full_split5 directories",
    )
    parser.add_argument(
        "--csv",
        default=None,
        help="Output CSV path; default: <root>/summary_SH000_SH003_split5.csv",
    )
    parser.add_argument(
        "--paper-style",
        action="store_true",
        help="Print one paper-style table per sequence and save Markdown.",
    )
    parser.add_argument(
        "--markdown",
        default=None,
        help="Markdown output path used with --paper-style.",
    )
    parser.add_argument(
        "--allow-missing",
        action="store_true",
        help="Summarize completed sequences and skip missing ones.",
    )
    args = parser.parse_args()

    sequences = args.sequences or ["SH000", "SH001", "SH002", "SH003"]
    rows, missing = _load_rows(sequences, args.root, allow_missing=args.allow_missing)

    if not rows:
        raise RuntimeError("No completed benchmark summaries found.")

    if args.paper_style:
        _paper_terminal(rows)
        markdown_path = args.markdown or os.path.join(
            args.root, "paper_tables_SH000_SH003_split5.md"
        )
        markdown = "# LSG-SLAM TartanAir 8:2 Results\n\n" + "\n".join(
            _paper_block(row) for row in rows
        )
        os.makedirs(os.path.dirname(markdown_path) or ".", exist_ok=True)
        with open(markdown_path, "w", encoding="utf-8") as f:
            f.write(markdown)
        print(f"\nSaved Markdown: {markdown_path}")
    else:
        _compact_table(rows)

    csv_path = args.csv or os.path.join(
        args.root, "summary_SH000_SH003_split5.csv"
    )
    _write_csv(rows, csv_path)
    print(f"Saved CSV:      {csv_path}")

    if missing:
        print("\nMissing / unfinished sequences:")
        for seq, path in missing:
            print(f"  {seq}: {path}")


if __name__ == "__main__":
    main()
