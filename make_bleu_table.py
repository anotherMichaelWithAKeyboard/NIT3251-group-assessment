"""
Report-ready BLEU table (report section 4.3, Table 2).

Reads bleu_comparison.csv, which holds the full test set evaluation at full floating
point precision, and writes the same figures rounded to four decimal places in the two
forms the report needs:

    bleu_table.csv   for pasting into Word as a table
    bleu_table.md    for pasting into the README or any Markdown document

Rounding happens here rather than by hand, so the report and the repository cannot drift
apart. The source values come from decoding all 810 held out test images with each
strategy, which is a separate and larger evaluation than the 250 image subset used to
rank the hyperparameter arms in section 4.5.

Run from the project root:
    python make_bleu_table.py
"""

import os

import pandas as pd

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

DECIMALS = 4


def main():
    df = pd.read_csv(os.path.join(BASE_DIR, "bleu_comparison.csv"))

    # Formatted as fixed width strings rather than rounded floats, so trailing zeros
    # survive into the CSV. Rounding alone writes 0.328 where the table needs 0.3280,
    # and a column with inconsistent decimal places reads badly in the report.
    def fixed(series):
        return series.map(lambda v: f"{v:.{DECIMALS}f}")

    # Name the beam width in the column heading so the table stands on its own when it
    # is lifted out of the report.
    table = pd.DataFrame({
        "Metric": df["Metric"],
        "Greedy": fixed(df["Greedy"]),
        "Beam search (k = 3)": fixed(df["Beam Search"]),
    })

    csv_path = os.path.join(BASE_DIR, "bleu_table.csv")
    table.to_csv(csv_path, index=False)

    # Written by hand rather than through DataFrame.to_markdown, which needs the
    # tabulate package. A four row table is not worth another dependency.
    md_path = os.path.join(BASE_DIR, "bleu_table.md")
    with open(md_path, "w", encoding="utf-8") as f:
        f.write("# Table 2. BLEU Scores for Greedy Decoding and Beam Search\n\n")
        f.write("Corpus BLEU over all 810 held out test images.\n\n")
        f.write("| Metric | Greedy | Beam search (k = 3) |\n")
        f.write("|---|---:|---:|\n")
        for _, row in table.iterrows():
            f.write(f"| {row['Metric']} | {row['Greedy']} "
                    f"| {row['Beam search (k = 3)']} |\n")

    print(table.to_string(index=False))
    print(f"\nwrote {csv_path}")
    print(f"wrote {md_path}")


if __name__ == "__main__":
    main()
