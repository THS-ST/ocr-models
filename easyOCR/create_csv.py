import pandas as pd
from pathlib import Path

csv_input = Path("ground_truth_lines_rel.csv")  # your CSV

# Load CSV
df = pd.read_csv(csv_input)

# Find rows where 'Unnamed: 2' is not empty
if "Unnamed: 2" in df.columns:
    non_empty_rows = df[df["Unnamed: 2"].notna() & (df["Unnamed: 2"].astype(str).str.strip() != "")]
    print(f"Found {len(non_empty_rows)} non-empty row(s) in 'Unnamed: 2':\n")
    for idx, row in non_empty_rows.iterrows():
        print(f"Row {idx + 2}: {row.to_dict()}")
else:
    print("No 'Unnamed: 2' column found.")
