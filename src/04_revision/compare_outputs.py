"""
Content-level comparison of two result folders (added 31 October 2026 after the supervisor's note that
"file differs (check by eye)" in the re-run report is not evidence).

    python src/04_revision/compare_outputs.py <reference_folder> <new_folder> [report.md]

Rules, per file present in both folders:
  .json  every leaf compared; numbers equal within 1e-9; keys holding run times or platform ignored
  .csv   parsed with pandas; same columns and shape; numeric columns equal within 1e-9, text columns equal
  .npz   same array names and shapes; numeric arrays equal within 1e-9 (np.allclose, NaN equal to NaN); text
         (object) arrays are not unpickled and are compared by their stored bytes
  .md, .txt  text equal after normalizing line endings (CRLF to LF) and removing a byte-order mark
  .png, .pdf drawn from the compared results; reported as not compared
For differing numeric values the largest absolute difference is reported.
"""
import json
import sys
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

IGNORE_KEYS = ("seconds", "platform", "python")
TOL = 1e-9


def leaves(x, p=""):
    if isinstance(x, dict):
        for k, v in x.items():
            yield from leaves(v, f"{p}/{k}")
    elif isinstance(x, list):
        for i, v in enumerate(x):
            yield from leaves(v, f"{p}[{i}]")
    else:
        yield p, x


def num(x):
    return isinstance(x, (int, float)) and not isinstance(x, bool)


def compare(ref: Path, new: Path) -> str:
    s = ref.suffix.lower()
    if s == ".json":
        a = dict(leaves(json.loads(ref.read_text("utf-8-sig"))))
        b = dict(leaves(json.loads(new.read_text("utf-8-sig"))))
        diff, worst = [], 0.0
        for k in sorted(set(a) | set(b)):
            if any(t in k for t in IGNORE_KEYS):
                continue
            x, y = a.get(k, "MISSING"), b.get(k, "MISSING")
            if num(x) and num(y):
                if abs(x - y) > TOL:
                    diff.append(k); worst = max(worst, abs(x - y))
            elif x != y:
                diff.append(k)
        return "identical" if not diff else f"{len(diff)} values differ (largest absolute difference {worst:.6g}; e.g. {diff[0]})"
    if s == ".csv":
        a, b = pd.read_csv(ref, encoding="utf-8-sig"), pd.read_csv(new, encoding="utf-8-sig")
        if list(a.columns) != list(b.columns) or a.shape != b.shape:
            return f"different columns or shape ({a.shape} against {b.shape})"
        bad = []
        for c in a.columns:
            x, y = a[c], b[c]
            if pd.api.types.is_numeric_dtype(x) and pd.api.types.is_numeric_dtype(y):
                if not np.allclose(x.to_numpy(float), y.to_numpy(float), atol=TOL, rtol=0, equal_nan=True):
                    bad.append(f"{c} (largest absolute difference {np.nanmax(np.abs(x.to_numpy(float) - y.to_numpy(float))):.6g})")
            elif not x.astype(str).equals(y.astype(str)):
                bad.append(c)
        return f"same content ({a.shape[0]} rows x {a.shape[1]} columns; line endings ignored)" if not bad else "columns differ: " + ", ".join(bad)
    if s == ".npz":
        a, b = np.load(ref, allow_pickle=False), np.load(new, allow_pickle=False)
        if sorted(a.files) != sorted(b.files):
            return f"different arrays ({sorted(a.files)} against {sorted(b.files)})"
        bad, note = [], []
        for k in a.files:
            try:
                x, y = a[k], b[k]
            except ValueError:            # object (text) array: not unpickled, compared as stored bytes
                za, zb = zipfile.ZipFile(ref), zipfile.ZipFile(new)
                same = za.read(k + ".npy") == zb.read(k + ".npy")
                (note if same else bad).append(f"{k} (text array, {'identical' if same else 'different'} stored bytes)")
                continue
            if x.shape != y.shape:
                bad.append(f"{k} shape {x.shape} against {y.shape}")
            elif np.issubdtype(x.dtype, np.number) and np.issubdtype(y.dtype, np.number):
                if not np.allclose(x, y, atol=TOL, rtol=0, equal_nan=True):
                    bad.append(f"{k} (largest absolute difference {np.nanmax(np.abs(x.astype(float) - y.astype(float))):.6g})")
            elif not np.array_equal(x, y):
                bad.append(k)
        return (f"same content ({len(a.files)} arrays; numeric arrays within 1e-9" + (f"; {', '.join(note)}" if note else "") + ")") if not bad \
            else "arrays differ: " + "; ".join(bad + note)
    if s in (".md", ".txt"):
        norm = lambda p: p.read_bytes().decode("utf-8-sig").replace("\r\n", "\n")
        return "same text (line endings ignored)" if norm(ref) == norm(new) else "text differs"
    if s in (".png", ".pdf"):
        return "figure file, not compared (drawn from the compared result files)"
    return "byte-identical" if ref.read_bytes() == new.read_bytes() else "differs (type not compared by content)"


def main():
    ref, new = Path(sys.argv[1]), Path(sys.argv[2])
    rows = []
    for p in sorted(new.glob("*")):
        if p.is_file() and (ref / p.name).exists():
            rows.append((p.name, compare(ref / p.name, p)))
    lines = [f"# Content comparison", "", f"Reference: {ref}", f"New: {new}", "", "| File | Result |", "| --- | --- |"]
    lines += [f"| {a} | {b} |" for a, b in rows]
    text = "\n".join(lines) + "\n"
    if len(sys.argv) > 3:
        Path(sys.argv[3]).write_text(text, "utf-8")
    print(text)


if __name__ == "__main__":
    main()
