"""SHA-256 of a file as stored, with LF line endings and with CRLF line endings (added 31 October 2026).

    python src/04_revision/hash_line_endings.py data/UNSW_NB15_training-set.csv data/UNSW_NB15_testing-set.csv

Used to explain why the first author's UNSW-NB15 copies had other SHA-256 values than the recorded ones: the
recorded files use CRLF line endings, and his copies have the same bytes with LF line endings."""
import hashlib
import sys

for f in sys.argv[1:]:
    b = open(f, "rb").read()
    lf = b.replace(b"\r\n", b"\n")
    print(f"{f}\n  as stored {hashlib.sha256(b).hexdigest()}\n  LF        {hashlib.sha256(lf).hexdigest()}\n"
          f"  CRLF      {hashlib.sha256(lf.replace(b'\n', b'\r\n')).hexdigest()}")
