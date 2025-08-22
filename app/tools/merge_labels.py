import sys, csv

if len(sys.argv) != 3:
    print("usage: merge_labels.py <log_csv> <labels_csv>", file=sys.stderr)
    sys.exit(1)

log_csv, lab_csv = sys.argv[1], sys.argv[2]
labels = {}
with open(lab_csv, newline="") as f:
    r = csv.reader(f)
    for row in r:
        if not row: continue
        req, lab = row[0].strip(), (row[1].strip().upper() if len(row)>1 else "")
        if req:
            labels[req] = lab

with open(log_csv, newline="") as f, sys.stdout as out:
    r = csv.DictReader(f)
    fieldnames = r.fieldnames or []
    if "label" not in fieldnames:
        fieldnames = fieldnames + ["label"]
    w = csv.DictWriter(out, fieldnames=fieldnames)
    w.writeheader()
    for row in r:
        rid = (row.get("request_id") or "").strip()
        if "label" not in row:
            row["label"] = ""
        if rid in labels:
            row["label"] = labels[rid]
        w.writerow(row)
