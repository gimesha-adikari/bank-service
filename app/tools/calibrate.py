import sys, csv, argparse, math, statistics as stats
from collections import defaultdict

METRICS = [
    ("face", "face_score"),
    ("live", "live_score"),
    ("ocr",  "ocr_score"),
    ("doc",  "doc_score"),
]

def qtile(sorted_vals, q: float):
    n = len(sorted_vals)
    if n == 0:
        return None
    if n == 1:
        return float(sorted_vals[0])
    pos = (n - 1) * max(0.0, min(1.0, q))
    i = int(math.floor(pos))
    frac = pos - i
    if i + 1 < n:
        return float(sorted_vals[i] * (1 - frac) + sorted_vals[i + 1] * frac)
    return float(sorted_vals[-1])

def to_float(x):
    try:
        return float(x)
    except Exception:
        return None

def parse_rows(paths):
    for p in paths:
        with open(p, newline="") as f:
            r = csv.DictReader(f)
            for row in r:
                yield row

def group_key(row):
    c = (row.get("country") or "UNK").strip() or "UNK"
    d = (row.get("doc_class") or row.get("docClass") or "UNK").strip() or "UNK"
    return (c, d)

def collect(paths):
    groups = defaultdict(lambda: {
        "GOOD": {m[0]: [] for m in METRICS},
        "BAD":  {m[0]: [] for m in METRICS},
        "ALL":  {m[0]: [] for m in METRICS},
        "N_GOOD": 0, "N_BAD": 0, "N_ALL": 0
    })
    for row in parse_rows(paths):
        key = group_key(row)
        label = (row.get("label") or "").strip().upper()
        bucket = "ALL"
        if label in ("HUMAN_OK", "OK", "GOOD", "POS"):
            bucket = "GOOD"
            groups[key]["N_GOOD"] += 1
        elif label in ("HUMAN_FAIL", "FAIL", "BAD", "NEG"):
            bucket = "BAD"
            groups[key]["N_BAD"] += 1
        groups[key]["N_ALL"] += 1

        for short, col in METRICS:
            v = to_float(row.get(col))
            if v is not None:
                groups[key][bucket][short].append(v)
                groups[key]["ALL"][short].append(v)
    return groups

def summarize(vals):
    if not vals:
        return {"n":0}
    s = sorted(v for v in vals if v is not None)
    return {
        "n": len(s),
        "mean": sum(s)/len(s),
        "p05": qtile(s, 0.05),
        "p10": qtile(s, 0.10),
        "p50": qtile(s, 0.50),
        "p90": qtile(s, 0.90),
        "p95": qtile(s, 0.95),
        "min": s[0],
        "max": s[-1],
    }

def rate_at_threshold(vals, thr, pass_if_ge=True):
    if not vals:
        return None
    if pass_if_ge:
        num = sum(1 for v in vals if v >= thr)
    else:
        num = sum(1 for v in vals if v < thr)
    return num / len(vals)

def suggest_threshold(good_vals, bad_vals, far_target):

    thr_far = None
    if bad_vals:
        s_bad = sorted(bad_vals)
        thr_far = qtile(s_bad, 1.0 - far_target)
    thr_good_floor = None
    if good_vals:
        s_good = sorted(good_vals)
        thr_good_floor = qtile(s_good, 0.10)
    candidates = [t for t in (thr_far, thr_good_floor) if t is not None]
    if not candidates:
        return None
    return max(candidates)

def fmt(x, nd=3):
    return "NA" if x is None else f"{x:.{nd}f}"

def main():
    ap = argparse.ArgumentParser(description="Calibrate thresholds from KYC aggregate CSV logs.")
    ap.add_argument("csv", nargs="+", help="One or more CSV files (e.g., ./calib/*.csv)")
    ap.add_argument("--far", type=float, default=0.01, help="Target FAR (default 0.01 = 1%)")
    ap.add_argument("--env-prefix", default="APP", help="Env var prefix (default APP)")
    args = ap.parse_args()

    groups = collect(args.csv)
    if not groups:
        print("No data.")
        return 0

    print(f"\n=== KYC Calibration Report (FAR target={args.far:.2%}) ===\n")

    env_lines = []

    for (country, doccls), data in sorted(groups.items()):
        print(f"\n--- Group: country={country}, doc_class={doccls} ---")
        print(f"Samples: ALL={data['N_ALL']}  GOOD={data['N_GOOD']}  BAD={data['N_BAD']}")
        for short, _ in METRICS:
            s_all  = summarize(data["ALL"][short])
            s_good = summarize(data["GOOD"][short])
            s_bad  = summarize(data["BAD"][short])

            print(f"\n[{short.upper()}]")
            print(" ALL :", f"n={s_all.get('n',0)}",
                  f"mean={fmt(s_all.get('mean'))}",
                  f"p10={fmt(s_all.get('p10'))}",
                  f"p50={fmt(s_all.get('p50'))}",
                  f"p90={fmt(s_all.get('p90'))}",
                  f"min={fmt(s_all.get('min'))}", f"max={fmt(s_all.get('max'))}")

            if s_good.get("n",0) > 0:
                print(" GOOD:", f"n={s_good['n']}",
                      f"p10={fmt(s_good.get('p10'))}",
                      f"p50={fmt(s_good.get('p50'))}",
                      f"p90={fmt(s_good.get('p90'))}")
            else:
                print(" GOOD: n=0")

            if s_bad.get("n",0) > 0:
                print(" BAD :", f"n={s_bad['n']}",
                      f"p90={fmt(s_bad.get('p90'))}",
                      f"p95={fmt(s_bad.get('p95'))}")
            else:
                print(" BAD : n=0")

            thr = suggest_threshold(data["GOOD"][short], data["BAD"][short], args.far)
            if thr is not None:
                far = rate_at_threshold(data["BAD"][short], thr, pass_if_ge=True)
                frr = 1.0 - rate_at_threshold(data["GOOD"][short], thr, pass_if_ge=True) if data["GOOD"][short] else None
                print(f" -> suggested_threshold={fmt(thr,3)}  (est FAR={fmt(far,3)}, FRR={fmt(frr,3)})")

                env_key_suffix = f"__{country}__{doccls}".upper().replace(" ", "_")
                env_key = {
                    "face": f"{args.env_prefix}_FACE_THRESHOLD{env_key_suffix}",
                    "live": f"{args.env_prefix}_LIVE_THRESHOLD{env_key_suffix}",
                    "ocr":  f"{args.env_prefix}_OCR_THRESHOLD{env_key_suffix}",
                    "doc":  f"{args.env_prefix}_DOC_THRESHOLD{env_key_suffix}",
                }[short]
                env_lines.append(f"{env_key}={thr:.2f}")
            else:
                print(" -> suggested_threshold=NA (insufficient labeled GOOD/BAD data)")

    if env_lines:
        print("\nSuggested .env lines (copy/paste what you need):")
        for line in env_lines:
            print(line)

    print("\nDone.")
    return 0

if __name__ == "__main__":
    sys.exit(main())
