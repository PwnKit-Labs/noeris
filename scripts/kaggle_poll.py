"""Poll a Kaggle kernel and pull results when complete.

Usage: KAGGLE_API_TOKEN=... python scripts/kaggle_poll.py [kernel-slug]
Default slug: peaktwilight/noeris-gpu-benchmark-14-operators-on-t4
"""
import argparse, os, sys, time, tempfile, json
from kaggle.api.kaggle_api_extended import KaggleApi

DEFAULT_SLUG = "peaktwilight/noeris-gpu-benchmark-14-operators-on-t4"

def get_api():
    api = KaggleApi(); api.authenticate(); return api

def check_status(api, slug):
    resp = api.kernels_status(slug)
    return resp["status"], resp.get("failureMessage")


def pull_output(api, slug):
    out_dir = tempfile.mkdtemp(prefix="kaggle_out_")
    api.kernels_output(slug, path=out_dir)
    files = os.listdir(out_dir)
    print(f"Output downloaded to {out_dir} ({len(files)} file(s))")
    for f in files:
        fp = os.path.join(out_dir, f)
        print(f"  {f}  ({os.path.getsize(fp):,} bytes)")
        if f.endswith(".json") and os.path.getsize(fp) < 50_000:
            with open(fp) as fh:
                print(json.dumps(json.load(fh), indent=2)[:2000])

def poll_once(api, slug):
    status, failure = check_status(api, slug)
    if status in ("running", "queued"):
        print(f"[{time.strftime('%H:%M:%S')}] Kernel {status}")
        return 0
    if status == "complete":
        print(f"[{time.strftime('%H:%M:%S')}] Kernel complete")
        pull_output(api, slug)
        return 0
    print(f"[{time.strftime('%H:%M:%S')}] Kernel failed: {status}")
    if failure: print(f"  {failure}")
    return 1

def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("slug", nargs="?", default=DEFAULT_SLUG)
    ap.add_argument("--wait", action="store_true", help="Poll until finished")
    ap.add_argument("--interval", type=int, default=60, help="Poll interval (s)")
    args = ap.parse_args()

    api = get_api()
    if not args.wait:
        sys.exit(poll_once(api, args.slug))

    while True:
        status, failure = check_status(api, args.slug)
        print(f"[{time.strftime('%H:%M:%S')}] {status}")
        if status == "complete":
            pull_output(api, args.slug)
            sys.exit(0)
        if status not in ("running", "queued"):
            print(f"Kernel failed: {status} — {failure}")
            sys.exit(1)
        time.sleep(args.interval)


if __name__ == "__main__":
    main()
