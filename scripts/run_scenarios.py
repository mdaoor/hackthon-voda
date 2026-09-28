"""Replay the organisers' example scenarios against a running companion (local or deployed).

    python scripts/run_scenarios.py --url https://<service>.ecs.<region>.on.aws --user U000001
    python scripts/run_scenarios.py --url http://localhost:8080 --user U000001 --fresh

Prints each reply with the tool actions, so you can rehearse before recording the
screen video. Uses only the standard library.
"""
import argparse
import json
import urllib.error
import urllib.request

SCENARIOS = [
    "I'm preparing my new home. Can you help me choose what I need?",
    "I would like to buy a new smartphone. What would suit me?",
    "That's too expensive. My budget is now 100, and I don't want the first option.",
    "Find something in this category within my budget.",
    "Please add the first option to my basket and check out.",
    "Remove that item before placing the order.",
]


def call(base, path, body=None):
    req = urllib.request.Request(base.rstrip("/") + path, data=json.dumps(body).encode() if body else None,
                                 headers={"Content-Type": "application/json"}, method="POST" if body else "GET")
    try:
        with urllib.request.urlopen(req, timeout=120) as r:
            return json.loads(r.read())
    except urllib.error.HTTPError as e:
        return {"http_error": e.code, "detail": e.read().decode()[:300]}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--url", default="http://localhost:8080")
    p.add_argument("--user", default="U000001")
    p.add_argument("--fresh", action="store_true", help="start a new conversation first")
    p.add_argument("--scenario", action="append", help="custom message(s) instead of the defaults")
    a = p.parse_args()

    print("Invalid ID check:", call(a.url, "/api/session/start", {"user_id": "NOT_A_USER"}).get("detail"))
    start = call(a.url, "/api/session/reset" if a.fresh else "/api/session/start", {"user_id": a.user})
    if "http_error" in start:
        raise SystemExit(start)
    print("\nGREETING:", start["transcript"][-1]["text"] if start["transcript"] else "(none)")
    for msg in a.scenario or SCENARIOS:
        r = call(a.url, "/api/chat", {"user_id": a.user, "message": msg, "channel": "text"})
        print("\n" + "=" * 90 + f"\nCUSTOMER: {msg}\n" + "-" * 90)
        if "http_error" in r:
            print(r)
            continue
        for act in r.get("actions", []):
            print(f"  [{'ok' if act['ok'] else 'X '}] {act['tool']} {json.dumps(act['input'])[:110]} -> {act['summary']}")
        print(f"\nCOMPANION: {r['text']}")
        print(f"  basket total={r['basket']['total']} items={[i['product_name'] for i in r['basket']['items']]}"
              f" pending_checkout={bool(r.get('pending_checkout'))}")
    print("\nMemory:", json.dumps(call(a.url, f"/api/customer/{a.user}/memory"), default=str)[:800])


if __name__ == "__main__":
    main()
