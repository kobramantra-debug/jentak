#!/usr/bin/env python3
"""Overi tokeny v .env, aniz by je kamkoliv vytiskl.

Pouze stdlib, zadne zavislosti. Konci kodem 0 kdyz je vse OK.
Cte .env ze stejneho adresare jako je repo root (rodic scripts/).
"""
import json
import os
import sys
import urllib.request

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def load_dotenv(path):
    vals = {}
    if not os.path.exists(path):
        return vals
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            k, v = line.split("=", 1)
            vals[k.strip()] = v.strip().strip("'\"")
    return vals


def check_hf(token):
    req = urllib.request.Request(
        "https://huggingface.co/api/whoami-v2",
        headers={"Authorization": f"Bearer {token}"},
    )
    with urllib.request.urlopen(req, timeout=20) as r:
        return json.load(r).get("name", "?"), r.status


def check_gh(token):
    req = urllib.request.Request(
        "https://api.github.com/user",
        headers={
            "Authorization": f"Bearer {token}",
            "Accept": "application/vnd.github+json",
        },
    )
    with urllib.request.urlopen(req, timeout=20) as r:
        return json.load(r).get("login", "?"), r.status


def main():
    env = load_dotenv(os.path.join(ROOT, ".env"))
    ok = True

    hf = env.get("HF_TOKEN", "")
    if not hf:
        print("HF_TOKEN:     CHYBI – dopln do .env")
        ok = False
    else:
        try:
            name, _ = check_hf(hf)
            print(f"HF_TOKEN:     OK (ucet {name})")
        except Exception as e:
            print(f"HF_TOKEN:     NEPLATNY ({e})")
            ok = False

    gh = env.get("GITHUB_TOKEN", "")
    if not gh:
        print("GITHUB_TOKEN: CHYBI – dopln do .env")
        ok = False
    else:
        try:
            login, _ = check_gh(gh)
            print(f"GITHUB_TOKEN: OK (ucet {login})")
        except Exception as e:
            print(f"GITHUB_TOKEN: NEPLATNY ({e})")
            ok = False

    # Jmena uctu jdou do stdout schvalne – tokeny nikdy.
    print("OK – muzu zalozit repo" if ok else "Dopln chybejici/neplatne tokeny do .env")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
