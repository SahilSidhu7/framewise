"""Make AIEditor licence keys. Seller-side only: never ship this with the app, never commit the private key.

    python tools/licence_issuer.py keygen                      # once: new signing key + updates the engine's PUBLIC_KEY
    python tools/licence_issuer.py issue --plan pro --to "Name"          # lifetime Pro key
    python tools/licence_issuer.py issue --plan pro --days 31 --to "..."  # monthly subscription period
    python tools/licence_issuer.py check AIE1-...

The private key lives in ~/.aieditor-issuer/licence_signing.key (override with --key or
$AIEDITOR_ISSUER_KEY). Back it up somewhere safe: without it no new keys can be made, and
anyone who has it can make keys.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import secrets
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "engine" / "src"))

from aieditor_engine import licence as L  # noqa: E402

ENGINE_FILE = ROOT / "engine" / "src" / "aieditor_engine" / "licence.py"


def default_key_path() -> Path:
    return Path(os.environ.get("AIEDITOR_ISSUER_KEY") or Path.home() / ".aieditor-issuer" / "licence_signing.key")


def load_seed(path: Path) -> bytes:
    seed = bytes.fromhex(path.read_text(encoding="ascii").strip())
    if len(seed) != 32:
        raise SystemExit(f"{path} is not an Ed25519 seed")
    return seed


def cmd_keygen(args: argparse.Namespace) -> int:
    path = Path(args.key or default_key_path())
    if path.exists() and not args.force:
        raise SystemExit(f"{path} already exists; keys made with it would stop working if replaced. Use --force to replace.")
    seed = secrets.token_bytes(32)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(seed.hex() + "\n", encoding="ascii")
    try:
        os.chmod(path, 0o600)
    except OSError:
        pass
    pub = L.public_key(seed).hex()
    if not args.no_update:
        src = ENGINE_FILE.read_text(encoding="utf-8")
        src, n = re.subn(r'^PUBLIC_KEY = "[0-9a-f]{64}".*$', f'PUBLIC_KEY = "{pub}"', src, count=1, flags=re.M)
        if n != 1:
            raise SystemExit("could not find PUBLIC_KEY in licence.py")
        ENGINE_FILE.write_text(src, encoding="utf-8")
    print(json.dumps({"private_key_file": str(path), "public_key": pub, "engine_updated": not args.no_update}, indent=1))
    print("Back up the private key file now. Do not commit it.", file=sys.stderr)
    return 0


def cmd_issue(args: argparse.Namespace) -> int:
    seed = load_seed(Path(args.key or default_key_path()))
    if L.public_key(seed).hex() != L.PUBLIC_KEY and not args.any_public:
        raise SystemExit("this private key does not match the engine's PUBLIC_KEY; keys would be rejected")
    now = int(time.time())
    payload = {"v": 1, "id": args.id or "L-" + secrets.token_hex(6), "plan": args.plan, "to": args.to or "",
               "seats": args.seats, "iat": now, "exp": now + int(args.days * 86400) if args.days else None}
    key = L.make_key(seed, payload)
    L.parse(key, L.public_key(seed).hex())  # self-check
    if args.json:
        print(json.dumps({"key": key, **payload}, indent=1))
    else:
        print(key)
    return 0


def cmd_check(args: argparse.Namespace) -> int:
    try:
        lic = L.parse(args.licence_key)
    except L.LicenceError as exc:
        print(json.dumps({"valid": False, "error": str(exc)}))
        return 1
    print(json.dumps({"valid": True, "expired": lic.expired(), **lic.__dict__}, indent=1))
    return 0


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--key", help="private key file")
    sub = p.add_subparsers(dest="cmd", required=True)
    k = sub.add_parser("keygen")
    k.add_argument("--force", action="store_true")
    k.add_argument("--no-update", action="store_true", help="do not write the public key into the engine")
    i = sub.add_parser("issue")
    i.add_argument("--plan", choices=L.PLANS, default="pro")
    i.add_argument("--to", help="licensee name shown in the app")
    i.add_argument("--seats", type=int, default=1)
    i.add_argument("--days", type=float, help="valid for this many days (default: no expiry)")
    i.add_argument("--id", help="licence id (default: random), e.g. the order number")
    i.add_argument("--json", action="store_true")
    i.add_argument("--any-public", action="store_true", help=argparse.SUPPRESS)
    c = sub.add_parser("check")
    c.add_argument("licence_key")
    args = p.parse_args(argv)
    return {"keygen": cmd_keygen, "issue": cmd_issue, "check": cmd_check}[args.cmd](args)


if __name__ == "__main__":
    raise SystemExit(main())
