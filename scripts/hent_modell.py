#!/usr/bin/env python3
"""Hent modellfilene til ``web/model/``.

Modellvektene er avledet av FlyWire-connectomet og er lisensiert under CC BY-NC 4.0, ikke under
MIT slik koden er. Derfor ligger de ikke i kodelageret, og dette skriptet henter dem i stedet.

    python scripts/hent_modell.py                    # til web/model/
    python scripts/hent_modell.py --ut /tmp/modell   # et annet sted
    python scripts/hent_modell.py --bare-json        # hopp over de store bytene

Etterpå virker spillet helt uten internett: ``finnesLokalModell()`` i web/app.js ser at
``brain.json`` ligger på plass og hopper over nedlastingen.

Brukes av CI og av den som vil bygge en Windows-utgave der modellen følger med.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import urllib.error
import urllib.request
from pathlib import Path

KILDE = "https://huggingface.co/cesp99/fly-chess/resolve/main/web"
FILER = ("brain.json", "brain.flyb.gz", "brain.flyb")
STANDARD_UT = Path(__file__).resolve().parent.parent / "web" / "model"


def hent(url: str, fil: Path) -> None:
    """Last ned ``url`` til ``fil`` med framdrift på samme linje."""
    with urllib.request.urlopen(url, timeout=120) as svar:
        total = int(svar.headers.get("content-length") or 0)
        tmp = fil.with_suffix(fil.suffix + ".del")
        mottatt = 0
        with tmp.open("wb") as ut:
            while True:
                bit = svar.read(1 << 20)
                if not bit:
                    break
                ut.write(bit)
                mottatt += len(bit)
                if total:
                    prosent = 100 * mottatt // total
                    print(f"\r  {fil.name}: {prosent:3d} %  ({mottatt / 1e6:5.1f} / {total / 1e6:.1f} MB)", end="")
                else:
                    print(f"\r  {fil.name}: {mottatt / 1e6:.1f} MB", end="")
        print()
        tmp.replace(fil)


def sjekksum(fil: Path) -> str:
    h = hashlib.sha256()
    with fil.open("rb") as f:
        for bit in iter(lambda: f.read(1 << 20), b""):
            h.update(bit)
    return h.hexdigest()


def main() -> int:
    p = argparse.ArgumentParser(description="Hent Fluesjakk-modellen fra Hugging Face.")
    p.add_argument("--ut", type=Path, default=STANDARD_UT, help=f"målmappe (standard: {STANDARD_UT})")
    p.add_argument("--bare-json", action="store_true", help="hent bare brain.json (headeren)")
    p.add_argument("--tving", action="store_true", help="hent på nytt selv om filene finnes")
    args = p.parse_args()

    args.ut.mkdir(parents=True, exist_ok=True)
    filer = ("brain.json",) if args.bare_json else FILER

    for navn in filer:
        fil = args.ut / navn
        if fil.exists() and not args.tving:
            print(f"  {navn}: finnes allerede ({fil.stat().st_size / 1e6:.1f} MB)")
            continue
        try:
            hent(f"{KILDE}/{navn}", fil)
        except (urllib.error.URLError, TimeoutError) as feil:
            print(f"\nKunne ikke hente {navn}: {feil}", file=sys.stderr)
            print("Sjekk internettforbindelsen og prøv igjen.", file=sys.stderr)
            return 1

    header = json.loads((args.ut / "brain.json").read_text(encoding="utf-8"))
    blob = args.ut / "brain.flyb"
    if blob.exists() and header.get("blob_sha256"):
        faktisk = sjekksum(blob)
        if faktisk != header["blob_sha256"]:
            print(f"\nSjekksummen stemmer ikke:\n  fant     {faktisk}\n  forventet {header['blob_sha256']}",
                  file=sys.stderr)
            return 1
        print("  sjekksum: OK")

    print(f"\nFerdig. Modellen ligger i {args.ut}")
    print(f"  {header.get('n', '?'):,} nevroner · {header.get('nnz_total', header.get('nnz', 0)):,} koblinger"
          f" · kjøring {header.get('run_name', '?')}".replace(",", " "))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
