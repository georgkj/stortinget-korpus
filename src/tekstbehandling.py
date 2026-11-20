#!/usr/bin/env python
# coding: utf-8
# %%

# %%
import json
import re
from pathlib import Path
import pandas as pd

def rens_html(tekst: str) -> str:
    """Fjerner HTML-tagger og rydder opp i doble mellomrom."""
    if not tekst:
        return ""
    uten_tagger = re.sub(r"<[^<>]*>", " ", tekst)
    return re.sub(r"\s+", " ", uten_tagger).strip()

def bygg_korpus_fra_raw(raw_dir: Path = Path("data/raw")) -> pd.DataFrame:
    """Leser alle JSON-filer i data/raw og bygger en renset DataFrame."""
    rader = []

    for filsti in raw_dir.rglob("*.json"):
        # Hopp over eventuelle skjulte Jupyter-filer eller tomme filer
        if ".ipynb_checkpoints" in filsti.parts or filsti.stat().st_size == 0:
            continue

        try:
            with open(filsti, "r", encoding="utf-8") as f:
                data = json.load(f)
        except json.JSONDecodeError:
            print(f"Hoppet over ugyldig JSON-fil: {filsti}")
            continue

        fra = data.get("sporsmal_fra") or {}
        parti = fra.get("parti") or {}

        rader.append({
            "id": data.get("id"),
            "sesjon": data.get("sesjon_id"),
            "dato": data.get("datert_dato"),
            "tittel": data.get("tittel"),
            "representant": f"{fra.get('fornavn', '')} {fra.get('etternavn', '')}".strip(),
            "parti": parti.get("navn", "Ukjent"),
            "minister": data.get("besvart_av_minister_tittel", "Ikke besvart"),
            "sporsmal_tekst": rens_html(data.get("sporsmal_tekst", "")),
            "svar_tekst": rens_html(data.get("svar_tekst", "")),
        })

    df = pd.DataFrame(rader)
    if not df.empty and "dato" in df.columns:
        df["dato"] = pd.to_datetime(df["dato"], errors="coerce")

    return df


# %%




