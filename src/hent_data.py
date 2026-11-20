#!/usr/bin/env python
# coding: utf-8
# %%

# %%
from datetime import datetime, timezone
import json
import re
import time
import xml.etree.ElementTree as ET
from pathlib import Path
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

BASE_URL = "https://data.stortinget.no/eksport"


def _lag_robust_sesjon() -> requests.Session:
    """Oppretter en HTTP-sesjon med User-Agent, gjenbruk av tilkobling og retry."""
    session = requests.Session()
    session.headers.update({
        "User-Agent": "StortingetKorpusProsjekt/1.0 (Python; akademisk tekstkorpus)"
    })
    retry_strategi = Retry(
        total=4,
        backoff_factor=5,
        status_forcelist=[429, 502, 503, 504],
        respect_retry_after_header=True,
    )
    adapter = HTTPAdapter(max_retries=retry_strategi)
    session.mount("https://", adapter)
    session.mount("http://", adapter)
    return session


def _hent_xml_tekst(rot: ET.Element, tag_navn: str) -> str:
    """Henter all tekst (inkludert under-tagger som <p>) fra en XML-tag."""
    for elem in rot.iter():
        ren_tag = elem.tag.split("}")[-1]
        if ren_tag == tag_navn:
            full_tekst = " ".join(t.strip() for t in elem.itertext() if t.strip())
            if full_tekst:
                return full_tekst
    return ""


def _konverter_ms_dato(dato_str: str | None) -> str | None:
    """Konverterer Stortingets '/Date(1790773419482+0200)/' til ISO-dato (YYYY-MM-DD)."""
    if not dato_str:
        return None
    treff = re.search(r"/Date\((-?\d+)", dato_str)
    if treff:
        ms = int(treff.group(1))
        return datetime.fromtimestamp(ms / 1000, tz=timezone.utc).strftime("%Y-%m-%d")
    return dato_str


def hent_sporsmal_til_raw(
    sesjon: str, maks_antall: int | None = None, raw_dir: Path = Path("data/raw")
) -> list[Path]:
    """Henter skriftlige spørsmål fra Stortinget og lagrer dem som JSON-filer."""
    sesjon_mappe = raw_dir / sesjon
    sesjon_mappe.mkdir(parents=True, exist_ok=True)

    http = _lag_robust_sesjon()

    # 1. Hent oversiktsliste
    oversikt_url = f"{BASE_URL}/skriftligesporsmal?sesjonid={sesjon}&format=json"
    respons = http.get(oversikt_url, timeout=20)
    respons.raise_for_status()
    sporsmal_liste = respons.json().get("sporsmal_liste", [])
    print(f"API returnerte {len(sporsmal_liste)} spørsmål i listen for sesjon {sesjon}.")

    lagrede_filer = []
    utvalg = sporsmal_liste[:maks_antall]

    # 2. Hent detaljer for hvert spørsmål via legacy_id
    for i, spm in enumerate(utvalg, start=1):
        spm_id = spm.get("legacy_id") or spm["id"]
        filsti = sesjon_mappe / f"sporsmal_{spm_id}.json"

        if not filsti.exists():
            detalj_url = f"{BASE_URL}/enkeltsporsmal?sporsmalid={spm_id}"
            try:
                detalj_resp = http.get(detalj_url, timeout=20)
                if detalj_resp.status_code == 200 and detalj_resp.content:
                    rot = ET.fromstring(detalj_resp.content)
                    fra_info = spm.get("sporsmal_fra") or {}
                    dato = _hent_xml_tekst(rot, "datert_dato") or _konverter_ms_dato(
                        spm.get("datert_dato")
                    )

                    strukturert_data = {
                        "id": spm_id,
                        "sesjon_id": sesjon,
                        "datert_dato": dato,
                        "tittel": _hent_xml_tekst(rot, "tittel") or spm.get("tittel"),
                        "sporsmal_fra": fra_info,
                        "besvart_av_minister_tittel": _hent_xml_tekst(
                            rot, "besvart_av_minister_tittel"
                        )
                        or spm.get("besvart_av_minister_tittel"),
                        "sporsmal_tekst": _hent_xml_tekst(rot, "sporsmal")
                        or _hent_xml_tekst(rot, "sporsmal_tekst")
                        or spm.get("tittel", ""),
                        "begrunnelse": _hent_xml_tekst(rot, "begrunnelse"),
                        "svar_tekst": _hent_xml_tekst(rot, "svar")
                        or _hent_xml_tekst(rot, "svar_tekst"),
                    }
                    with open(filsti, "w", encoding="utf-8") as f:
                        json.dump(strukturert_data, f, ensure_ascii=False, indent=2)
                else:
                    print(
                        f"Uventet svar for {spm_id} (status={detalj_resp.status_code})"
                    )

                time.sleep(1.2)

            except Exception as e:
                print(
                    f"Feil ved spørsmål {spm_id} ({i}/{len(utvalg)}): {type(e).__name__}: {e}"
                )
                time.sleep(60)
                continue

        if filsti.exists():
            lagrede_filer.append(filsti)

        if i % 50 == 0:
            print(
                f"Fremdrift: {i} / {len(utvalg)} behandlet (lagret på disk: {len(lagrede_filer)})..."
            )

    return lagrede_filer


# %%





# %%
