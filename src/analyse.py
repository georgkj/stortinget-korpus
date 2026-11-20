import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from collections import Counter
import re

# Vanlige norske + typiske stortings-stoppord
NORSKE_STOPPORD = {
    "og", "i", "jeg", "det", "at", "en", "et", "den", "til", "er", "som", "på",
    "de", "med", "han", "av", "ikke", "ikkje", "der", "så", "var", "meg", "seg",
    "men", "ett", "har", "om", "vi", "min", "mitt", "ha", "hadde", "hun", "nå",
    "over", "da", "ved", "fra", "du", "ut", "sin", "dem", "oss", "opp", "man",
    "kan", "hans", "hvor", "eller", "hva", "skal", "selv", "sjøl", "her", "alle",
    "vil", "bli", "ble", "blei", "blitt", "kunne", "inn", "når", "være", "kom",
    "noen", "noe", "ville", "dere", "som", "deres", "kun", "ja", "etter", "ned",
    "skulle", "denne", "for", "også", "bare", "mot", "uden", "under", "hvordan",
    "hvilke", "hvilken", "hvilket", "hvorfor", "siden", "andre", "få", "fått", "slik",
    "dette", "blir", "mange", "får", "initiativ", "disse", "mellom", "uten", "samt",
    "gjort", "dersom", "vært", "mer", "både", "legge", "samtidig", "før", "siste"
    "herunder", "frem", "enn", "gjennom", "enn", "kva"
    # Domenespesifikke stoppord for skriftlige spørsmål:
    "statsråden", "statsråd", "regjeringen", "stortinget", "spørsmål", "svar",
    "mener", "vurderer", "sikre", "tiltak", "arbeidet", "ærede", "president", "statsråden"
}

def beregn_ordfrekvenser(
    df: pd.DataFrame,
    parti: str | None = None,
    tekstkolonner: list[str] = ["tittel", "sporsmal_tekst"],
    topp_n: int = 25,
    ngram_range: tuple[int, int] = (1, 1),
    ekstra_stoppord: set[str] | None = None,
) -> pd.DataFrame:
    """Beregner de vanligste ordene (eller ordparene) for hele korpuset eller et valgt parti."""
    utvalg = df.copy()

    # Filtrer på parti hvis valgt
    if parti and parti != "Alle partier":
        utvalg = utvalg[utvalg["parti"] == parti]

    if utvalg.empty:
        return pd.DataFrame(columns=["ord", "antall", "andel_dokumenter_pst"])

    # Slå sammen valgte tekstkolonner til én streng per spørsmål
    kombinert_tekst = (
        utvalg[tekstkolonner]
        .fillna("")
        .agg(" ".join, axis=1)
    )

    stoppord = set(NORSKE_STOPPORD)
    if ekstra_stoppord:
        stoppord.update(w.lower().strip() for w in ekstra_stoppord)

    vectorizer = CountVectorizer(
        lowercase=True,
        stop_words=list(stoppord),
        token_pattern=r"(?u)\b[a-zæøåA-ZÆØÅ]{3,}\b",  # Kun ord på minst 3 bokstaver (inkl. æøå)
        ngram_range=ngram_range,
        max_features=5000,
    )

    try:
        X = vectorizer.fit_transform(kombinert_tekst)
    except ValueError:
        # Skjer hvis utvalget er for lite til å ha gyldige ord etter stoppord-filtrering
        return pd.DataFrame(columns=["ord", "antall", "andel_dokumenter_pst"])

    ord_liste = vectorizer.get_feature_names_out()
    total_antall = X.sum(axis=0).A1
    dok_frekvens = (X > 0).sum(axis=0).A1
    antall_dok = len(utvalg)

    resultat = pd.DataFrame({
        "ord": ord_liste,
        "antall": total_antall,
        "andel_dokumenter_pst": (dok_frekvens / antall_dok * 100).round(1),
    })

    return resultat.sort_values(by="antall", ascending=False).head(topp_n).reset_index(drop=True)


def beregn_sarpregede_ord(
    df: pd.DataFrame,
    parti: str,
    tekstkolonner: list[str] = ["tittel", "sporsmal_tekst"],
    topp_n: int = 25,
    min_forekomster: int = 3,
    ngram_range: tuple[int, int] = (1, 1),
    ekstra_stoppord: set[str] | None = None,
) -> pd.DataFrame:
    """Finner ord som er statistisk overrepresentert hos ett parti sammenlignet med resten."""
    if not parti or parti == "Alle partier":
        return pd.DataFrame(columns=["ord", "antall_i_partiet", "overrepresentasjon_x"])

    utvalg = df.dropna(subset=["parti"]).copy()
    er_valgt_parti = utvalg["parti"] == parti

    if er_valgt_parti.sum() == 0 or (~er_valgt_parti).sum() == 0:
        return pd.DataFrame(columns=["ord", "antall_i_partiet", "overrepresentasjon_x"])

    kombinert_tekst = utvalg[tekstkolonner].fillna("").agg(" ".join, axis=1)

    stoppord = set(NORSKE_STOPPORD)
    if ekstra_stoppord:
        stoppord.update(w.lower().strip() for w in ekstra_stoppord)

    vectorizer = CountVectorizer(
        lowercase=True,
        stop_words=list(stoppord),
        token_pattern=r"(?u)\b[a-zæøåA-ZÆØÅ]{3,}\b",
        ngram_range=ngram_range,
        min_df=min_forekomster,  # Ord må dukke opp i minst X dokumenter totalt
    )

    try:
        X = vectorizer.fit_transform(kombinert_tekst)
    except ValueError:
        return pd.DataFrame(columns=["ord", "antall_i_partiet", "overrepresentasjon_x"])

    ord_liste = vectorizer.get_feature_names_out()

    # Tell ord i valgt parti vs. alle andre partier
    X_parti = X[er_valgt_parti.values]
    X_andre = X[~er_valgt_parti.values]

    antall_parti = X_parti.sum(axis=0).A1
    antall_andre = X_andre.sum(axis=0).A1

    totalt_ord_parti = antall_parti.sum()
    totalt_ord_andre = antall_andre.sum()

    if totalt_ord_parti == 0 or totalt_ord_andre == 0:
        return pd.DataFrame(columns=["ord", "antall_i_partiet", "overrepresentasjon_x"])

    # Relativ andel per 10 000 ord (med Laplace-glatting +1 for å unngå deling på null)
    rate_parti = (antall_parti + 1) / (totalt_ord_parti + len(ord_liste))
    rate_andre = (antall_andre + 1) / (totalt_ord_andre + len(ord_liste))
    ratio = rate_parti / rate_andre

    resultat = pd.DataFrame({
        "ord": ord_liste,
        "antall_i_partiet": antall_parti,
        "antall_andre_partier": antall_andre,
        "overrepresentasjon_x": ratio.round(2),
    })

    # Krev at partiet faktisk har brukt ordet minst 'min_forekomster' ganger
    resultat = resultat[resultat["antall_i_partiet"] >= min_forekomster]

    return (
        resultat.sort_values(by="overrepresentasjon_x", ascending=False)
        .head(topp_n)
        .reset_index(drop=True)
    )


def beregn_kollokasjoner(
    df: pd.DataFrame,
    sokeord: str,
    parti: str | None = None,
    tekstkolonner: list[str] = ["tittel", "sporsmal_tekst"],
    vindu: int = 5,
    topp_n: int = 25,
    ekstra_stoppord: set[str] | None = None,
) -> tuple[pd.DataFrame, list[str]]:
    """
    Finner ord som ofte opptrer innenfor ±'vindu' ord fra 'sokeord'.
    Returnerer både en DataFrame med kollokasjoner og en liste med KWIC-eksempler (kontekst).
    """
    mal_ord = sokeord.lower().strip()
    if not mal_ord:
        return pd.DataFrame(columns=["kollokat", "antall_sammen"]), []

    utvalg = df.copy()
    if parti and parti != "Alle partier":
        utvalg = utvalg[utvalg["parti"] == parti]

    if utvalg.empty:
        return pd.DataFrame(columns=["kollokat", "antall_sammen"]), []

    stoppord = set(NORSKE_STOPPORD)
    if ekstra_stoppord:
        stoppord.update(w.lower().strip() for w in ekstra_stoppord)

    kombinert_tekst = utvalg[tekstkolonner].fillna("").agg(" ".join, axis=1)

    teller = Counter()
    eksempler = []

    for tekst in kombinert_tekst:
        # Trekk ut rene ord på minst 3 bokstaver (inkl. æøå)
        tokens = re.findall(r"\b[a-zæøåA-ZÆØÅ]{3,}\b", tekst.lower())

        for idx, token in enumerate(tokens):
            if token == mal_ord:
                start = max(0, idx - vindu)
                slutt = min(len(tokens), idx + vindu + 1)

                # Hent naboordene i vinduet (uten søkeordet selv og uten stoppord)
                naboer = [
                    tokens[j]
                    for j in range(start, slutt)
                    if j != idx and tokens[j] not in stoppord and tokens[j] != mal_ord
                ]
                teller.update(naboer)

                # Ta vare på inntil 15 tekstutdrag (KWIC) for visning
                if len(eksempler) < 15:
                    kontekst_tokens = tokens[start:slutt]
                    # Uthev søkeordet med fet skrift
                    formatert = " ".join(
                        f"**{t}**" if (start + k) == idx else t
                        for k, t in enumerate(kontekst_tokens)
                    )
                    eksempler.append(f"... {formatert} ...")

    if not teller:
        return pd.DataFrame(columns=["kollokat", "antall_sammen"]), eksempler

    resultat = pd.DataFrame(
        teller.most_common(topp_n),
        columns=["kollokat", "antall_sammen"],
    )
    return resultat, eksempler


