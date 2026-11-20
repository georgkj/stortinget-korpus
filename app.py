# +
from pathlib import Path
import pandas as pd
import streamlit as st
from src.analyse import beregn_ordfrekvenser, beregn_sarpregede_ord, beregn_kollokasjoner
from wordcloud import WordCloud

st.set_page_config(page_title="Stortinget Korpus 2025–2026", layout="wide")
st.title("Skriftlige spørsmål på Stortinget (2025–2026)")

PARQUET_FIL = Path("data/processed/korpus.parquet")


@st.cache_data(ttl=10)
def last_korpus(filsti: Path) -> pd.DataFrame:
    return pd.read_parquet(filsti)


if not PARQUET_FIL.exists():
    st.error(f"Fant ikke korpusfilen på `{PARQUET_FIL}`. Kjør notebooken for å bygge korpuset først.")
    st.stop()

df = last_korpus(PARQUET_FIL)

# Sidepanel med oppdateringsknapp og nøkkeltall
st.sidebar.metric("Antall spørsmål i korpuset", len(df))
if st.sidebar.button("Oppdater data nå"):
    st.cache_data.clear()
    st.rerun()
    
st.sidebar.markdown("---")
st.sidebar.caption(
    "**Datakilde:** [data.stortinget.no](https://data.stortinget.no) "
    "(Skriftlige spørsmål, sesjonen 2025–2026)"
)

tab1, tab2, tab3, tab4 = st.tabs([
    "Dokumentleser",
    "Ordfrekvenser per parti",
    "Særpregede ord (Overrepresentasjon)",
    "Kollokasjoner",
])

# --- FANE 1: DOKUMENTLESER ---
with tab1:
    col_s1, col_s2, col_s3 = st.columns([2, 1, 1])
    with col_s1:
        sokeord = st.text_input("Fritekstsøk i tittel, spørsmål og svar", "")
    with col_s2:
        parti_filter = st.selectbox(
            "Filtrer på parti",
            ["Alle"] + sorted(df["parti"].dropna().unique().tolist()),
            key="leser_parti",
        )
    with col_s3:
        minister_filter = st.selectbox(
            "Filtrer på statsråd",
            ["Alle"] + sorted(df["minister"].dropna().unique().tolist()),
            key="leser_minister",
        )

    filtrert = df.copy()
    if parti_filter != "Alle":
        filtrert = filtrert[filtrert["parti"] == parti_filter]
    if minister_filter != "Alle":
        filtrert = filtrert[filtrert["minister"] == minister_filter]
    if sokeord.strip():
        mask = (
            filtrert["tittel"].fillna("").str.contains(sokeord, case=False)
            | filtrert["sporsmal_tekst"].fillna("").str.contains(sokeord, case=False)
            | filtrert["svar_tekst"].fillna("").str.contains(sokeord, case=False)
        )
        filtrert = filtrert[mask]

    st.caption(f"Viser {len(filtrert)} av {len(df)} dokumenter")

    if filtrert.empty:
        st.info("Ingen dokumenter matchet søket.")
    else:
        st.dataframe(
            filtrert[["dato", "parti", "representant", "minister", "tittel"]],
            use_container_width=True,
            hide_index=True,
        )

        st.divider()
        valgt_idx = st.selectbox(
            "Velg et spørsmål for å lese hele teksten",
            options=filtrert.index,
            format_func=lambda i: f"{filtrert.loc[i, 'dato']} | {filtrert.loc[i, 'representant']} ({filtrert.loc[i, 'parti']}): {filtrert.loc[i, 'tittel'][:90]}...",
        )
        rad = filtrert.loc[valgt_idx]
        st.markdown(f"### {rad['tittel']}")
        st.markdown(f"**Fra:** {rad['representant']} ({rad['parti']})  |  **Til:** {rad['minister']}  |  **Dato:** {rad['dato']}")
        st.markdown("#### Spørsmål")
        st.write(rad["sporsmal_tekst"])
        if "begrunnelse" in rad and pd.notna(rad["begrunnelse"]) and str(rad["begrunnelse"]).strip():
            st.markdown("#### Begrunnelse")
            st.write(rad["begrunnelse"])
        st.markdown("#### Svar")
        st.write(rad["svar_tekst"])

# --- FANE 2: ORDFREKVENSER PER PARTI ---
with tab2:
    st.subheader("Mest brukte ord i skriftlige spørsmål")

    col1, col2, col3 = st.columns(3)
    with col1:
        partier = ["Alle partier"] + sorted(df["parti"].dropna().unique().tolist())
        valgt_parti = st.selectbox("Velg parti", partier, key="frekvens_parti")
    with col2:
        tekst_valg = st.selectbox(
            "Hvilken tekst skal analyseres?",
            ["Kun spørsmål (tittel + spørsmål)", "Kun statsrådens svar", "Hele dokumentet"],
        )
    with col3:
        ngram_valg = st.radio("Ordtyper", ["Enkeltord", "Ordpar (bigrammer)"], horizontal=True)

    ekstra_stopp = st.text_input(
        "Ekstra ord du vil filtrere bort (kommaseparert)",
        placeholder="f.eks. fylke, kommune, tiltak",
    )

    kolonne_map = {
        "Kun spørsmål (tittel + spørsmål)": ["tittel", "sporsmal_tekst"],
        "Kun statsrådens svar": ["svar_tekst"],
        "Hele dokumentet": ["tittel", "sporsmal_tekst", "svar_tekst"],
    }
    ngram = (1, 1) if ngram_valg == "Enkeltord" else (2, 2)
    ekstra_set = {w for w in ekstra_stopp.split(",") if w.strip()}

    df_frekvens = beregn_ordfrekvenser(
        df=df,
        parti=valgt_parti,
        tekstkolonner=kolonne_map[tekst_valg],
        topp_n=50,
        ngram_range=ngram,
        ekstra_stoppord=ekstra_set,
    )

    if df_frekvens.empty:
        st.info("Ingen treff for dette utvalget.")
    else:
        col_sky, col_tabell = st.columns([3, 2])
        with col_sky:
            frekvens_dict = dict(zip(df_frekvens["ord"], df_frekvens["antall"]))
            ordsky = WordCloud(
                width=800,
                height=450,
                background_color="white",
                colormap="viridis",
                prefer_horizontal=0.9,
            ).generate_from_frequencies(frekvens_dict)
            st.image(ordsky.to_array(), use_container_width=True)
        with col_tabell:
            st.dataframe(df_frekvens, use_container_width=True, hide_index=True)
            
# --- FANE 3: SÆRPREGEDE ORD PER PARTI ---
with tab3:
    st.subheader("Hvilke ord skiller hvert parti fra resten av Stortinget?")
    st.caption(
        "Viser ord som brukes oftere av det valgte partiet sammenlignet med gjennomsnittet av de andre partiene "
        "(f.eks. 4.5x = 4,5 ganger høyere relativ ordbruk)."
    )

    col_p1, col_p2, col_p3 = st.columns(3)
    with col_p1:
        partier_uten_alle = sorted(df["parti"].dropna().unique().tolist())
        valgt_parti_profil = st.selectbox("Velg parti", partier_uten_alle, key="profil_parti")
    with col_p2:
        tekst_valg_profil = st.selectbox(
            "Tekstgrunnlag",
            ["Kun spørsmål (tittel + spørsmål)", "Hele dokumentet"],
            key="profil_tekst",
        )
    with col_p3:
        min_antall = st.slider(
            "Minst antall ganger brukt av partiet",
            min_value=2,
            max_value=15,
            value=3,
            help="Høyere verdi filtrerer bort sjeldne enkeltord og løfter frem gjennomgående hjertesaker.",
        )

    kolonne_map_profil = {
        "Kun spørsmål (tittel + spørsmål)": ["tittel", "sporsmal_tekst"],
        "Hele dokumentet": ["tittel", "sporsmal_tekst", "svar_tekst"],
    }

    df_profil = beregn_sarpregede_ord(
        df=df,
        parti=valgt_parti_profil,
        tekstkolonner=kolonne_map_profil[tekst_valg_profil],
        topp_n=25,
        min_forekomster=min_antall,
    )

    if df_profil.empty:
        st.info("For få dokumenter for dette partiet med valgt terskelverdi. Prøv å senke slideren til 2.")
    else:
        col_graf3, col_tabell3 = st.columns([3, 2])
        with col_graf3:
            st.bar_chart(df_profil.set_index("ord")["overrepresentasjon_x"])
        with col_tabell3:
            st.dataframe(df_profil, use_container_width=True, hide_index=True)
            
# --- FANE 4: KOLLOKASJONER ---
with tab4:
    st.subheader("Hvilke ord brukes sammen med et gitt søkeord?")
    st.caption(
        "Finner ord som opptrer rett før eller rett etter søkeordet innenfor et valgt ordvindu."
    )

    col_k1, col_k2, col_k3, col_k4 = st.columns([2, 2, 2, 1])
    with col_k1:
        kollok_ord = st.text_input("Søkeord (eksakt ord)", value="norge", key="kollok_ord")
    with col_k2:
        partier_kollok = ["Alle partier"] + sorted(df["parti"].dropna().unique().tolist())
        valgt_parti_kollok = st.selectbox("Velg parti", partier_kollok, key="kollok_parti")
    with col_k3:
        tekst_valg_kollok = st.selectbox(
            "Tekstgrunnlag",
            ["Kun spørsmål (tittel + spørsmål)", "Kun statsrådens svar", "Hele dokumentet"],
            key="kollok_tekst",
        )
    with col_k4:
        vindu_str = st.number_input("Ordvindu (±)", min_value=1, max_value=15, value=5)

    kolonne_map_kollok = {
        "Kun spørsmål (tittel + spørsmål)": ["tittel", "sporsmal_tekst"],
        "Kun statsrådens svar": ["svar_tekst"],
        "Hele dokumentet": ["tittel", "sporsmal_tekst", "svar_tekst"],
    }

    df_kollok, kwic_eksempler = beregn_kollokasjoner(
        df=df,
        sokeord=kollok_ord,
        parti=valgt_parti_kollok,
        tekstkolonner=kolonne_map_kollok[tekst_valg_kollok],
        vindu=int(vindu_str),
        topp_n=25,
    )

    if df_kollok.empty:
        st.info(f"Fant ingen kollokasjoner for «{kollok_ord}» med dette filteret.")
    else:
        st.dataframe(df_kollok, use_container_width=True, hide_index=True)

        if kwic_eksempler:
            with st.expander(f"Se tekstutdrag der «{kollok_ord}» brukes (KWIC)"):
                for eks in kwic_eksempler:
                    st.markdown(f"- {eks}")
# -


