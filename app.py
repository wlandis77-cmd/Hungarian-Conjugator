# --------------------------- environment & imports ---------------------------
import os
os.environ.setdefault("STREAMLIT_SERVER_FILE_WATCHER_TYPE", "none")
os.environ.setdefault("STREAMLIT_SERVER_RUN_ON_SAVE", "false")

import json
import random
import re
from dataclasses import dataclass
from functools import lru_cache
from io import BytesIO
from pathlib import Path
from typing import Dict, Optional, Tuple

import pandas as pd
import streamlit as st
import toml

# emMorph transducer (optional but preferred)
try:
    from hfst_optimized_lookup import Transducer
except Exception:
    Transducer = None  # graceful fallback if lib not installed

# Optional GitHub integration
try:
    from github import Github  # pip install PyGithub (optional)
    _GITHUB_OK = True
except Exception:
    _GITHUB_OK = False

# Optional neural generator (disabled by default; heavyweight)
_TRANSFORMERS_OK = False
try:
    from transformers import pipeline  # pip install transformers torch sentencepiece
    _TRANSFORMERS_OK = True
except Exception:
    pass

# Optional TTS backends
_HAS_GTTS = False
try:
    from gtts import gTTS  # pip install gTTS
    _HAS_GTTS = True
except Exception:
    pass

_HAS_GOOGLE_TTS = False
try:
    from google.cloud import texttospeech  # type: ignore
    from google.oauth2 import service_account  # type: ignore
    _HAS_GOOGLE_TTS = True
except Exception:
    pass


# --------------------------- config & emMorph loader -------------------------
def _load_cfg() -> dict:
    try:
        return toml.load("config.toml")
    except Exception:
        return {}

_CFG = _load_cfg()
_FST_PATH = Path(
    _CFG.get("emmorph", {}).get(
        "hfst_path",
        _CFG.get("analyzer_path", "/home/willl/emMorph/hfst/hu.hfstol")
    )
).as_posix()

@st.cache_resource(show_spinner=False)
def load_analyzer():
    if Transducer is None:
        return None
    try:
        return Transducer(_FST_PATH)
    except Exception:
        return None

ANALYZER = load_analyzer()

def analyze(surface: str):
    """Return emMorph analyses for a surface form, or [] if unavailable."""
    if ANALYZER is None:
        return []
    try:
        return ANALYZER.lookup(surface)
    except Exception:
        return []


# --------------------------- page config & theme -----------------------------
st.set_page_config(
    page_title="Hungarian Conjugations & Declensions Trainer",
    page_icon="🇭🇺",
    layout="wide",
)

st.markdown(
    """
    <style>
    :root{
      --bg1:#f6f0ff; --bg2:#e9f7ff; --surface:#eef2ff; --surface2:#e7f7f4;
      --ink:#1f2937; --muted:#4b5563; --accent:#a7c8ff; --accent-ink:#0f172a;
      --border:#cfd8ee; --pill:#d6efff; --good:#1c8c4e; --bad:#b21b1b;
    }
    [data-testid="stAppViewContainer"]{
      background: linear-gradient(135deg,var(--bg1) 0%,var(--bg2) 100%);
    }
    [data-testid="stSidebar"]{
      background: linear-gradient(180deg,#f8eaff 0%,#e6f7ff 100%);
      border-right: 1px solid var(--border);
    }
    [data-testid="stHeader"]{ background: transparent; }
    .block-container{ padding-top: 1rem; }
    .prompt-card{
      border: 1px solid var(--border); padding: 1rem 1.25rem; border-radius: 12px;
      background: var(--surface); box-shadow: 0 1px 0 rgba(16,24,40,.03); margin-bottom: 1rem;
    }
    .pill{
      display:inline-block; font-size:.85rem; padding:.12rem .6rem; border:1px solid var(--border);
      border-radius:999px; margin-right:.35rem; background: var(--pill);
    }
    .big-title{ font-size: 1.8rem; font-weight: 700; letter-spacing: .2px; margin-bottom: .25rem; color:var(--ink); }
    .subtitle{ color: var(--muted); margin-bottom: 1rem; }
    .stButton > button{
      background: var(--accent) !important; color: var(--accent-ink) !important;
      border: 1px solid var(--border) !important; border-radius: 10px !important;
      box-shadow: 0 1px 0 rgba(16,24,40,.06) !important;
    }
    .stButton > button:hover{ filter: brightness(0.98); transform: translateY(-1px); transition: transform .08s ease; }
    input, textarea, select{
      background-color: var(--surface) !important; color: var(--ink) !important; border: 1px solid var(--border) !important;
    }
    .stTextInput > div > div > input{ background-color: var(--surface) !important; }
    .metric-card{ background: var(--surface2); border: 1px solid var(--border); border-radius: 12px; padding: .75rem 1rem; }
    .good{ color: var(--good); font-weight: 700; } .bad{ color: var(--bad); font-weight: 700; }
    .muted{ color: var(--muted); } .mono{ font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, "Liberation Mono", monospace; }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown('<div class="big-title">Hungarian Conjugations and Declensions</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">Practice accurate paradigms with a clean pastel interface.</div>', unsafe_allow_html=True)


# --------------------------- quick emMorph tester ----------------------------
with st.sidebar.expander("emMorph analyzer", expanded=False):
    st.caption(f"Analyzer path: `{_FST_PATH}`")
    st.write("Status:", "✅ Loaded" if ANALYZER else "⚠️ Not available")
    q = st.text_input("Type a word to analyze", "látok")
    if q:
        out = analyze(q)
        if not out:
            st.caption("No analysis returned. Check config.toml → [emmorph].hfst_path and that the file exists.")
        else:
            # Show first few analyses
            for a, w in out[:10]:
                st.write(a, w)


# --------------------------- data types --------------------------------------
@dataclass(frozen=True)
class VerbTask:
    lemma: str
    gloss: str
    mood: str     # "Ind" or "Cnd"
    tense: str    # "Pres", "Past", "Fut"
    definite: bool
    person: int
    number: str   # "Sing" or "Plur"
    is_ik: bool
    ud_key: str

@dataclass(frozen=True)
class NounTask:
    lemma: str
    gloss: str
    case: str     # Nom, Acc, Dat, Ine, Sup, Ade, Ill, Sub, All, Ins, Gen
    number: str   # "Sing" or "Plur"
    ud_key: str


# --------------------------- settings sidebar --------------------------------
with st.sidebar:
    st.header("Settings")

    source = st.radio(
        "Corpus source",
        ["Upload CSV", "Load from GitHub"],
        help="Provide one corpus CSV with both verbs and nouns. The app caches it for speed."
    )
    df: Optional[pd.DataFrame] = None

    if source == "Load from GitHub":
        repo_full = st.text_input("owner/repo", placeholder="yourname/yourrepo")
        path_in_repo = st.text_input("path in repo", value="data/hungarian_corpus.csv")
        ref = st.text_input("branch or tag", value="main")
        st.caption("Add GITHUB_TOKEN to Streamlit secrets before loading.")

        def load_from_github() -> Optional[pd.DataFrame]:
            if not _GITHUB_OK:
                st.error("PyGithub is not installed. Run: pip install PyGithub")
                return None
            token = st.secrets.get("GITHUB_TOKEN", None)
            if not token:
                st.error("GITHUB_TOKEN is missing from Streamlit secrets.")
                return None
            try:
                gh = Github(token)
                repo = gh.get_repo(repo_full)
                f = repo.get_contents(path_in_repo, ref=ref)
                content = f.decoded_content
                return pd.read_csv(BytesIO(content))
            except Exception as e:
                st.error(f"GitHub load failed: {e}")
                return None

        if st.button("Load CSV from GitHub"):
            df = load_from_github()
    else:
        uploaded = st.file_uploader("Upload corpus CSV", type=["csv"])
        if uploaded is not None:
            try:
                df = pd.read_csv(uploaded)
            except Exception as e:
                st.error(f"CSV parse failed: {e}")

    st.divider()

    st.subheader("Practice scope")
    want_verbs = st.checkbox("Verbs", value=True)
    want_nouns = st.checkbox("Nouns", value=True)

    st.caption("Verb modes")
    VERB_MODE_OPTIONS = [
        "Present Indefinite", "Present Definite",
        "Past Indefinite", "Past Definite",
        "Conditional Present Indefinite", "Conditional Present Definite",
        "Future Indefinite", "Future Definite",
    ]
    verb_modes = st.multiselect(
        "Select verb modes",
        options=VERB_MODE_OPTIONS,
        default=VERB_MODE_OPTIONS,
        help="Pick which finite paradigms to practice."
    )

    st.caption("Noun modes")
    NOUN_CASE_OPTIONS = [
        "Nominative", "Accusative", "Dative",
        "Inessive", "Superessive", "Adessive",
        "Illative", "Sublative", "Allative",
        "Instrumental", "Genitive"
    ]
    noun_modes = st.multiselect(
        "Cases",
        options=NOUN_CASE_OPTIONS,
        default=[
            "Accusative", "Dative", "Inessive", "Superessive", "Adessive",
            "Illative", "Sublative", "Allative", "Instrumental", "Genitive"
        ],
        help="Practice the most common cases. Genitive here uses the -é possessive form."
    )
    noun_numbers = st.multiselect(
        "Noun number",
        options=["Singular", "Plural"],
        default=["Singular", "Plural"],
        help="Practice singular and plural case forms."
    )

    st.divider()

    advanced = st.expander("Advanced accuracy")
    with advanced:
        prefer_ml = st.selectbox(
            "Inflection strategy",
            ["CSV overrides first, then ML generator, then rules", "CSV overrides only", "CSV overrides then rules only"],
            help="For maximum accuracy choose the ML generator path. It uses UD-style features."
        )
        # emMorph is always used to accept answers where possible
        ignore_accents = st.checkbox("Accept answers that ignore accents", value=True)
        show_hu_pronouns = st.checkbox("Show Hungarian pronouns for verb prompts", value=True)
        allow_reveal = st.checkbox("Allow Reveal Answer", value=True)

    st.divider()

    tts_expander = st.expander("Pronunciation (AI TTS)")
    with tts_expander:
        tts_provider = st.selectbox("TTS provider", ["Off", "gTTS (local, free)", "Google Cloud TTS"], index=0)
        tts_rate = st.slider("Speaking rate", 0.6, 1.4, 1.0, 0.05)
        auto_say_answer = st.checkbox("Auto speak correct answer on Reveal or when correct", value=True)
        st.caption("For Google Cloud, add a service account JSON to Streamlit secrets as GOOGLE_TTS_SERVICE_ACCOUNT_JSON. Hungarian: hu-HU.")
        if st.button("Test TTS with “Szia!”"):
            audio = None
            try:
                if tts_provider.startswith("gTTS"):
                    if not _HAS_GTTS:
                        st.error("gTTS is not installed. Run: pip install gTTS")
                    else:
                        g = gTTS("Szia!", lang="hu")
                        buf = BytesIO(); g.write_to_fp(buf); buf.seek(0)
                        audio = buf.read()
                elif tts_provider.startswith("Google"):
                    if not _HAS_GOOGLE_TTS:
                        st.error("google-cloud-texttospeech is not installed. Run: pip install google-cloud-texttospeech google-auth")
                    else:
                        sa = st.secrets.get("GOOGLE_TTS_SERVICE_ACCOUNT_JSON", None)
                        creds = service_account.Credentials.from_service_account_info(sa) if isinstance(sa, dict) else None
                        client = texttospeech.TextToSpeechClient(credentials=creds) if creds else texttospeech.TextToSpeechClient()
                        inp = texttospeech.SynthesisInput(text="Szia!")
                        voice = texttospeech.VoiceSelectionParams(language_code="hu-HU")
                        cfg = texttospeech.AudioConfig(audio_encoding=texttospeech.AudioEncoding.MP3, speaking_rate=float(tts_rate))
                        resp = client.synthesize_speech(input=inp, voice=voice, audio_config=cfg)
                        audio = resp.audio_content
            except Exception as e:
                st.error(f"TTS test failed: {e}")
            if audio:
                st.audio(audio, format="audio/mp3")


# --------------------------- csv template ------------------------------------
CSV_TEMPLATE = """
pos,lemma,english,is_ik,forms
VERB,kér,to ask,False,"{""VERB VerbForm=Fin|Mood=Ind|Tense=Pres|Person=1|Number=Sing|Definite=Ind"": ""kérek"", ""VERB VerbForm=Fin|Mood=Ind|Tense=Past|Person=3|Number=Sing|Definite=Def"": ""kérte""}"
VERB,dolgozik,to work,True,"{""VERB VerbForm=Fin|Mood=Ind|Tense=Pres|Person=1|Number=Sing|Definite=Ind"": ""dolgozom"", ""VERB VerbForm=Fin|Mood=Ind|Tense=Pres|Person=3|Number=Sing|Definite=Ind"": ""dolgozik""}"
NOUN,bolt,shop,,"{""NOUN Case=Ine|Number=Sing"": ""boltban"", ""NOUN Case=Ine|Number=Plur"": ""boltokban"", ""NOUN Case=Gen|Number=Plur"": ""boltoké""}"
""".strip()
with st.sidebar:
    st.download_button("Download CSV template", data=CSV_TEMPLATE, file_name="hungarian_corpus_template.csv", mime="text/csv")


# --------------------------- core utils --------------------------------------
BACK_VOWELS = set("aáoóuú")
FRONT_UNR = set("eéií")
FRONT_R = set("öőüű")
ALL_VOWELS = BACK_VOWELS | FRONT_UNR | FRONT_R

def has_back(s: str) -> bool:
    return any(ch in BACK_VOWELS for ch in s)

def has_front_rounded(s: str) -> bool:
    return any(ch in FRONT_R for ch in s)

def last_vowel(s: str) -> Optional[str]:
    last = None
    for ch in s.lower():
        if ch in ALL_VOWELS:
            last = ch
    return last

def harmony_set(s: str) -> str:
    if has_back(s):
        return "back"
    if has_front_rounded(s):
        return "front_r"
    return "front_unr"

ACCENT_STRIP = str.maketrans({
    "á":"a","é":"e","í":"i","ó":"o","ö":"o","ő":"o","ú":"u","ü":"u","ű":"u",
    "Á":"a","É":"e","Í":"i","Ó":"o","Ö":"o","Ő":"o","Ú":"u","Ü":"u","Ű":"u",
})
def normalize_answer(s: str, strip_accents: bool) -> str:
    s = s.strip()
    if strip_accents:
        s = s.translate(ACCENT_STRIP)
    return s.lower()


# --------------------------- noun engine -------------------------------------
def pluralize(noun: str) -> str:
    if noun.endswith("a"): return noun[:-1] + "á" + "k"
    if noun.endswith("e"): return noun[:-1] + "é" + "k"
    if noun[-1].lower() in ALL_VOWELS: return noun + "k"
    h = harmony_set(noun)
    return noun + ("ok" if h == "back" else "ök" if h == "front_r" else "ek")

class HuNoun:
    @staticmethod
    def nominative(noun: str, number: str) -> str:
        return noun if number == "Sing" else pluralize(noun)
    @staticmethod
    def dative(noun: str, number: str) -> str:
        base = noun if number == "Sing" else pluralize(noun)
        return base + ("nak" if harmony_set(base) == "back" else "nek")
    @staticmethod
    def inessive(noun: str, number: str) -> str:
        base = noun if number == "Sing" else pluralize(noun)
        return base + ("ban" if harmony_set(base) == "back" else "ben")
    @staticmethod
    def superessive(noun: str, number: str) -> str:
        base = noun if number == "Sing" else pluralize(noun)
        if base[-1].lower() in ALL_VOWELS and number == "Sing":
            return base + "n"
        h = harmony_set(base)
        return base + ("on" if h == "back" else "ön" if h == "front_r" else "en")
    @staticmethod
    def adessive(noun: str, number: str) -> str:
        base = noun if number == "Sing" else pluralize(noun)
        return base + ("nál" if harmony_set(base) == "back" else "nél")
    @staticmethod
    def illative(noun: str, number: str) -> str:
        base = noun if number == "Sing" else pluralize(noun)
        return base + ("ba" if harmony_set(base) == "back" else "be")
    @staticmethod
    def sublative(noun: str, number: str) -> str:
        base = noun if number == "Sing" else pluralize(noun)
        return base + ("ra" if harmony_set(base) == "back" else "re")
    @staticmethod
    def allative(noun: str, number: str) -> str:
        base = noun if number == "Sing" else pluralize(noun)
        h = harmony_set(base)
        return base + ("hoz" if h == "back" else "höz" if h == "front_r" else "hez")
    @staticmethod
    def instrumental(noun: str, number: str) -> str:
        base = noun if number == "Sing" else pluralize(noun)
        if base[-1].lower() in ALL_VOWELS:
            return base + ("val" if harmony_set(base) == "back" else "vel")
        ending = "al" if harmony_set(base) == "back" else "el"
        if base.endswith(("sz","zs","cs","gy","ny","ty","ly")):
            return base + base[-2:] + ending
        last_char = base[-1]
        return base + last_char + ending
    @staticmethod
    def genitive(noun: str, number: str) -> str:
        if number == "Sing":
            if noun.endswith("a"): return noun[:-1] + "á" + "é"
            if noun.endswith("e"): return noun[:-1] + "é" + "é"
            return noun + "é"
        base = pluralize(noun)
        return base + "é"
    @staticmethod
    def accusative(noun: str, number: str) -> str:
        base = noun if number == "Sing" else pluralize(noun)
        if base.endswith("a"): return base[:-1] + "á" + "t"
        if base.endswith("e"): return base[:-1] + "é" + "t"
        if base[-1].lower() in ALL_VOWELS: return base + "t"
        last = base[-1].lower()
        bigram = base[-2:].lower() if len(base) >= 2 else ""
        sibilant_like = last in {"s","z","c"} or bigram in {"sz","zs","cs","dz","dzs"}
        dental_like = last in {"t","d"}
        plural_k = last == "k"
        use_link = dental_like or sibilant_like or plural_k
        lv = "a" if last_vowel(base) in {"a","á"} else "e" if last_vowel(base) in {"e","é"} else (
            "o" if harmony_set(base) == "back" else "ö" if harmony_set(base) == "front_r" else "e"
        )
        return base + (lv + "t" if use_link else "t")


# --------------------------- verb engine -------------------------------------
class HuVerb:
    @staticmethod
    def is_ik(lemma: str, csv_flag: Optional[bool]) -> bool:
        if csv_flag is True: return True
        if csv_flag is False: return False
        return lemma.endswith("ik")

    @staticmethod
    def pres_indef(lemma: str, person: int, number: str, is_ik: bool) -> str:
        h = harmony_set(lemma)
        v_ok = {"back": "ok", "front_unr": "ek", "front_r": "ök"}[h]
        v_1pl = {"back": "unk", "front_unr": "ünk", "front_r": "ünk"}[h]
        v_2pl = {"back": "tok", "front_unr": "tek", "front_r": "tök"}[h]
        v_3pl = {"back": "nak", "front_unr": "nek", "front_r": "nek"}[h]

        if number == "Sing" and person == 1:
            if is_ik:
                v = {"back": "om", "front_unr": "em", "front_r": "öm"}[h]
                return lemma.removesuffix("ik") + v if lemma.endswith("ik") else lemma + v
            return lemma + v_ok
        if number == "Sing" and person == 2:
            if re.search(r"(s|z|sz|zs)$", lemma):
                link = {"back": "ol", "front_unr": "el", "front_r": "öl"}[h]
                return lemma + link
            return lemma + "sz"
        if number == "Sing" and person == 3:
            if is_ik:
                base = lemma.removesuffix("ik") if lemma.endswith("ik") else lemma
                return base + "ik"
            return lemma
        if number == "Plur" and person == 1: return lemma + v_1pl
        if number == "Plur" and person == 2: return lemma + v_2pl
        if number == "Plur" and person == 3: return lemma + v_3pl
        return lemma

    @staticmethod
    def pres_def(lemma: str, person: int, number: str) -> str:
        h = harmony_set(lemma)
        v_1sg = {"back": "om", "front_unr": "em", "front_r": "öm"}[h]
        v_2sg = {"back": "od", "front_unr": "ed", "front_r": "öd"}[h]
        v_1pl = {"back": "juk", "front_unr": "jük", "front_r": "jük"}[h]
        v_2pl_cons = {"back": "játok", "front_unr": "itek", "front_r": "itek"}[h]
        v_2pl_vow  = {"back": "játok", "front_unr": "jétek", "front_r": "jétek"}[h]
        v_3pl_default = {"back": "ják", "front_unr": "ik", "front_r": "ik"}[h]
        ends_vowel = lemma[-1].lower() in ALL_VOWELS

        if number == "Sing" and person == 1: return lemma + v_1sg
        if number == "Sing" and person == 2: return lemma + v_2sg
        if number == "Sing" and person == 3:
            if re.search(r"(z)$", lemma): return lemma + "i"
            if re.search(r"(s|sz|zs)$", lemma):
                if lemma.endswith("sz"): base = lemma[:-2] + "ssz"
                elif lemma.endswith("zs"): base = lemma[:-2] + "zzs"
                elif lemma.endswith("s"): base = lemma[:-1] + "ss"
                else: base = lemma
                return base + ("a" if h == "back" else "e")
            return lemma + ("ja" if h == "back" else "je")
        if number == "Plur" and person == 1: return lemma + v_1pl
        if number == "Plur" and person == 2: return lemma + (v_2pl_vow if ends_vowel else v_2pl_cons)
        if number == "Plur" and person == 3:
            if re.search(r"(z|sz|zs)$", lemma): return lemma + "ik"
            return lemma + v_3pl_default
        return lemma

    @staticmethod
    def past_indef(lemma: str, person: int, number: str) -> str:
        h = harmony_set(lemma)
        v_a = "a" if h == "back" else "e"
        v_1pl = "unk" if h == "back" else "ünk"
        if number == "Sing" and person == 1: return lemma + "t" + v_a + "m"
        if number == "Sing" and person == 2: return lemma + "t" + ("ál" if h == "back" else "él")
        if number == "Sing" and person == 3: return lemma + "t"
        if number == "Plur" and person == 1: return lemma + "t" + v_1pl
        if number == "Plur" and person == 2: return lemma + "t" + ("atok" if h == "back" else "etek")
        if number == "Plur" and person == 3: return lemma + "t" + ("ak" if h == "back" else "ek")
        return lemma + "t"

    @staticmethod
    def past_def(lemma: str, person: int, number: str) -> str:
        h = harmony_set(lemma)
        v_a = "a" if h == "back" else "e"
        v_á = "á" if h == "back" else "é"
        if number == "Sing" and person == 1: return lemma + "t" + v_a + "m"
        if number == "Sing" and person == 2: return lemma + "t" + v_a + "d"
        if number == "Sing" and person == 3: return lemma + "t" + v_a
        if number == "Plur" and person == 1: return lemma + "t" + ("uk" if h == "back" else "ük")
        if number == "Plur" and person == 2: return lemma + "t" + v_á + ("tok" if h == "back" else "tek")
        if number == "Plur" and person == 3: return lemma + "t" + v_á + "k"
        return lemma + "t"

    @staticmethod
    def cond_indef(lemma: str, person: int, number: str) -> str:
        h = harmony_set(lemma)
        if number == "Sing" and person == 1: return lemma + "nék"
        if number == "Sing" and person == 2: return lemma + ("nál" if h == "back" else "nél")
        if number == "Sing" and person == 3: return lemma + ("na" if h == "back" else "ne")
        if number == "Plur" and person == 1: return lemma + ("nánk" if h == "back" else "nénk")
        if number == "Plur" and person == 2: return lemma + ("nátok" if h == "back" else "nétek")
        if number == "Plur" and person == 3: return lemma + ("nának" if h == "back" else "nének")
        return lemma + "nék"

    @staticmethod
    def cond_def(lemma: str, person: int, number: str) -> str:
        h = harmony_set(lemma)
        if number == "Sing" and person == 1: return lemma + ("nám" if h == "back" else "ném")
        if number == "Sing" and person == 2: return lemma + ("nád" if h == "back" else "néd")
        if number == "Sing" and person == 3: return lemma + ("ná" if h == "back" else "né")
        if number == "Plur" and person == 1: return lemma + ("nánk" if h == "back" else "nénk")
        if number == "Plur" and person == 2: return lemma + ("nátok" if h == "back" else "nétek")
        if number == "Plur" and person == 3: return lemma + ("nák" if h == "back" else "nék")
        return lemma + ("nám" if h == "back" else "ném")

    @staticmethod
    def infinitive(lemma: str) -> str:
        base = lemma[:-2] if lemma.endswith("ik") else lemma
        return base + "ni"

    @staticmethod
    def fog_indef(person: int, number: str) -> str:
        return {
            ("Sing",1):"fogok",("Sing",2):"fogsz",("Sing",3):"fog",
            ("Plur",1):"fogunk",("Plur",2):"fogtok",("Plur",3):"fognak",
        }[(number,person)]

    @staticmethod
    def fog_def(person: int, number: str) -> str:
        return {
            ("Sing",1):"fogom",("Sing",2):"fogod",("Sing",3):"fogja",
            ("Plur",1):"fogjuk",("Plur",2):"fogjátok",("Plur",3):"fogják",
        }[(number,person)]

    @staticmethod
    def future_form(lemma: str, definite: bool, person: int, number: str) -> str:
        aux = HuVerb.fog_def(person, number) if definite else HuVerb.fog_indef(person, number)
        return f"{aux} {HuVerb.infinitive(lemma)}"


# --------------------------- NYTK generator (optional) -----------------------
@st.cache_resource(show_spinner=False)
def get_nytk_generator():
    if not _TRANSFORMERS_OK:
        return None
    try:
        return pipeline(task="text2text-generation", model="NYTK/morphological-generator-ud-mt5-hungarian")
    except Exception:
        return None

def nyt_generate(lemma: str, ud_key: str) -> Optional[str]:
    gen = get_nytk_generator()
    if not gen:
        return None
    prompt = f"morph: {lemma} {ud_key}"
    try:
        out = gen(prompt, max_new_tokens=16, num_return_sequences=1)[0]["generated_text"]
        return out.strip()
    except Exception:
        return None


# --------------------------- corpus utils ------------------------------------
REQUIRED_COLS = {"pos", "lemma", "english"}
OPTIONAL_COLS = {"is_ik", "forms"}

def validate_corpus(df: pd.DataFrame) -> Tuple[bool, str]:
    missing = REQUIRED_COLS - set(df.columns)
    if missing:
        return False, f"Missing required columns: {', '.join(missing)}"
    return True, "ok"

@lru_cache(maxsize=4096)
def lookup_override(forms_json: str | None, ud_key: str) -> Optional[str]:
    if not forms_json or (isinstance(forms_json, float) and pd.isna(forms_json)):
        return None
    try:
        data = json.loads(forms_json)
        return data.get(ud_key) or None
    except Exception:
        return None

def get_is_ik_flag(row) -> Optional[bool]:
    try:
        val = row.get("is_ik", None)
        if pd.isna(val): return None
        if isinstance(val, bool): return val
        s = str(val).strip().lower()
        if s in {"true","1","yes"}: return True
        if s in {"false","0","no"}: return False
        return None
    except Exception:
        return None


# --------------------------- UD helpers --------------------------------------
PRONOUNS_HU = {
    ("Sing", 1): "én", ("Sing", 2): "te", ("Sing", 3): "ő",
    ("Plur", 1): "mi", ("Plur", 2): "ti", ("Plur", 3): "ők",
}

CASE_TO_UD = {
    "Nominative":"Nom","Accusative":"Acc","Dative":"Dat","Inessive":"Ine","Superessive":"Sup",
    "Adessive":"Ade","Illative":"Ill","Sublative":"Sub","Allative":"All","Instrumental":"Ins","Genitive":"Gen",
}

def parse_verb_mode(mode: str) -> Tuple[str, str, bool]:
    if mode.startswith("Present"): mood, tense, definite = "Ind", "Pres", "Definite" in mode
    elif mode.startswith("Past"):   mood, tense, definite = "Ind", "Past", "Definite" in mode
    elif mode.startswith("Conditional"): mood, tense, definite = "Cnd", "Pres", "Definite" in mode
    else: mood, tense, definite = "Ind", "Fut", "Definite" in mode
    return mood, tense, definite

def make_ud_key_for_verb(mood: str, tense: str, definite: bool, person: int, number: str) -> str:
    dval = "Def" if definite else "Ind"
    return f"VERB VerbForm=Fin|Mood={mood}|Tense={tense}|Person={person}|Number={'Sing' if number=='Sing' else 'Plur'}|Definite={dval}"

def make_ud_key_for_noun(case: str, number: str) -> str:
    case_code = CASE_TO_UD[case]
    return f"NOUN Case={case_code}|Number={'Sing' if number=='Sing' else 'Plur'}"

def choose_person_number() -> Tuple[int, str]:
    return random.choice([1,2,3]), random.choice(["Sing","Plur"])


# --------------------------- task generation ---------------------------------
def next_task(df: pd.DataFrame) -> Tuple[str, Dict, str]:
    scope = []
    if want_verbs and verb_modes: scope.append("verb")
    if want_nouns and noun_modes and noun_numbers: scope.append("noun")
    if not scope: st.stop()
    which = random.choice(scope)

    if which == "verb":
        sub = df[df["pos"].str.upper().eq("VERB")]
        if sub.empty: st.stop()
        row = sub.sample(1).iloc[0]
        mode_choice = random.choice(verb_modes)
        mood, tense, definite = parse_verb_mode(mode_choice)
        person, number = choose_person_number()
        ud_key = make_ud_key_for_verb(mood, tense, definite, person, number)
        is_ik = HuVerb.is_ik(str(row["lemma"]), get_is_ik_flag(row))
        task = VerbTask(
            lemma=str(row["lemma"]), gloss=str(row["english"]),
            mood=mood, tense=tense, definite=definite, person=person, number=number,
            is_ik=is_ik, ud_key=ud_key
        )
        sol = realize_verb(row, task)
        return "verb", task.__dict__, sol

    sub = df[df["pos"].str.upper().eq("NOUN")]
    if sub.empty: st.stop()
    row = sub.sample(1).iloc[0]
    case = random.choice(noun_modes)
    number = "Sing" if random.choice(noun_numbers) == "Singular" else "Plur"
    ud_key = make_ud_key_for_noun(case, number)
    task = NounTask(lemma=str(row["lemma"]), gloss=str(row["english"]), case=case, number=number, ud_key=ud_key)
    sol = realize_noun(row, task)
    return "noun", task.__dict__, sol


# --------------------------- realization -------------------------------------
def realize_from_overrides(row, ud_key: str) -> Optional[str]:
    return lookup_override(row.get("forms", None), ud_key)

def realize_verb(row, task: VerbTask) -> str:
    override = realize_from_overrides(row, task.ud_key)
    if override: return override
    if "ML generator" in prefer_ml and _TRANSFORMERS_OK:
        gen = nyt_generate(task.lemma, task.ud_key)
        if gen: return gen
    if task.tense == "Pres" and task.mood == "Ind":
        return HuVerb.pres_def(task.lemma, task.person, task.number) if task.definite else HuVerb.pres_indef(task.lemma, task.person, task.number, task.is_ik)
    if task.tense == "Past" and task.mood == "Ind":
        return HuVerb.past_def(task.lemma, task.person, task.number) if task.definite else HuVerb.past_indef(task.lemma, task.person, task.number)
    if task.tense == "Pres" and task.mood == "Cnd":
        return HuVerb.cond_def(task.lemma, task.person, task.number) if task.definite else HuVerb.cond_indef(task.lemma, task.person, task.number)
    if task.tense == "Fut" and task.mood == "Ind":
        return HuVerb.future_form(task.lemma, task.definite, task.person, task.number)
    return task.lemma

def realize_noun(row, task: NounTask) -> str:
    override = realize_from_overrides(row, task.ud_key)
    if override: return override
    if "ML generator" in prefer_ml and _TRANSFORMERS_OK:
        gen = nyt_generate(task.lemma, task.ud_key)
        if gen: return gen
    c, n = task.case, task.number
    return {
        "Nominative": HuNoun.nominative,
        "Accusative": HuNoun.accusative,
        "Dative": HuNoun.dative,
        "Inessive": HuNoun.inessive,
        "Superessive": HuNoun.superessive,
        "Adessive": HuNoun.adessive,
        "Illative": HuNoun.illative,
        "Sublative": HuNoun.sublative,
        "Allative": HuNoun.allative,
        "Instrumental": HuNoun.instrumental,
        "Genitive": HuNoun.genitive,
    }[c](task.lemma, n)


# --------------------------- emMorph validation (forefront) -------------------
def _ana_has(a_low: str, *keys: str) -> bool:
    return all(k in a_low for k in keys)

def emmorph_accepts_verb(surface: str, lemma: str, mood: str, tense: str, definite: bool, person: int, number: str) -> Tuple[bool, Optional[str]]:
    if ANALYZER is None: return False, None
    readings = analyze(surface)
    if not readings: return False, None

    for a, _w in readings:
        a_low = a.lower()

        # require lemma mention
        if lemma.lower() not in a_low:
            continue

        # POS looks like [/V] in many outputs; tolerate absence
        # tense/mood heuristics
        if tense == "Pres" and not ("prs" in a_low or "pres" in a_low):
            # Allow present often unmarked; do not hard-fail
            pass
        if tense == "Past" and not ("past" in a_low or "pst" in a_low or "múlt" in a_low):
            continue
        if mood == "Cnd" and not ("cond" in a_low or "cnd" in a_low or "felt" in a_low):
            continue

        # definiteness heuristic
        if definite:
            if not ("def" in a_low):
                continue
        else:
            # reject explicit Def when we need Indef
            if "def" in a_low and not ("ind" in a_low or "indef" in a_low):
                continue

        # person/number patterns: 1sg/sg1/1 sg, etc.
        target = f"{person}{'sg' if number=='Sing' else 'pl'}"
        patterns = {
            target,
            f"{'sg' if number=='Sing' else 'pl'}{person}",
            f"{person} {'sg' if number=='Sing' else 'pl'}",
            f"p{person}",
        }
        if not any(pat in a_low for pat in patterns):
            # Try full words
            if number == "Sing" and "sg" not in a_low and "sing" not in a_low:
                continue
            if number == "Plur" and "pl" not in a_low and "plur" not in a_low:
                continue

        return True, a  # accepted by emMorph
    return False, None

def emmorph_accepts_noun(surface: str, lemma: str, case: str) -> Tuple[bool, Optional[str]]:
    if ANALYZER is None: return False, None
    readings = analyze(surface)
    if not readings: return False, None

    case_token = CASE_TO_UD.get(case, case)  # e.g., Acc, Dat, ...
    case_token = case_token.lower()

    for a, _w in readings:
        a_low = a.lower()
        if lemma.lower() not in a_low:
            continue
        if case_token == "nom":
            if "nom" in a_low: return True, a
            # Often nominative may be implicit; allow base-form acceptance if no other clear case tags present
            if not any(t in a_low for t in ["acc","dat","ine","sup","ade","ill","sub","all","ins","gen"]):
                return True, a
        else:
            if case_token in a_low:
                return True, a
    return False, None


# --------------------------- session state -----------------------------------
if "df" not in st.session_state: st.session_state.df = None
if df is not None:
    ok, msg = validate_corpus(df)
    if ok: st.session_state.df = df.copy()
    else:  st.error(msg)

for k, v in {"score":0, "total":0, "current":None, "solution":"", "kind":"", "feedback":"", "tts_last_audio":None, "checked":False}.items():
    if k not in st.session_state: st.session_state[k] = v

def new_question():
    st.session_state.feedback = ""
    st.session_state.tts_last_audio = None
    st.session_state.checked = False
    if st.session_state.df is None:
        st.warning("Upload or load a corpus CSV to begin.")
        return
    kind, payload, solution = next_task(st.session_state.df)
    st.session_state.kind = kind
    st.session_state.current = payload
    st.session_state.solution = solution


# --------------------------- TTS helper --------------------------------------
def tts_speak_hu(text: str, rate: float) -> Optional[bytes]:
    if not text or tts_provider == "Off":
        return None
    try:
        if tts_provider.startswith("gTTS"):
            if not _HAS_GTTS:
                st.error("gTTS is not installed. Run: pip install gTTS"); return None
            t = gTTS(text, lang="hu")
            fp = BytesIO(); t.write_to_fp(fp); fp.seek(0)
            return fp.read()
        if tts_provider.startswith("Google"):
            if not _HAS_GOOGLE_TTS:
                st.error("google-cloud-texttospeech is not installed. Run: pip install google-cloud-texttospeech google-auth"); return None
            sa = st.secrets.get("GOOGLE_TTS_SERVICE_ACCOUNT_JSON", None)
            creds = service_account.Credentials.from_service_account_info(sa) if isinstance(sa, dict) else None
            client = texttospeech.TextToSpeechClient(credentials=creds) if creds else texttospeech.TextToSpeechClient()
            synthesis_input = texttospeech.SynthesisInput(text=text)
            voice = texttospeech.VoiceSelectionParams(language_code="hu-HU")
            audio_config = texttospeech.AudioConfig(audio_encoding=texttospeech.AudioEncoding.MP3, speaking_rate=float(rate))
            response = client.synthesize_speech(input=synthesis_input, voice=voice, audio_config=audio_config)
            return response.audio_content
    except Exception as e:
        st.error(f"TTS failed: {e}")
    return None


# --------------------------- main interaction --------------------------------
colL, colR = st.columns([2, 1])

with colL:
    if st.session_state.current is None and st.session_state.df is not None:
        new_question()
    elif st.session_state.df is None:
        st.info("Use the sidebar to upload your corpus or load it from GitHub, then click Next.")

    if st.button("Next", use_container_width=True):
        new_question()

    if st.session_state.current:
        c = st.session_state.current
        if st.session_state.kind == "verb":
            pron = PRONOUNS_HU[(c["number"], c["person"])] if show_hu_pronouns else ""
            mode_map = {("Ind","Pres"): "present", ("Ind","Past"): "past", ("Cnd","Pres"): "conditional present", ("Ind","Fut"): "future"}
            mode_label = mode_map.get((c["mood"], c["tense"]), "present")
            conj = f"{'definite' if c['definite'] else 'indefinite'} {mode_label}"
            pron_part = pron if pron else f"person {c['person']}, {c['number']}"
            aux_text = f"Tense and pronoun: {mode_label}, {pron_part}".capitalize()
            st.markdown(
                f"""
                <div class="prompt-card">
                  <div><span class="pill">Verb</span><span class="pill">{conj}</span></div>
                  <div class="mono" style="font-size:1.25rem;margin-top:.25rem;"><b>{c["lemma"]}</b></div>
                  <div class="muted">Meaning: {c["gloss"]}</div>
                  <div class="muted">{aux_text}</div>
                </div>
                """,
                unsafe_allow_html=True
            )
        else:
            st.markdown(
                f"""
                <div class="prompt-card">
                  <div><span class="pill">Noun</span><span class="pill">{c["case"]} • {c["number"]}</span></div>
                  <div class="mono" style="font-size:1.25rem;margin-top:.25rem;"><b>{c["lemma"]}</b></div>
                  <div class="muted">Meaning: {c["gloss"]}</div>
                </div>
                """,
                unsafe_allow_html=True
            )

        colT1, colT2 = st.columns([1, 1])
        with colT1:
            if st.button("🔊 Speak prompt"):
                audio = tts_speak_hu(c["lemma"], tts_rate)
                if audio: st.session_state.tts_last_audio = audio
        with colT2:
            if st.button("🔊 Speak correct form"):
                audio = tts_speak_hu(st.session_state.solution, tts_rate)
                if audio: st.session_state.tts_last_audio = audio

        if st.session_state.tts_last_audio:
            st.audio(st.session_state.tts_last_audio, format="audio/mp3")

        answer = st.text_input("Type the correct form")

        colA, colB = st.columns([1, 1])
        with colA:
            if st.button("Check", disabled=st.session_state.checked or not answer.strip()):
                user = normalize_answer(answer, ignore_accents)
                gold = normalize_answer(st.session_state.solution, ignore_accents)
                st.session_state.total += 1

                accepted_by_emmorph = False
                em_reading = None

                if st.session_state.kind == "verb":
                    ok, em_reading = emmorph_accepts_verb(
                        surface=answer,
                        lemma=c["lemma"],
                        mood=c["mood"],
                        tense=c["tense"],
                        definite=c["definite"],
                        person=c["person"],
                        number=c["number"],
                    )
                    accepted_by_emmorph = ok
                else:
                    ok, em_reading = emmorph_accepts_noun(
                        surface=answer,
                        lemma=c["lemma"],
                        case=c["case"],
                    )
                    accepted_by_emmorph = ok

                if user == gold or accepted_by_emmorph:
                    st.session_state.score += 1
                    if accepted_by_emmorph and user != gold:
                        st.session_state.feedback = f"<span class='good'>Correct (emMorph).</span> Your answer is valid: <span class='mono'>{answer}</span><br><span class='muted'>emMorph reading: {em_reading}</span><br><span class='muted'>Rule/CSV target was: <b>{st.session_state.solution}</b></span>"
                    else:
                        st.session_state.feedback = f"<span class='good'>Correct.</span> {st.session_state.solution}"
                    if auto_say_answer:
                        audio = tts_speak_hu(answer if accepted_by_emmorph else st.session_state.solution, tts_rate)
                        if audio: st.session_state.tts_last_audio = audio
                else:
                    st.session_state.feedback = f"<span class='bad'>Not quite.</span> Expected: <b>{st.session_state.solution}</b>"
                st.session_state.checked = True

        with colB:
            if allow_reveal and st.button("Reveal"):
                st.session_state.feedback = f"Answer: <b>{st.session_state.solution}</b>"
                if auto_say_answer:
                    audio = tts_speak_hu(st.session_state.solution, tts_rate)
                    if audio: st.session_state.tts_last_audio = audio

        if st.session_state.feedback:
            st.markdown(st.session_state.feedback, unsafe_allow_html=True)

with colR:
    acc, tot = st.session_state.score, st.session_state.total
    rate = f"{(100 * acc / tot):.0f}%" if tot else "0%"
    st.markdown(
        f"""
        <div class="metric-card">
          <div class="muted" style="font-size:.9rem;">Accuracy</div>
          <div style="font-size:1.6rem; font-weight:700;">{rate}</div>
          <div class="muted">{acc}/{tot} correct</div>
        </div>
        """,
        unsafe_allow_html=True
    )
    if st.session_state.df is not None:
        st.caption("Corpus loaded and cached for quick sampling.")

st.caption(
    "Noun cases: Nom, Acc, Dat, Ine, Sup, Ade, Ill, Sub, All, Ins, Gen. "
    "Verbs: present, past, conditional, and future (def/indef). "
    "emMorph is used to accept valid answers even when heuristics would disagree."
)
