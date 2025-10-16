# app.py
# Hungarian Conjugation and Declension Practice
# Python 3.12 • Streamlit app with GitHub corpus loading,
# pastel UI, single-button navigation, AI TTS, and 11 noun cases:
# Nom, Acc, Dat, Ine, Sup, Ade, Ill, Sub, All, Ins, Gen

from __future__ import annotations

import json
import random
import re
from dataclasses import dataclass
from functools import lru_cache
from io import BytesIO
from typing import Dict, Optional, Tuple

import pandas as pd
import streamlit as st

# Optional GitHub integration
try:
    from github import Github  # pip install PyGithub
    _GITHUB_OK = True
except Exception:
    _GITHUB_OK = False

# Optional neural morphological generator
_TRANSFORMERS_OK = False
try:
    from transformers import pipeline  # pip install transformers torch sentencepiece
    _TRANSFORMERS_OK = True
except Exception:
    pass

# Optional AI TTS backends
_HAS_GTTS = False
try:
    # pip install gTTS
    from gtts import gTTS
    _HAS_GTTS = True
except Exception:
    pass

_HAS_GOOGLE_TTS = False
try:
    # pip install google-cloud-texttospeech google-auth
    from google.cloud import texttospeech  # type: ignore
    from google.oauth2 import service_account  # type: ignore
    _HAS_GOOGLE_TTS = True
except Exception:
    pass


# ------------------------- PAGE CONFIG + PASTEL THEME -------------------------

st.set_page_config(
    page_title="Hungarian Conjugations & Declensions Trainer",
    page_icon="🇭🇺",
    layout="wide",
)

st.markdown(
    """
    <style>
    :root{
      --bg1:#f6f0ff;
      --bg2:#e9f7ff;
      --surface:#eef2ff;
      --surface2:#e7f7f4;
      --ink:#1f2937;
      --muted:#4b5563;
      --accent:#a7c8ff;
      --accent-ink:#0f172a;
      --border:#cfd8ee;
      --pill:#d6efff;
      --good:#1c8c4e;
      --bad:#b21b1b;
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
      border: 1px solid var(--border);
      padding: 1rem 1.25rem;
      border-radius: 12px;
      background: var(--surface);
      box-shadow: 0 1px 0 rgba(16,24,40,.03);
      margin-bottom: 1rem;
    }
    .pill{
      display:inline-block;
      font-size:.85rem;
      padding:.12rem .6rem;
      border:1px solid var(--border);
      border-radius:999px;
      margin-right:.35rem;
      background: var(--pill);
    }
    .big-title{ font-size: 1.8rem; font-weight: 700; letter-spacing: .2px; margin-bottom: .25rem; color:var(--ink); }
    .subtitle{ color: var(--muted); margin-bottom: 1rem; }

    .stButton > button{
      background: var(--accent) !important;
      color: var(--accent-ink) !important;
      border: 1px solid var(--border) !important;
      border-radius: 10px !important;
      box-shadow: 0 1px 0 rgba(16,24,40,.06) !important;
    }
    .stButton > button:hover{
      filter: brightness(0.98);
      transform: translateY(-1px);
      transition: transform .08s ease;
    }

    input, textarea, select{
      background-color: var(--surface) !important;
      color: var(--ink) !important;
      border: 1px solid var(--border) !important;
    }
    .stTextInput > div > div > input{ background-color: var(--surface) !important; }
    .stSelectbox div[role="combobox"]{
      background-color: var(--surface) !important;
      border: 1px solid var(--border) !important;
    }

    .metric-card{
      background: var(--surface2);
      border: 1px solid var(--border);
      border-radius: 12px;
      padding: .75rem 1rem;
    }

    .good{ color: var(--good); font-weight: 700; }
    .bad{ color: var(--bad); font-weight: 700; }
    .muted{ color: var(--muted); }
    .mono{ font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, "Liberation Mono", monospace; }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown('<div class="big-title">Hungarian Conjugations and Declensions</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">Practice accurate present‑tense verb forms and core noun cases with a clean pastel interface.</div>', unsafe_allow_html=True)


# ------------------------- DATA TYPES -------------------------

@dataclass(frozen=True)
class VerbTask:
    lemma: str
    gloss: str
    definite: bool
    person: int
    number: str
    is_ik: bool
    ud_key: str

@dataclass(frozen=True)
class NounTask:
    lemma: str
    gloss: str
    case: str
    ud_key: str


# ------------------------- SETTINGS SIDEBAR -------------------------

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
    verb_modes = st.multiselect(
        "Present tense",
        options=["Indefinite", "Definite"],
        default=["Indefinite", "Definite"],
        help="Indefinite vs. definite conjugation endings in Hungarian present tense."
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
        default=["Accusative", "Dative", "Inessive", "Superessive", "Adessive", "Illative", "Sublative", "Allative", "Instrumental"],
        help="Practice the most common cases; Genitive here is the possessive -é form."
    )

    st.divider()

    advanced = st.expander("Advanced accuracy")
    with advanced:
        prefer_ml = st.selectbox(
            "Inflection strategy",
            ["CSV overrides first, then ML generator, then rules", "CSV overrides only", "CSV overrides then rules only"],
            help="For maximum accuracy choose the ML generator path. It uses UD-style features."
        )
        ignore_accents = st.checkbox("Accept answers that ignore accents", value=True)
        show_hu_pronouns = st.checkbox("Show Hungarian pronouns for verb prompts", value=True)
        allow_reveal = st.checkbox("Allow Reveal Answer", value=True)

    st.divider()

    tts_expander = st.expander("Pronunciation (AI TTS)")
    with tts_expander:
        tts_provider = st.selectbox("TTS provider", ["Off", "gTTS (local, free)", "Google Cloud TTS"], index=0)
        tts_rate = st.slider("Speaking rate", 0.6, 1.4, 1.0, 0.05)
        auto_say_answer = st.checkbox("Auto speak correct answer on Reveal or when correct", value=True)
        st.caption(
            "For Google Cloud, add a service account JSON to Streamlit secrets as GOOGLE_TTS_SERVICE_ACCOUNT_JSON. "
            "Hungarian language code is hu-HU."
        )
        if st.button("Test TTS with “Szia!”"):
            audio = None
            try:
                if tts_provider.startswith("gTTS"):
                    if not _HAS_GTTS:
                        st.error("gTTS is not installed. Run: pip install gTTS")
                    else:
                        g = gTTS("Szia!", lang="hu")
                        buf = BytesIO()
                        g.write_to_fp(buf)
                        buf.seek(0)
                        audio = buf.read()
                elif tts_provider.startswith("Google"):
                    if not _HAS_GOOGLE_TTS:
                        st.error("google-cloud-texttospeech is not installed. Run: pip install google-cloud-texttospeech google-auth")
                    else:
                        sa = st.secrets.get("GOOGLE_TTS_SERVICE_ACCOUNT_JSON", None)
                        creds = service_account.Credentials.from_service_account_info(sa) if sa else None
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


# ------------------------- CSV TEMPLATE DOWNLOAD -------------------------

CSV_TEMPLATE = """
pos,lemma,english,is_ik,forms
VERB,kér,to ask,False,"{""VERB VerbForm=Fin|Mood=Ind|Tense=Pres|Person=1|Number=Sing|Definite=Ind"": ""kérek"", ""VERB VerbForm=Fin|Mood=Ind|Tense=Pres|Person=3|Number=Plur|Definite=Def"": ""kérik""}"
VERB,dolgozik,to work,True,"{""VERB VerbForm=Fin|Mood=Ind|Tense=Pres|Person=1|Number=Sing|Definite=Ind"": ""dolgozom"", ""VERB VerbForm=Fin|Mood=Ind|Tense=Pres|Person=3|Number=Sing|Definite=Ind"": ""dolgozik""}"
NOUN,bolt,shop,, "{""NOUN Case=Ine|Number=Sing"": ""boltban"", ""NOUN Case=Ade|Number=Sing"": ""boltnál"", ""NOUN Case=Ill|Number=Sing"": ""boltba"", ""NOUN Case=Sub|Number=Sing"": ""boltra"", ""NOUN Case=All|Number=Sing"": ""bolthoz"", ""NOUN Case=Sup|Number=Sing"": ""bolton"", ""NOUN Case=Dat|Number=Sing"": ""boltnak"", ""NOUN Case=Ins|Number=Sing"": ""bolttal"", ""NOUN Case=Acc|Number=Sing"": ""boltot"", ""NOUN Case=Gen|Number=Sing"": ""bolté""}"
""".strip()

with st.sidebar:
    st.download_button("Download CSV template", data=CSV_TEMPLATE, file_name="hungarian_corpus_template.csv", mime="text/csv")


# ------------------------- CORE UTILITIES -------------------------

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

ACCENT_STRIP = str.maketrans(
    {
        "á":"a","é":"e","í":"i","ó":"o","ö":"o","ő":"o","ú":"u","ü":"u","ű":"u",
        "Á":"a","É":"e","Í":"i","Ó":"o","Ö":"o","Ő":"o","Ú":"u","Ü":"u","Ű":"u",
    }
)

def normalize_answer(s: str, strip_accents: bool) -> str:
    s = s.strip()
    if strip_accents:
        s = s.translate(ACCENT_STRIP)
    return s.lower()


# ------------------------- RULE ENGINE (EXPANDED NOUN CASES) -------------------------

class HuRules:
    """Targeted fallback rules for common cases. The ML generator or CSV overrides remain the primary accuracy path."""

    # Baseline cases
    @staticmethod
    def nominative(noun: str) -> str:
        return noun

    @staticmethod
    def dative(noun: str) -> str:
        return noun + ("nak" if harmony_set(noun) == "back" else "nek")

    @staticmethod
    def inessive(noun: str) -> str:
        return noun + ("ban" if harmony_set(noun) == "back" else "ben")

    @staticmethod
    def superessive(noun: str) -> str:
        # Many vowel-final nouns prefer plain -n (autón). Otherwise -on/-en/-ön by harmony.
        if noun[-1].lower() in ALL_VOWELS:
            return noun + "n"
        h = harmony_set(noun)
        return noun + ("on" if h == "back" else "ön" if h == "front_r" else "en")

    @staticmethod
    def adessive(noun: str) -> str:
        return noun + ("nál" if harmony_set(noun) == "back" else "nél")

    @staticmethod
    def illative(noun: str) -> str:
        return noun + ("ba" if harmony_set(noun) == "back" else "be")

    @staticmethod
    def sublative(noun: str) -> str:
        return noun + ("ra" if harmony_set(noun) == "back" else "re")

    @staticmethod
    def allative(noun: str) -> str:
        h = harmony_set(noun)
        return noun + ("hoz" if h == "back" else "höz" if h == "front_r" else "hez")

    @staticmethod
    def instrumental(noun: str) -> str:
        # -val/-vel with v-assimilation and gemination after consonant stems
        if noun[-1].lower() in ALL_VOWELS:
            return noun + ("val" if harmony_set(noun) == "back" else "vel")
        # Consonant-final: drop v and double final consonant, choose -al/-el by harmony
        ending = "al" if harmony_set(noun) == "back" else "el"
        last_char = noun[-1]
        # Handle digraphs like sz, zs, cs, gy, ny, ty, ly by doubling logically
        if noun.endswith(("sz","zs","cs","gy","ny","ty","ly")):
            return noun + noun[-2:] + ending
        return noun + last_char + ending

    @staticmethod
    def genitive(noun: str) -> str:
        # Practical learner-friendly "possessive -é" form, with a/e lengthening
        if noun.endswith("a"):
            return noun[:-1] + "á" + "é"
        if noun.endswith("e"):
            return noun[:-1] + "é" + "é"
        return noun + "é"

    @staticmethod
    def accusative(noun: str) -> str:
        # Heuristic:
        # 1) a/e at word end lengthen + t
        if noun.endswith("a"):
            return noun[:-1] + "á" + "t"
        if noun.endswith("e"):
            return noun[:-1] + "é" + "t"
        # 2) other vowel-final: +t
        if noun[-1].lower() in ALL_VOWELS:
            return noun + "t"
        # 3) consonant-final: choose between bare -t and a linking vowel + t
        # Use linking vowel with final d/t or sibilants, and often with hard clusters
        last = noun[-1].lower()
        fin_bigram = noun[-2:].lower() if len(noun) >= 2 else ""
        sibilant_like = last in {"s","z","c"} or fin_bigram in {"sz","zs","cs","dz","dzs"}
        dental_like = last in {"t","d"}
        use_link = dental_like or sibilant_like
        # Pick a linking vowel. Prefer 'a' after last a/á, 'e' after e/é, else by harmony.
        lv = "a" if last_vowel(noun) in {"a","á"} else "e" if last_vowel(noun) in {"e","é"} else (
            "o" if harmony_set(noun) == "back" else "ö" if harmony_set(noun) == "front_r" else "e"
        )
        if use_link:
            return noun + lv + "t"
        # Otherwise many polysyllabic consonant-final nouns accept bare -t
        return noun + "t"

    # Verb helpers from earlier app
    @staticmethod
    def is_ik(lemma: str, csv_flag: Optional[bool]) -> bool:
        if csv_flag is True:
            return True
        if csv_flag is False:
            return False
        return lemma.endswith("ik")

    @staticmethod
    def present_indef(lemma: str, person: int, number: str, is_ik: bool) -> str:
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
        if number == "Plur" and person == 1:
            return lemma + v_1pl
        if number == "Plur" and person == 2:
            return lemma + v_2pl
        if number == "Plur" and person == 3:
            return lemma + v_3pl
        return lemma

    @staticmethod
    def present_def(lemma: str, person: int, number: str) -> str:
        h = harmony_set(lemma)
        v_1sg = {"back": "om", "front_unr": "em", "front_r": "öm"}[h]
        v_2sg = {"back": "od", "front_unr": "ed", "front_r": "öd"}[h]
        v_1pl = {"back": "juk", "front_unr": "jük", "front_r": "jük"}[h]
        v_2pl_cons = {"back": "játok", "front_unr": "itek", "front_r": "itek"}[h]
        v_2pl_vow = {"back": "játok", "front_unr": "jétek", "front_r": "jétek"}[h]
        v_3pl_default = {"back": "ják", "front_unr": "ik", "front_r": "ik"}[h]
        ends_vowel = lemma[-1].lower() in ALL_VOWELS

        if number == "Sing" and person == 1:
            return lemma + v_1sg
        if number == "Sing" and person == 2:
            return lemma + v_2sg
        if number == "Sing" and person == 3:
            if re.search(r"(z)$", lemma):
                return lemma + "i"
            if re.search(r"(s|sz|zs)$", lemma):
                if lemma.endswith("sz"):
                    base = lemma[:-2] + "ssz"
                elif lemma.endswith("zs"):
                    base = lemma[:-2] + "zzs"
                elif lemma.endswith("s"):
                    base = lemma[:-1] + "ss"
                else:
                    base = lemma
                return base + ("a" if h == "back" else "e")
            return lemma + ("ja" if h == "back" else "je")
        if number == "Plur" and person == 1:
            return lemma + v_1pl
        if number == "Plur" and person == 2:
            return lemma + (v_2pl_vow if ends_vowel else v_2pl_cons)
        if number == "Plur" and person == 3:
            if re.search(r"(z|sz|zs)$", lemma):
                return lemma + "ik"
            return lemma + v_3pl_default
        return lemma


# ------------------------- NYTK mT5 GENERATOR -------------------------

@st.cache_resource(show_spinner=False)
def get_nytk_generator():
    if not _TRANSFORMERS_OK:
        return None
    try:
        gen = pipeline(task="text2text-generation", model="NYTK/morphological-generator-ud-mt5-hungarian")
        return gen
    except Exception:
        return None

def nyt_generate(lemma: str, ud_key: str) -> Optional[str]:
    gen = get_nytk_generator()
    if not gen:
        return None
    prompt = f"morph: {lemma} {ud_key}"
    try:
        out = gen(prompt, max_new_tokens=12, num_return_sequences=1)[0]["generated_text"]
        return out.strip()
    except Exception:
        return None


# ------------------------- CORPUS HANDLING -------------------------

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
        if pd.isna(val):
            return None
        if isinstance(val, bool):
            return val
        if str(val).strip().lower() in {"true", "1", "yes"}:
            return True
        if str(val).strip().lower() in {"false", "0", "no"}:
            return False
        return None
    except Exception:
        return None


# ------------------------- QUESTION GENERATION -------------------------

PRONOUNS_HU = {
    ("Sing", 1): "én",
    ("Sing", 2): "te",
    ("Sing", 3): "ő",
    ("Plur", 1): "mi",
    ("Plur", 2): "ti",
    ("Plur", 3): "ők",
}

CASE_TO_UD = {
    "Nominative": "Nom",
    "Accusative": "Acc",
    "Dative": "Dat",
    "Inessive": "Ine",
    "Superessive": "Sup",
    "Adessive": "Ade",
    "Illative": "Ill",
    "Sublative": "Sub",
    "Allative": "All",
    "Instrumental": "Ins",
    "Genitive": "Gen",
}

def make_ud_key_for_verb(definite: bool, person: int, number: str) -> str:
    dval = "Def" if definite else "Ind"
    return f"VERB VerbForm=Fin|Mood=Ind|Tense=Pres|Person={person}|Number={'Sing' if number=='Sing' else 'Plur'}|Definite={dval}"

def make_ud_key_for_noun(case: str) -> str:
    case_code = CASE_TO_UD[case]
    return f"NOUN Case={case_code}|Number=Sing"

def choose_person_number() -> Tuple[int, str]:
    person = random.choice([1, 2, 3])
    number = random.choice(["Sing", "Plur"])
    return person, number

def next_task(df: pd.DataFrame) -> Tuple[str, Dict, str]:
    scope = []
    if want_verbs and verb_modes:
        scope.append("verb")
    if want_nouns and noun_modes:
        scope.append("noun")
    if not scope:
        st.stop()

    which = random.choice(scope)

    if which == "verb":
        sub = df[df["pos"].str.upper().eq("VERB")]
        if sub.empty:
            st.stop()
        row = sub.sample(1).iloc[0]
        definite = random.choice([m for m in ["Indefinite", "Definite"] if m in verb_modes]) == "Definite"
        person, number = choose_person_number()
        ud_key = make_ud_key_for_verb(definite, person, number)
        is_ik = HuRules.is_ik(str(row["lemma"]), get_is_ik_flag(row))
        task = VerbTask(
            lemma=str(row["lemma"]),
            gloss=str(row["english"]),
            definite=definite,
            person=person,
            number=number,
            is_ik=is_ik,
            ud_key=ud_key
        )
        sol = realize_verb(row, task)
        return "verb", task.__dict__, sol

    sub = df[df["pos"].str.upper().eq("NOUN")]
    if sub.empty:
        st.stop()
    row = sub.sample(1).iloc[0]
    case = random.choice(noun_modes)
    ud_key = make_ud_key_for_noun(case)
    task = NounTask(
        lemma=str(row["lemma"]),
        gloss=str(row["english"]),
        case=case,
        ud_key=ud_key
    )
    sol = realize_noun(row, task)
    return "noun", task.__dict__, sol


# ------------------------- REALIZATION -------------------------

def realize_from_overrides(row, ud_key: str) -> Optional[str]:
    return lookup_override(row.get("forms", None), ud_key)

def realize_verb(row, task: VerbTask) -> str:
    override = realize_from_overrides(row, task.ud_key)
    if override:
        return override
    if "ML generator" in prefer_ml and _TRANSFORMERS_OK:
        gen = nyt_generate(task.lemma, task.ud_key)
        if gen:
            return gen
    if task.definite:
        return HuRules.present_def(task.lemma, task.person, task.number)
    return HuRules.present_indef(task.lemma, task.person, task.number, task.is_ik)

def realize_noun(row, task: NounTask) -> str:
    override = realize_from_overrides(row, task.ud_key)
    if override:
        return override
    if "ML generator" in prefer_ml and _TRANSFORMERS_OK:
        gen = nyt_generate(task.lemma, task.ud_key)
        if gen:
            return gen
    # Fallback rules per case
    c = task.case
    if c == "Nominative":
        return HuRules.nominative(task.lemma)
    if c == "Accusative":
        return HuRules.accusative(task.lemma)
    if c == "Dative":
        return HuRules.dative(task.lemma)
    if c == "Inessive":
        return HuRules.inessive(task.lemma)
    if c == "Superessive":
        return HuRules.superessive(task.lemma)
    if c == "Adessive":
        return HuRules.adessive(task.lemma)
    if c == "Illative":
        return HuRules.illative(task.lemma)
    if c == "Sublative":
        return HuRules.sublative(task.lemma)
    if c == "Allative":
        return HuRules.allative(task.lemma)
    if c == "Instrumental":
        return HuRules.instrumental(task.lemma)
    if c == "Genitive":
        return HuRules.genitive(task.lemma)
    return task.lemma


# ------------------------- SESSION STATE -------------------------

if "df" not in st.session_state:
    st.session_state.df = None
if df is not None:
    ok, msg = validate_corpus(df)
    if ok:
        st.session_state.df = df.copy()
    else:
        st.error(msg)

if "score" not in st.session_state:
    st.session_state.score = 0
if "total" not in st.session_state:
    st.session_state.total = 0
if "current" not in st.session_state:
    st.session_state.current = None
if "solution" not in st.session_state:
    st.session_state.solution = ""
if "kind" not in st.session_state:
    st.session_state.kind = ""
if "feedback" not in st.session_state:
    st.session_state.feedback = ""
if "tts_last_audio" not in st.session_state:
    st.session_state.tts_last_audio = None

def new_question():
    st.session_state.feedback = ""
    st.session_state.tts_last_audio = None
    if st.session_state.df is None:
        st.warning("Upload or load a corpus CSV to begin.")
        return
    kind, payload, solution = next_task(st.session_state.df)
    st.session_state.kind = kind
    st.session_state.current = payload
    st.session_state.solution = solution


# ------------------------- AI TTS -------------------------

def tts_speak_hu(text: str, rate: float) -> Optional[bytes]:
    if not text or tts_provider == "Off":
        return None
    try:
        if tts_provider.startswith("gTTS"):
            if not _HAS_GTTS:
                st.error("gTTS is not installed. Run: pip install gTTS")
                return None
            t = gTTS(text, lang="hu")
            fp = BytesIO()
            t.write_to_fp(fp)
            fp.seek(0)
            return fp.read()
        if tts_provider.startswith("Google"):
            if not _HAS_GOOGLE_TTS:
                st.error("google-cloud-texttospeech is not installed. Run: pip install google-cloud-texttospeech google-auth")
                return None
            sa = st.secrets.get("GOOGLE_TTS_SERVICE_ACCOUNT_JSON", None)
            creds = service_account.Credentials.from_service_account_info(sa) if sa else None
            client = texttospeech.TextToSpeechClient(credentials=creds) if creds else texttospeech.TextToSpeechClient()
            synthesis_input = texttospeech.SynthesisInput(text=text)
            voice = texttospeech.VoiceSelectionParams(language_code="hu-HU")
            audio_config = texttospeech.AudioConfig(audio_encoding=texttospeech.AudioEncoding.MP3, speaking_rate=float(rate))
            response = client.synthesize_speech(input=synthesis_input, voice=voice, audio_config=audio_config)
            return response.audio_content
    except Exception as e:
        st.error(f"TTS failed: {e}")
    return None


# ------------------------- MAIN INTERACTION -------------------------

colL, colR = st.columns([2, 1])

with colL:
    if st.session_state.current is None and st.session_state.df is not None:
        new_question()
    elif st.session_state.df is None:
        st.info("Use the sidebar to upload your corpus or load it from GitHub, then click New Question.")

    if st.button("New Question", use_container_width=True):
        new_question()

    if st.session_state.current:
        c = st.session_state.current
        if st.session_state.kind == "verb":
            pron = PRONOUNS_HU[(c["number"], c["person"])] if show_hu_pronouns else ""
            conj = "definite present" if c["definite"] else "indefinite present"
            aux = f"Tense and pronoun: present, {pron or f'person {c['person']}, {c['number']}'.capitalize()}"
            st.markdown(
                f"""
                <div class="prompt-card">
                  <div><span class="pill">Verb</span><span class="pill">{conj}</span></div>
                  <div class="mono" style="font-size:1.25rem;margin-top:.25rem;"><b>{c["lemma"]}</b></div>
                  <div class="muted">Meaning: {c["gloss"]}</div>
                  <div class="muted">{aux}</div>
                </div>
                """,
                unsafe_allow_html=True
            )
        else:
            st.markdown(
                f"""
                <div class="prompt-card">
                  <div><span class="pill">Noun</span><span class="pill">{c["case"]}</span></div>
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
                if audio:
                    st.session_state.tts_last_audio = audio
        with colT2:
            if st.button("🔊 Speak correct form"):
                audio = tts_speak_hu(st.session_state.solution, tts_rate)
                if audio:
                    st.session_state.tts_last_audio = audio

        if st.session_state.tts_last_audio:
            st.audio(st.session_state.tts_last_audio, format="audio/mp3")

        answer = st.text_input("Type the correct form")

        colA, colB = st.columns([1, 1])
        with colA:
            if st.button("Check"):
                user = normalize_answer(answer, ignore_accents)
                gold = normalize_answer(st.session_state.solution, ignore_accents)
                st.session_state.total += 1
                if user == gold and len(gold) > 0:
                    st.session_state.score += 1
                    st.session_state.feedback = f"<span class='good'>Correct.</span> {st.session_state.solution}"
                    if auto_say_answer:
                        audio = tts_speak_hu(st.session_state.solution, tts_rate)
                        if audio:
                            st.session_state.tts_last_audio = audio
                else:
                    st.session_state.feedback = f"<span class='bad'>Not quite.</span> Expected: <b>{st.session_state.solution}</b>"
        with colB:
            if allow_reveal and st.button("Reveal"):
                st.session_state.feedback = f"Answer: <b>{st.session_state.solution}</b>"
                if auto_say_answer:
                    audio = tts_speak_hu(st.session_state.solution, tts_rate)
                    if audio:
                        st.session_state.tts_last_audio = audio

        if st.session_state.feedback:
            st.markdown(st.session_state.feedback, unsafe_allow_html=True)

with colR:
    acc = st.session_state.score
    tot = st.session_state.total
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
    "Noun cases supported: Nominative, Accusative, Dative, Inessive, Superessive, Adessive, Illative, Sublative, Allative, Instrumental, and Genitive as the -é possessive form. "
    "For highest accuracy, provide forms in your CSV or enable the neural generator; otherwise fast fallback rules apply."
)
