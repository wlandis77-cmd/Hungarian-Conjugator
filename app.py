# app.py
# Hungarian Conjugation and Declension Practice
# Python 3.12 • Streamlit app with optional GitHub-backed corpus loading
# Accuracy first: CSV overrides > NYTK mT5 generator (optional) > rule engine fallback

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

# Optional neural morphological generator (accurate but heavy on first load)
_TRANSFORMERS_OK = False
try:
    from transformers import pipeline  # pip install transformers torch sentencepiece
    _TRANSFORMERS_OK = True
except Exception:
    pass

# ------------------------- UI THEME AND STYLES -------------------------

st.set_page_config(
    page_title="Hungarian Conjugations & Declensions Trainer",
    page_icon="🇭🇺",
    layout="wide",
)

st.markdown(
    """
    <style>
    .big-title { font-size: 1.8rem; font-weight: 700; letter-spacing: .2px; margin-bottom: .25rem; }
    .subtitle { color: #666; margin-bottom: 1rem; }
    .prompt-card {
        border: 1px solid #e6e9ef;
        padding: 1rem 1.25rem;
        border-radius: 10px;
        background: #fafbff;
        box-shadow: 0 1px 0 rgba(16,24,40,.02);
        margin-bottom: 1rem;
    }
    .good { color: #0a7f2e; font-weight: 700; }
    .bad { color: #b21b1b; font-weight: 700; }
    .pill { display: inline-block; font-size: .85rem; padding: .1rem .5rem; border: 1px solid #e2e8f0; border-radius: 999px; margin-right: .35rem; background: white; }
    .muted { color: #6b7280; }
    .mono { font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, "Liberation Mono", monospace; }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown('<div class="big-title">Hungarian Conjugations and Declensions</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">Practice accurate present‑tense verb forms and core noun cases with a fast, clean workflow.</div>', unsafe_allow_html=True)

# ------------------------- DATA TYPES -------------------------

@dataclass(frozen=True)
class VerbTask:
    lemma: str
    gloss: str
    definite: bool  # True = definite conjugation, False = indefinite
    person: int     # 1..3
    number: str     # "Sing" or "Plur"
    is_ik: bool     # -ik verb behavior hints
    ud_key: str     # UD-style feature key for generator/CSV overrides


@dataclass(frozen=True)
class NounTask:
    lemma: str
    gloss: str
    case: str       # "Ine" or "Ade" (inessive/adessive)
    ud_key: str     # UD-style feature key for generator/CSV overrides


# ------------------------- SETTINGS SIDEBAR -------------------------

with st.sidebar:
    st.header("Settings")
    source = st.radio(
        "Corpus source",
        ["Upload CSV", "Load from GitHub"],
        help="Provide one corpus CSV with both verbs and nouns. The app caches it for speed."
    )

    df: Optional[pd.DataFrame] = None

    # GitHub settings
    if source == "Load from GitHub":
        repo_full = st.text_input("owner/repo", placeholder="yourname/yourrepo")
        path_in_repo = st.text_input("path in repo", value="data/hungarian_corpus.csv")
        ref = st.text_input("branch or tag", value="main")
        token_hint = "Place your PAT in Streamlit secrets as GITHUB_TOKEN."
        st.caption(token_hint)

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
    noun_modes = st.multiselect(
        "Cases",
        options=["Inessive", "Adessive"],
        default=["Inessive", "Adessive"],
        help="Inessive is -ban/-ben. Adessive is -nál/-nél."
    )

    st.divider()

    advanced = st.expander("Advanced accuracy")
    with advanced:
        prefer_ml = st.selectbox(
            "Inflection strategy",
            ["CSV overrides first, then ML generator, then rules", "CSV overrides only", "CSV overrides then rules only"],
            help="For maximum accuracy choose the ML generator path. It uses NYTK’s mT5 model and UD features."
        )
        ignore_accents = st.checkbox("Accept answers that ignore accents", value=True)
        show_hu_pronouns = st.checkbox("Show Hungarian pronouns for verb prompts", value=True)
        allow_reveal = st.checkbox("Allow Reveal Answer", value=True)
        st.caption("The ML generator requires Transformers and will download a model on first use.")

# ------------------------- CSV CONTRACT -------------------------

CSV_TEMPLATE = """
pos,lemma,english,is_ik,forms
VERB,kér,to ask,False,"{""VERB VerbForm=Fin|Mood=Ind|Tense=Pres|Person=1|Number=Sing|Definite=Ind"": ""kérek"", ""VERB VerbForm=Fin|Mood=Ind|Tense=Pres|Person=3|Number=Plur|Definite=Def"": ""kérik""}"
VERB,dolgozik,to work,True,"{""VERB VerbForm=Fin|Mood=Ind|Tense=Pres|Person=1|Number=Sing|Definite=Ind"": ""dolgozom"", ""VERB VerbForm=Fin|Mood=Ind|Tense=Pres|Person=3|Number=Sing|Definite=Ind"": ""dolgozik""}"
NOUN,iroda,office,, "{""NOUN Case=Ine|Number=Sing"": ""irodában"", ""NOUN Case=Ade|Number=Sing"": ""irodánál""}"
NOUN,bolt,shop,, "{""NOUN Case=Ine|Number=Sing"": ""boltban"", ""NOUN Case=Ade|Number=Sing"": ""boltnál""}"
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

def last_vowel_group(s: str) -> str:
    # Helps select linking vowels
    last = None
    for ch in s.lower():
        if ch in ALL_VOWELS:
            last = ch
    return last or "a"

def harmony_set(s: str) -> str:
    # "back", "front_unr", "front_r"
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

# ------------------------- RULE ENGINE (TARGETED) -------------------------

class HuRules:
    """Rule-based generator for specific practice modes.
    Nouns: Inessive (-ban/-ben) and Adessive (-nál/-nél).
    Verbs: Present tense, prioritizing regular patterns. -ik handling for 1sg and 3sg (indef)."""

    @staticmethod
    def inessive(noun: str) -> str:
        h = harmony_set(noun)
        return noun + ("ban" if h == "back" else "ben")

    @staticmethod
    def adessive(noun: str) -> str:
        h = harmony_set(noun)
        return noun + ("nál" if h == "back" else "nél")

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
        # Heuristic: after vowel use -játok, after consonant use -itek/-itek selection by harmony
        v_2pl_cons = {"back": "játok", "front_unr": "itek", "front_r": "itek"}[h]
        v_2pl_vow = {"back": "játok", "front_unr": "jétek", "front_r": "jétek"}[h]
        v_3pl_default = {"back": "ják", "front_unr": "ik", "front_r": "ik"}[h]

        ends_vowel = lemma[-1].lower() in ALL_VOWELS

        if number == "Sing" and person == 1:
            return lemma + v_1sg
        if number == "Sing" and person == 2:
            return lemma + v_2sg
        if number == "Sing" and person == 3:
            # Default to -ja/-je with assimilation for s/z endings; use -i for many -z verbs like "néz"
            if re.search(r"(z)$", lemma):
                return lemma + "i"
            if re.search(r"(s|sz|zs)$", lemma):
                # geminate sibilant + a/e
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
            # Many -z/-sz verbs use -ik, otherwise -ják/-ik by harmony; this is a heuristic.
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
    # UD format example: "morph: munka NOUN Case=Acc|Number=Sing"
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
    if not forms_json or (isinstance(forms_json, float) and pd.isna(forms_json)):  # NaN safe
        return None
    try:
        data = json.loads(forms_json)
        # Two keys accepted: exact UD key or simplified fallback
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

def make_ud_key_for_verb(definite: bool, person: int, number: str) -> str:
    # UD-style feature bundle
    # Example: VERB VerbForm=Fin|Mood=Ind|Tense=Pres|Person=1|Number=Sing|Definite=Ind
    dval = "Def" if definite else "Ind"
    return f"VERB VerbForm=Fin|Mood=Ind|Tense=Pres|Person={person}|Number={'Sing' if number=='Sing' else 'Plur'}|Definite={dval}"

def make_ud_key_for_noun(case: str) -> str:
    # Case: "Inessive" -> "Ine", "Adessive" -> "Ade"
    c = "Ine" if case == "Inessive" else "Ade"
    return f"NOUN Case={c}|Number=Sing"

def choose_person_number() -> Tuple[int, str]:
    person = random.choice([1, 2, 3])
    number = random.choice(["Sing", "Plur"])
    return person, number

def next_task(df: pd.DataFrame) -> Tuple[str, Dict, str]:
    # Returns ("verb" or "noun", payload dict, solution string)
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

# ------------------------- REALIZATION (INFLECTION) -------------------------

def realize_from_overrides(row, ud_key: str) -> Optional[str]:
    return lookup_override(row.get("forms", None), ud_key)

def realize_verb(row, task: VerbTask) -> str:
    # Priority 1: explicit CSV overrides
    override = realize_from_overrides(row, task.ud_key)
    if override:
        return override

    # Priority 2: NYTK mT5 generator, if chosen
    if "ML generator" in prefer_ml and _TRANSFORMERS_OK:
        gen = nyt_generate(task.lemma, task.ud_key)
        if gen:
            return gen

    # Priority 3: rule engine fallback
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

    if task.case == "Inessive":
        return HuRules.inessive(task.lemma)
    return HuRules.adessive(task.lemma)

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

def new_question():
    st.session_state.feedback = ""
    if st.session_state.df is None:
        st.warning("Upload or load a corpus CSV to begin.")
        return
    kind, payload, solution = next_task(st.session_state.df)
    st.session_state.kind = kind
    st.session_state.current = payload
    st.session_state.solution = solution

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
            case_name = "inessive" if c["case"] == "Inessive" else "adessive"
            st.markdown(
                f"""
                <div class="prompt-card">
                  <div><span class="pill">Noun</span><span class="pill">{case_name}</span></div>
                  <div class="mono" style="font-size:1.25rem;margin-top:.25rem;"><b>{c["lemma"]}</b></div>
                  <div class="muted">Meaning: {c["gloss"]}</div>
                </div>
                """,
                unsafe_allow_html=True
            )

        answer = st.text_input("Type the correct form")
        colA, colB, colC = st.columns([1, 1, 1])
        with colA:
            if st.button("Check"):
                user = normalize_answer(answer, ignore_accents)
                gold = normalize_answer(st.session_state.solution, ignore_accents)
                st.session_state.total += 1
                if user == gold and len(gold) > 0:
                    st.session_state.score += 1
                    st.session_state.feedback = f"<span class='good'>Correct.</span> {st.session_state.solution}"
                else:
                    st.session_state.feedback = f"<span class='bad'>Not quite.</span> Expected: <b>{st.session_state.solution}</b>"
        with colB:
            if allow_reveal and st.button("Reveal"):
                st.session_state.feedback = f"Answer: <b>{st.session_state.solution}</b>"
        with colC:
            if st.button("Next"):
                new_question()

        if st.session_state.feedback:
            st.markdown(st.session_state.feedback, unsafe_allow_html=True)

with colR:
    acc = st.session_state.score
    tot = st.session_state.total
    rate = f"{(100 * acc / tot):.0f}%" if tot else "—"
    st.metric(label="Accuracy", value=rate, delta=f"{acc}/{tot} correct" if tot else "")

    if st.session_state.df is not None:
        st.caption("Corpus loaded and cached for quick sampling.")

# ------------------------- FOOTER AND REFERENCES -------------------------

st.caption(
    "Suffix choices for -ban/-ben and -nál/-nél follow Hungarian vowel harmony and standard case rules. "
    "Definite and indefinite present endings follow standard paradigms with special treatment for -ik verbs. "
    "For maximum accuracy, provide explicit forms in the CSV or enable the NYTK mT5 morphological generator in Advanced settings."
)

