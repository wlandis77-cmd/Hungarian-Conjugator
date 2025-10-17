# -*- coding: utf-8 -*-
import io
import random
import re
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import pandas as pd
import streamlit as st
from gtts import gTTS

st.set_page_config(page_title="Hungarian Conjugator & Declensions", page_icon="🇭🇺", layout="wide")

PASTEL_CSS = """
<style>
:root {
  --bg: #f3f0ff;
  --panel: #e8f6ff;
  --panel-2: #ffeef2;
  --text: #1f2937;
  --accent: #a3e3d8;
  --accent-2: #ffd6a5;
  --accent-3: #cdeac0;
  --muted: #b8c1ec;
}
.stApp, .stApp > div, .block-container {
  background: linear-gradient(180deg, var(--bg), #eef7ff 60%);
}
.stSidebar, .stSelectbox, .stTextInput, .stButton>button, .stAlert, .stDataFrame, .stRadio, .stCheckbox {
  border-radius: 14px !important;
}
section[data-testid="stSidebar"] {
  background: var(--panel);
  border-right: 2px solid #d2e8ff;
}
.stButton>button {
  background: var(--accent);
  color: var(--text);
  border: 1px solid #94dacc;
  box-shadow: 0 2px 6px rgba(0,0,0,0.05);
}
.stButton>button:hover { filter: brightness(0.97); }
button[kind="secondary"] {
  background: var(--accent-2) !important;
  border: 1px solid #ffc794 !important;
}
div[data-baseweb="input"] input {
  background: #faf2ff !important;
}
.stAlert {
  background: var(--panel-2);
  border: 1px solid #ffc7d6;
}
[data-testid="stMetricValue"] {
  color: #2b353e;
}
::selection { background: #cdeac0; color: #1f2937; }
</style>
"""
st.markdown(PASTEL_CSS, unsafe_allow_html=True)

PERSON_LABELS = {
    ("1", "sg"): ("én", "I"),
    ("2", "sg"): ("te", "you (sg)"),
    ("3", "sg"): ("ő", "he/she/it"),
    ("1", "pl"): ("mi", "we"),
    ("2", "pl"): ("ti", "you (pl)"),
    ("3", "pl"): ("ők", "they"),
}

VOWELS_BACK = set("aáoóuú")
VOWELS_FRONT_UNR = set("eéií")
VOWELS_FRONT_R = set("öőüű")

def has_back_vowel(s: str) -> bool:
    return any(ch in VOWELS_BACK for ch in s)

def has_front_vowel(s: str) -> bool:
    return any(ch in VOWELS_FRONT_UNR or ch in VOWELS_FRONT_R for ch in s)

def harmony_pair(stem: str, back: str, front: str) -> str:
    return back if has_back_vowel(stem) else front

def clean_stem(verb: str) -> str:
    return re.sub(r"ni$", "", verb)

IRREGULARS = {
    "eszik": {
        "present_indef": {
            ("1","sg"): "eszem",
            ("2","sg"): "eszel",
            ("3","sg"): "eszik",
            ("1","pl"): "eszünk",
            ("2","pl"): "esztek",
            ("3","pl"): "esznek",
        },
        "past_indef": {
            ("1","sg"): "ettem",
            ("2","sg"): "ettél",
            ("3","sg"): "evett",
            ("1","pl"): "ettünk",
            ("2","pl"): "ettetek",
            ("3","pl"): "ettek",
        },
        "cond_present_indef": {
            ("1","sg"): "ennék",
            ("2","sg"): "ennél",
            ("3","sg"): "enne",
            ("1","pl"): "ennénk",
            ("2","pl"): "ennétek",
            ("3","pl"): "ennének",
        },
        "inf": "enni",
    },
    "iszik": {
        "present_indef": {
            ("1","sg"): "iszom",
            ("2","sg"): "iszol",
            ("3","sg"): "iszik",
            ("1","pl"): "iszunk",
            ("2","pl"): "isztok",
            ("3","pl"): "isznak",
        },
        "past_indef": {
            ("1","sg"): "ittam",
            ("2","sg"): "ittál",
            ("3","sg"): "ivott",
            ("1","pl"): "ittunk",
            ("2","pl"): "ittatok",
            ("3","pl"): "ittak",
        },
        "cond_present_indef": {
            ("1","sg"): "innék",
            ("2","sg"): "innál",
            ("3","sg"): "inne",
            ("1","pl"): "innénk",
            ("2","pl"): "innétek",
            ("3","pl"): "innének",
        },
        "inf": "inni",
    },
    "alszik": {
        "present_indef": {
            ("1","sg"): "alszom",
            ("2","sg"): "alszol",
            ("3","sg"): "alszik",
            ("1","pl"): "alszunk",
            ("2","pl"): "alszotok",
            ("3","pl"): "alszanak",
        },
        "past_indef": {
            ("1","sg"): "aludtam",
            ("2","sg"): "aludtál",
            ("3","sg"): "aludt",
            ("1","pl"): "aludtunk",
            ("2","pl"): "aludtatok",
            ("3","pl"): "aludtak",
        },
        "cond_present_indef": {
            ("1","sg"): "aludnék",
            ("2","sg"): "aludnál",
            ("3","sg"): "aludna",
            ("1","pl"): "aludnánk",
            ("2","pl"): "aludnátok",
            ("3","pl"): "aludnának",
        },
        "inf": "aludni",
    },
}

IRREGULARS_PRESENT_DEF = {
    "eszik": {
        ("1","sg"): "eszem",
        ("2","sg"): "eszed",
        ("3","sg"): "eszi",
        ("1","pl"): "esszük",
        ("2","pl"): "eszitek",
        ("3","pl"): "eszik",
    },
    "iszik": {
        ("1","sg"): "iszom",
        ("2","sg"): "iszod",
        ("3","sg"): "issza",
        ("1","pl"): "isszuk",
        ("2","pl"): "isszátok",
        ("3","pl"): "isszák",
    },
    "alszik": {
        ("1","sg"): "alszom",
        ("2","sg"): "alszod",
        ("3","sg"): "alussza",
        ("1","pl"): "alusszuk",
        ("2","pl"): "alusszátok",
        ("3","pl"): "alusszák",
    },
}

ENDINGS_PRESENT_INDEF = {
    ("1","sg"): ["ok","ek","ök"],
    ("2","sg"): ["sz"],
    ("3","sg"): [""],
    ("1","pl"): ["unk","ünk"],
    ("2","pl"): ["tok","tek","tök"],
    ("3","pl"): ["nak","nek"],
}
ENDINGS_PRESENT_DEF = {
    ("1","sg"): ["om","em","öm"],
    ("2","sg"): ["od","ed","öd"],
    ("3","sg"): ["ja","i"],
    ("1","pl"): ["juk","jük"],
    ("2","pl"): ["játok","itek"],
    ("3","pl"): ["ják","ik"],
}
ENDINGS_PAST_INDEF = {
    ("1","sg"): ["tam","tem"],
    ("2","sg"): ["tál","tél"],
    ("3","sg"): ["ott","ett"],
    ("1","pl"): ["tunk","tünk"],
    ("2","pl"): ["tatok","tetek"],
    ("3","pl"): ["tak","tek"],
}
ENDINGS_PAST_DEF = {
    ("1","sg"): ["tam","tem"],
    ("2","sg"): ["tad","ted"],
    ("3","sg"): ["ta","te"],
    ("1","pl"): ["tuk","tük"],
    ("2","pl"): ["tátok","tétek"],
    ("3","pl"): ["ták","ték"],
}
ENDINGS_COND_INDEF = {
    ("1","sg"): ["nék","nék"],
    ("2","sg"): ["nál","nél"],
    ("3","sg"): ["na","ne"],
    ("1","pl"): ["nánk","nénk"],
    ("2","pl"): ["nátok","nétek"],
    ("3","pl"): ["nának","nének"],
}
ENDINGS_COND_DEF = ENDINGS_COND_INDEF.copy()

FUTURE_FOG = {
    ("1","sg"): "fogok",
    ("2","sg"): "fogsz",
    ("3","sg"): "fog",
    ("1","pl"): "fogunk",
    ("2","pl"): "fogtok",
    ("3","pl"): "fognak",
}

PRONOUN_ORDER = [("1","sg"), ("2","sg"), ("3","sg"), ("1","pl"), ("2","pl"), ("3","pl")]

CASES = {
    "nominative": "",
    "accusative": "t",
    "dative": "nak/nek",
    "inessive": "ban/ben",
    "superessive": "n",
    "adessive": "nál/nél",
    "illative": "ba/be",
    "elative": "ból/ből",
    "allative": "hoz/hez/höz",
    "ablative": "tól/től",
    "possessive_é": "é",
}

def case_suffix(stem: str, case_key: str) -> str:
    def has_back_vowel(s: str) -> bool:
        return any(ch in "aáoóuú" for ch in s)
    def has_front_vowel(s: str) -> bool:
        return any(ch in "eéiíöőüű" for ch in s)
    if case_key == "nominative": return ""
    if case_key == "accusative": return "t"
    if case_key == "dative": return "nak" if has_back_vowel(stem) else "nek"
    if case_key == "inessive": return "ban" if has_back_vowel(stem) else "ben"
    if case_key == "superessive": return "n"
    if case_key == "adessive": return "nál" if has_back_vowel(stem) else "nél"
    if case_key == "illative": return "ba" if has_back_vowel(stem) else "be"
    if case_key == "elative": return "ból" if has_back_vowel(stem) else "ből"
    if case_key == "allative":
        if has_back_vowel(stem): return "hoz"
        return "höz" if any(ch in "öőüű" for ch in stem) else "hez"
    if case_key == "ablative": return "tól" if has_back_vowel(stem) else "től"
    if case_key == "possessive_é": return "é"
    return ""

def join_stem_suffix(stem: str, suffix: str) -> str:
    if not suffix: return stem
    return stem + suffix

@dataclass
class VerbSlot:
    tense: str
    definiteness: str
    person: str
    number: str

def present_indef_regular(stem: str, slot: VerbSlot) -> str:
    if slot.person == "3" and slot.number == "sg":
        return stem
    if slot.person == "2" and slot.number == "sg":
        return stem + "sz"
    if slot.person == "1" and slot.number == "sg":
        return stem + ("ok" if any(ch in "aáoóuú" for ch in stem) else ("ek" if any(ch in "eéií" for ch in stem) else "ök"))
    if slot.person == "1" and slot.number == "pl":
        return stem + ("unk" if any(ch in "aáoóuú" for ch in stem) else "ünk")
    if slot.person == "2" and slot.number == "pl":
        if any(ch in "aáoóuú" for ch in stem): return stem + "tok"
        if any(ch in "öőüű" for ch in stem): return stem + "tök"
        return stem + "tek"
    if slot.person == "3" and slot.number == "pl":
        return stem + ("nak" if any(ch in "aáoóuú" for ch in stem) else "nek")
    return stem

def present_def_regular(stem: str, slot: VerbSlot) -> str:
    back = any(ch in "aáoóuú" for ch in stem)
    front_only = not back
    if slot.person == "1" and slot.number == "sg":
        return stem + ("om" if back else ("em" if any(ch in "eéií" for ch in stem) else "öm"))
    if slot.person == "2" and slot.number == "sg":
        return stem + ("od" if back else ("ed" if any(ch in "eéií" for ch in stem) else "öd"))
    if slot.person == "3" and slot.number == "sg":
        return stem + ("ja" if back else "i")
    if slot.person == "1" and slot.number == "pl":
        return stem + ("juk" if back else "jük")
    if slot.person == "2" and slot.number == "pl":
        return stem + ("játok" if back else "itek")
    if slot.person == "3" and slot.number == "pl":
        return stem + ("ják" if back else "ik")
    return stem

def past_regular(stem: str, slot: VerbSlot) -> str:
    back = any(ch in "aáoóuú" for ch in stem)
    endings = ENDINGS_PAST_DEF if slot.definiteness == "definite" else ENDINGS_PAST_INDEF
    a, b = endings[(slot.person, slot.number)]
    return stem + (a if back else b)

def cond_present_regular(stem: str, slot: VerbSlot) -> str:
    back = any(ch in "aáoóuú" for ch in stem)
    endings = ENDINGS_COND_DEF if slot.definiteness == "definite" else ENDINGS_COND_INDEF
    a, b = endings[(slot.person, slot.number)]
    base = stem + (a if back else b)
    if slot.definiteness == "definite" and slot.person == "3" and slot.number == "sg":
        return stem + ("nája" if back else "néje")
    return base

def future_with_fog(lemma: str, slot: VerbSlot) -> str:
    base_inf = IRREGULARS.get(lemma, {}).get("inf", None)
    inf = base_inf if base_inf else (lemma if lemma.endswith("ni") else lemma + "ni")
    return f"{inf} {FUTURE_FOG[(slot.person, slot.number)]}"

def conj(lemma: str, slot: VerbSlot) -> str:
    l = lemma.strip().lower()
    if l in IRREGULARS:
        if slot.tense == "present" and slot.definiteness == "indefinite":
            return IRREGULARS[l]["present_indef"][(slot.person, slot.number)]
        if slot.tense == "present" and slot.definiteness == "definite":
            return IRREGULARS_PRESENT_DEF.get(l, {}).get((slot.person, slot.number), present_def_regular(clean_stem(l), slot))
        if slot.tense == "past":
            irregular = IRREGULARS[l].get("past_indef", None)
            if irregular and slot.definiteness == "indefinite":
                return irregular[(slot.person, slot.number)]
            return past_regular(clean_stem(l), slot)
        if slot.tense == "conditional":
            irregular = IRREGULARS[l].get("cond_present_indef", None)
            if irregular and slot.definiteness == "indefinite":
                return irregular[(slot.person, slot.number)]
            return cond_present_regular(clean_stem(l), slot)
        if slot.tense == "future":
            return future_with_fog(l, slot)

    stem = clean_stem(l)
    if slot.tense == "present":
        return present_indef_regular(stem, slot) if slot.definiteness == "indefinite" else present_def_regular(stem, slot)
    if slot.tense == "past":
        return past_regular(stem, slot)
    if slot.tense == "conditional":
        return cond_present_regular(stem, slot)
    if slot.tense == "future":
        return future_with_fog(l, slot)
    return stem

def init_state():
    ss = st.session_state
    ss.setdefault("score", 0)
    ss.setdefault("total", 0)
    ss.setdefault("current_task", None)
    ss.setdefault("checked", False)
    ss.setdefault("verblist", [])
    ss.setdefault("nounlist", [])

init_state()

st.sidebar.header("Practice settings")
mode = st.sidebar.radio("Practice", ["Verb conjugations", "Noun declensions"])

tense = st.sidebar.selectbox("Tense", ["present", "past", "conditional", "future"])
definiteness = st.sidebar.selectbox("Definiteness", ["indefinite", "definite"]) if mode == "Verb conjugations" else "indefinite"

case_keys = list(CASES.keys())
selected_cases = st.sidebar.multiselect("Cases to practice", case_keys, default=["nominative","accusative","dative","inessive","adessive","illative","elative","allative","ablative","superessive","possessive_é"])

st.sidebar.subheader("Upload CSVs")
verbs_csv = st.sidebar.file_uploader("verbs.csv", type=["csv"])
nouns_csv = st.sidebar.file_uploader("nouns.csv", type=["csv"])

if verbs_csv:
    try:
        dfv = pd.read_csv(verbs_csv)
        st.session_state.verblist = list(dfv.fillna("").itertuples(index=False, name=None))
        st.sidebar.success(f"Loaded {len(st.session_state.verblist)} verbs")
    except Exception as e:
        st.sidebar.error(f"Verb CSV error: {e}")

if nouns_csv:
    try:
        dfn = pd.read_csv(nouns_csv)
        st.session_state.nounlist = list(dfn.fillna("").itertuples(index=False, name=None))
        st.sidebar.success(f"Loaded {len(st.session_state.nounlist)} nouns")
    except Exception as e:
        st.sidebar.error(f"Noun CSV error: {e}")

enable_tts = st.sidebar.toggle("Play prompt TTS", value=False)
voice_lang = st.sidebar.selectbox("TTS language code", ["hu","en"], index=0)

def next_task(mode, tense, definiteness, selected_cases):
    if mode == "Verb conjugations" and st.session_state.verblist:
        verb_root, meaning = random.choice(st.session_state.verblist)
        person, number = random.choice([("1","sg"),("2","sg"),("3","sg"),("1","pl"),("2","pl"),("3","pl")])
        slot = VerbSlot(tense=tense, definiteness=definiteness, person=person, number=number)
        expected = conj(verb_root, slot)
        pron_hu, pron_en = PERSON_LABELS[(person, number)]
        prompt = f"Conjugate “{verb_root}” ({meaning}) in {slot.tense} {slot.definiteness}, for {pron_hu}."
        return {"kind":"verb","lemma":verb_root,"meaning":meaning,"slot":slot,"expected":expected,"prompt":prompt,"pron_hu":pron_hu,"pron_en":pron_en}
    if mode == "Noun declensions" and st.session_state.nounlist:
        noun, meaning = random.choice(st.session_state.nounlist)
        case_key = random.choice(selected_cases) if selected_cases else "nominative"
        suf = case_suffix(noun, case_key)
        expected = join_stem_suffix(noun, suf)
        prompt = f"Decline “{noun}” ({meaning}) in case: {case_key.replace('_',' ')}."
        return {"kind":"noun","noun":noun,"meaning":meaning,"case_key":case_key,"expected":expected,"prompt":prompt}
    return None

def set_new_task():
    st.session_state.current_task = next_task(mode, tense, definiteness, selected_cases)
    st.session_state.checked = False
    st.session_state.last_answer = ""

if st.session_state.current_task is None:
    set_new_task()

left, right = st.columns([2,1])

with left:
    st.title("Hungarian Practice")
    task = st.session_state.current_task
    if not task:
        st.info("Upload at least one CSV to begin.")
    else:
        st.write(task["prompt"])
        if task["kind"] == "verb":
            meta = f"Tense: {task['slot'].tense}, Definiteness: {task['slot'].definiteness}, Pronoun: {task['pron_hu']} ({task['pron_en']})"
            st.caption(meta)
        else:
            st.caption(f"Case: {task['case_key']}")

        if enable_tts:
            try:
                tts_text = task["prompt"]
                mp3 = io.BytesIO()
                gTTS(text=tts_text, lang=voice_lang, slow=False).write_to_fp(mp3)
                st.audio(mp3.getvalue(), format="audio/mp3")
            except Exception as e:
                st.warning(f"TTS error: {e}")

        answer = st.text_input("Type the correct form", value=st.session_state.get("last_answer",""), disabled=st.session_state.checked, key="answer_input")

        c1, c2 = st.columns([1,1])
        with c1:
            check_clicked = st.button("Check", disabled=st.session_state.checked or task is None, type="primary")
        with c2:
            next_clicked = st.button("Next", type="secondary")

        if check_clicked and not st.session_state.checked:
            st.session_state.checked = True
            st.session_state.last_answer = answer.strip()
            expected = task["expected"]
            given = answer.strip().lower()
            canon = lambda s: re.sub(r"[\\s\\-]+","", s.lower())
            if canon(given) == canon(expected):
                st.success("Correct!")
                st.session_state.score += 1
            else:
                st.error(f"Not quite. Expected: {expected}")
            st.session_state.total += 1

        if next_clicked:
            set_new_task()
            st.rerun()

with right:
    st.subheader("Progress")
    st.metric("Score", f"{st.session_state.score}/{st.session_state.total}")
    if st.session_state.total > 0:
        pct = 100 * st.session_state.score / st.session_state.total
        st.caption(f"Accuracy: {pct:.1f}%")
    st.divider()
    st.subheader("Now practicing")
    st.write(f"Mode: **{mode}**")
    if mode == "Verb conjugations":
        st.write(f"Tense: **{tense}**  |  Definiteness: **{definiteness}**")
    else:
        st.write("Cases: " + ", ".join(selected_cases) if selected_cases else "nominative")
    st.divider()
    st.subheader("Notes")
    st.write("Futures are built with **fog** + infinitive. Irregulars **eszik/iszik/alszik** have targeted overrides. The prompt's person (te vs ti) is exactly the slot used for evaluation, fixing the earlier mismatch.")
