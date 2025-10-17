# Set safe defaults before importing Streamlit.
import os
os.environ.setdefault("STREAMLIT_SERVER_FILE_WATCHER_TYPE", "none")
os.environ.setdefault("STREAMLIT_SERVER_RUN_ON_SAVE", "false")

import random
import re
import unicodedata
from dataclasses import dataclass
from io import BytesIO
from typing import Dict, List, Optional, Tuple

import pandas as pd
import streamlit as st

# Optional GitHub integration (for lemma lists only)
try:
    from github import Github  # pip install PyGithub
    _GITHUB_OK = True
except Exception:
    _GITHUB_OK = False

# Optional AI TTS backends
_HAS_GTTS = False
try:
    from gtts import gTTS
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


# ============================ UI CHROME ============================

st.set_page_config(page_title="Hungarian FST Trainer", page_icon="🇭🇺", layout="wide")

st.markdown(
    """
    <style>
    :root{
      --bg1:#f6f0ff; --bg2:#e9f7ff; --surface:#eef2ff; --surface2:#e7f7f4;
      --ink:#1f2937; --muted:#4b5563; --accent:#a7c8ff; --accent-ink:#0f172a;
      --border:#cfd8ee; --pill:#d6efff; --good:#1c8c4e; --bad:#b21b1b;
    }
    [data-testid="stAppViewContainer"]{ background: linear-gradient(135deg,var(--bg1) 0%,var(--bg2) 100%); }
    [data-testid="stSidebar"]{ background: linear-gradient(180deg,#f8eaff 0%,#e6f7ff 100%); border-right: 1px solid var(--border); }
    [data-testid="stHeader"]{ background: transparent; }
    .block-container{ padding-top: 1rem; }
    .prompt-card{ border:1px solid var(--border); padding:1rem 1.25rem; border-radius:12px; background:var(--surface); box-shadow:0 1px 0 rgba(16,24,40,.03); margin-bottom:1rem; }
    .pill{ display:inline-block; font-size:.85rem; padding:.12rem .6rem; border:1px solid var(--border); border-radius:999px; margin-right:.35rem; background: var(--pill); }
    .big-title{ font-size: 1.8rem; font-weight: 700; margin-bottom:.25rem; color:var(--ink); }
    .subtitle{ color: var(--muted); margin-bottom: 1rem; }
    .metric-card{ background:var(--surface2); border:1px solid var(--border); border-radius:12px; padding:.75rem 1rem; }
    .good{ color: var(--good); font-weight: 700; }
    .bad{ color: var(--bad); font-weight: 700; }
    .muted{ color: var(--muted); }
    .mono{ font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, "Liberation Mono", monospace; }
    </style>
    """,
    unsafe_allow_html=True,
)
st.markdown('<div class="big-title">Hungarian Conjugations and Declensions</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">Finite-state morphological generator with built‑in irregulars.</div>', unsafe_allow_html=True)


# ============================ DATA & SETTINGS ============================

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
    case: str
    number: str
    ud_key: str

with st.sidebar:
    st.header("Settings")
    source = st.radio("Corpus source", ["Upload CSV", "Load from GitHub"])

    # Persist important settings in session_state
    defaults = dict(
        want_verbs=True, want_nouns=True, ignore_accents=True, show_hu_pronouns=True,
        allow_reveal=True, tts_provider="Off", tts_rate=1.0, auto_say_answer=True
    )
    for k, v in defaults.items():
        st.session_state.setdefault(k, v)

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
                return pd.read_csv(BytesIO(f.decoded_content))
            except Exception as e:
                st.error(f"GitHub load failed: {e if '404' not in str(e) else 'file or ref not found'}")
                return None
        if st.button("Load CSV from GitHub"):
            df = load_from_github()
    else:
        up = st.file_uploader("Upload corpus CSV", type=["csv"])
        if up is not None:
            try:
                df = pd.read_csv(up)
            except Exception as e:
                st.error(f"CSV parse failed: {e}")

    st.divider()

    st.subheader("Practice scope")
    st.session_state.want_verbs = st.checkbox("Verbs", value=st.session_state.want_verbs)
    st.session_state.want_nouns = st.checkbox("Nouns", value=st.session_state.want_nouns)

    VERB_MODE_OPTIONS = [
        "Present Indefinite", "Present Definite",
        "Past Indefinite", "Past Definite",
        "Conditional Present Indefinite", "Conditional Present Definite",
        "Future Indefinite", "Future Definite",
    ]
    verb_modes = st.multiselect("Select verb modes", VERB_MODE_OPTIONS, default=VERB_MODE_OPTIONS)

    NOUN_CASE_OPTIONS = ["Nominative","Accusative","Dative","Inessive","Superessive","Adessive","Illative","Sublative","Allative","Instrumental","Genitive"]
    noun_modes = st.multiselect("Cases", NOUN_CASE_OPTIONS, default=[
        "Accusative","Dative","Inessive","Superessive","Adessive","Illative","Sublative","Allative","Instrumental","Genitive"
    ])
    noun_numbers = st.multiselect("Noun number", ["Singular","Plural"], default=["Singular","Plural"])

    st.divider()
    adv = st.expander("Answer checking")
    with adv:
        st.session_state.ignore_accents = st.checkbox("Accept answers that ignore accents", value=st.session_state.ignore_accents)
        st.session_state.show_hu_pronouns = st.checkbox("Show Hungarian pronouns for verb prompts", value=st.session_state.show_hu_pronouns)
        st.session_state.allow_reveal = st.checkbox("Allow Reveal Answer", value=st.session_state.allow_reveal)

    st.divider()
    tts_exp = st.expander("Pronunciation (AI TTS)")
    with tts_exp:
        st.session_state.tts_provider = st.selectbox("TTS provider", ["Off","gTTS (local, free)","Google Cloud TTS"], index=["Off","gTTS (local, free)","Google Cloud TTS"].index(st.session_state.tts_provider))
        st.session_state.tts_rate = st.slider("Speaking rate", 0.6, 1.4, float(st.session_state.tts_rate), 0.05)
        st.session_state.auto_say_answer = st.checkbox("Auto speak correct answer on Reveal or when correct", value=st.session_state.auto_say_answer)
        st.caption("For Google Cloud, add a service account JSON to Streamlit secrets as GOOGLE_TTS_SERVICE_ACCOUNT_JSON. hu-HU voice.")

        if st.button('Test TTS with "Szia!"'):
            audio = None
            try:
                if st.session_state.tts_provider.startswith("gTTS"):
                    if not _HAS_GTTS:
                        st.error("gTTS is not installed.")
                    else:
                        g = gTTS("Szia!", lang="hu")
                        buf = BytesIO(); g.write_to_fp(buf); buf.seek(0); audio = buf.read()
                elif st.session_state.tts_provider.startswith("Google"):
                    if not _HAS_GOOGLE_TTS:
                        st.error("google-cloud-texttospeech is not installed.")
                    else:
                        sa = st.secrets.get("GOOGLE_TTS_SERVICE_ACCOUNT_JSON", None)
                        creds = service_account.Credentials.from_service_account_info(sa) if isinstance(sa, dict) else None
                        client = texttospeech.TextToSpeechClient(credentials=creds) if creds else texttospeech.TextToSpeechClient()
                        inp = texttospeech.SynthesisInput(text="Szia!")
                        voice = texttospeech.VoiceSelectionParams(language_code="hu-HU")
                        cfg = texttospeech.AudioConfig(audio_encoding=texttospeech.AudioEncoding.MP3, speaking_rate=float(st.session_state.tts_rate))
                        resp = client.synthesize_speech(input=inp, voice=voice, audio_config=cfg)
                        audio = resp.audio_content
            except Exception as e:
                st.error(f"TTS test failed: {e}")
            if audio:
                st.audio(audio, format="audio/mp3")


# ============================ LEXICON & HARMONY ============================

BACK_VOWELS = set("aáoóuú")
FRONT_UNR = set("eéií")
FRONT_R = set("öőüű")
ALL_VOWELS = BACK_VOWELS | FRONT_UNR | FRONT_R

def has_back(s: str) -> bool:
    return any(ch in BACK_VOWELS for ch in s.lower())

def has_front_rounded(s: str) -> bool:
    return any(ch in FRONT_R for ch in s.lower())

def last_vowel(s: str) -> Optional[str]:
    lv = None
    for ch in s.lower():
        if ch in ALL_VOWELS:
            lv = ch
    return lv

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
    s = unicodedata.normalize("NFC", s or "").strip()
    if strip_accents:
        s = s.translate(ACCENT_STRIP)
    return s.lower()

def is_ik_verb(lemma: str, csv_flag: Optional[bool]) -> bool:
    if isinstance(csv_flag, bool):
        return csv_flag
    return lemma.endswith("ik")

def infinitive(lemma: str) -> str:
    base = lemma[:-2] if lemma.endswith("ik") else lemma
    return base + "ni"


# ============================ FINITE-STATE CORE ============================

class FST:
    """A tiny deterministic FST for morpheme-level generation.
    Arcs are labeled with (inp, out). We feed a sequence of input symbols like ["STEM","PRES","IND","INDEF","P2","SG"]
    and receive one output string (surface form). Epsilon input is "".
    """
    def __init__(self):
        self.start = 0
        self.final = set([0])
        self.arcs: Dict[int, List[Tuple[str, str, int]]] = {0: []}

    def add_state(self) -> int:
        sid = len(self.arcs)
        self.arcs[sid] = []
        return sid

    def add_arc(self, s_from: int, s_to: int, inp: str, out: str):
        self.arcs[s_from].append((inp, out, s_to))

    def set_final(self, s: int):
        self.final.add(s)

    def transduce(self, inputs: List[str]) -> str:
        """Run through the machine consuming inputs; follow matching arcs by input label or epsilon.
        Preference: consume all possible epsilons before consuming the next input symbol.
        """
        state = self.start
        out = []

        i = 0
        while True:
            progressed = True
            # epsilon closure
            while progressed:
                progressed = False
                for (inp, o, nxt) in self.arcs.get(state, []):
                    if inp == "":
                        out.append(o)
                        state = nxt
                        progressed = True
                        break

            if i >= len(inputs):
                break

            sym = inputs[i]
            moved = False
            for (inp, o, nxt) in self.arcs.get(state, []):
                if inp == sym:
                    out.append(o)
                    state = nxt
                    i += 1
                    moved = True
                    break
            if not moved:
                # dead end, reject
                return ""
        # accept if final is reachable via epsilons too
        # consume trailing epsilons
        progressed = True
        while progressed:
            progressed = False
            for (inp, o, nxt) in self.arcs.get(state, []):
                if inp == "":
                    out.append(o); state = nxt; progressed = True; break
        return "".join(out) if state in self.final else ""


# ============================ FST BUILDERS ============================

def build_present_indef_fst(lemma: str, is_ik: bool) -> FST:
    h = harmony_set(lemma)
    v_ok = {"back":"ok","front_unr":"ek","front_r":"ök"}[h]
    v_1pl = {"back":"unk","front_unr":"ünk","front_r":"ünk"}[h]
    v_2pl = {"back":"tok","front_unr":"tek","front_r":"tök"}[h]
    v_3pl = {"back":"nak","front_unr":"nek","front_r":"nek"}[h]
    link = {"back":"ol","front_unr":"el","front_r":"öl"}[h]

    fst = FST()
    s1 = fst.add_state(); s2 = fst.add_state(); s3 = fst.add_state(); s4 = fst.add_state(); s5 = fst.add_state(); s6 = fst.add_state()
    fst.set_final(s1); fst.set_final(s2); fst.set_final(s3); fst.set_final(s4); fst.set_final(s5); fst.set_final(s6)

    # Base emission: lemma or -ik base for certain slots
    base = lemma[:-2] if is_ik and lemma.endswith("ik") else lemma

    # 1SG
    fst.add_arc(0, s1, "P1SG", base + ({"back":"om","front_unr":"em","front_r":"öm"}[h] if is_ik else v_ok))
    # 2SG
    if re.search(r"(s|z|sz|zs)$", lemma):
        fst.add_arc(0, s2, "P2SG", lemma + link)
    else:
        fst.add_arc(0, s2, "P2SG", lemma + "sz")
    # 3SG
    fst.add_arc(0, s3, "P3SG", (base + "ik") if is_ik else lemma)
    # 1PL
    fst.add_arc(0, s4, "P1PL", lemma + v_1pl)
    # 2PL
    fst.add_arc(0, s5, "P2PL", lemma + v_2pl)
    # 3PL
    fst.add_arc(0, s6, "P3PL", lemma + v_3pl)
    return fst

def build_present_def_fst(lemma: str) -> FST:
    h = harmony_set(lemma)
    v_1sg = {"back":"om","front_unr":"em","front_r":"öm"}[h]
    v_2sg = {"back":"od","front_unr":"ed","front_r":"öd"}[h]
    v_1pl = {"back":"juk","front_unr":"jük","front_r":"jük"}[h]
    v_2pl_cons = {"back":"játok","front_unr":"itek","front_r":"itek"}[h]
    v_2pl_vow  = {"back":"játok","front_unr":"jétek","front_r":"jétek"}[h]
    v_3pl_def  = {"back":"ják","front_unr":"ik","front_r":"ik"}[h]
    ends_vowel = lemma[-1].lower() in ALL_VOWELS

    fst = FST()
    s1=s2=s3=s4=s5=s6=None
    s1=fst.add_state(); s2=fst.add_state(); s3=fst.add_state(); s4=fst.add_state(); s5=fst.add_state(); s6=fst.add_state()
    for s in [s1,s2,s3,s4,s5,s6]: fst.set_final(s)

    fst.add_arc(0, s1, "P1SG", lemma + v_1sg)
    fst.add_arc(0, s2, "P2SG", lemma + v_2sg)

    # 3SG
    if re.search(r"(z)$", lemma):
        fst.add_arc(0, s3, "P3SG", lemma + "i")
    elif re.search(r"(s|sz|zs)$", lemma):
        if lemma.endswith("sz"):
            base = lemma[:-2] + "ssz"
        elif lemma.endswith("zs"):
            base = lemma[:-2] + "zzs"
        elif lemma.endswith("s"):
            base = lemma[:-1] + "ss"
        else:
            base = lemma
        fst.add_arc(0, s3, "P3SG", base + ("a" if h=="back" else "e"))
    else:
        fst.add_arc(0, s3, "P3SG", lemma + ("ja" if h=="back" else "je"))

    fst.add_arc(0, s4, "P1PL", lemma + v_1pl)
    fst.add_arc(0, s5, "P2PL", lemma + (v_2pl_vow if ends_vowel else v_2pl_cons))

    if re.search(r"(z|sz|zs)$", lemma):
        fst.add_arc(0, s6, "P3PL", lemma + "ik")
    else:
        fst.add_arc(0, s6, "P3PL", lemma + v_3pl_def)
    return fst

def build_past_indef_fst(lemma: str) -> FST:
    h = harmony_set(lemma)
    stem = lemma
    past_mark = "tt" if stem[-1].lower() in ALL_VOWELS else "t"
    v_a = "a" if h == "back" else "e"
    v_1pl = "unk" if h == "back" else "ünk"

    fst = FST()
    s1=s2=s3=s4=s5=s6=None
    s1=fst.add_state(); s2=fst.add_state(); s3=fst.add_state(); s4=fst.add_state(); s5=fst.add_state(); s6=fst.add_state()
    for s in [s1,s2,s3,s4,s5,s6]: fst.set_final(s)

    fst.add_arc(0, s1, "P1SG", stem + past_mark + v_a + "m")
    fst.add_arc(0, s2, "P2SG", stem + past_mark + ("ál" if h=="back" else "él"))
    fst.add_arc(0, s3, "P3SG", stem + past_mark)
    fst.add_arc(0, s4, "P1PL", stem + past_mark + v_1pl)
    fst.add_arc(0, s5, "P2PL", stem + past_mark + ("atok" if h=="back" else "etek"))
    fst.add_arc(0, s6, "P3PL", stem + past_mark + ("ak" if h=="back" else "ek"))
    return fst

def build_past_def_fst(lemma: str) -> FST:
    h = harmony_set(lemma)
    stem = lemma
    past_mark = "tt" if stem[-1].lower() in ALL_VOWELS else "t"
    v_a = "a" if h == "back" else "e"
    v_á = "á" if h == "back" else "é"

    fst = FST()
    s1=s2=s3=s4=s5=s6=None
    s1=fst.add_state(); s2=fst.add_state(); s3=fst.add_state(); s4=fst.add_state(); s5=fst.add_state(); s6=fst.add_state()
    for s in [s1,s2,s3,s4,s5,s6]: fst.set_final(s)

    fst.add_arc(0, s1, "P1SG", stem + past_mark + v_a + "m")
    fst.add_arc(0, s2, "P2SG", stem + past_mark + v_a + "d")
    fst.add_arc(0, s3, "P3SG", stem + past_mark + v_a)
    fst.add_arc(0, s4, "P1PL", stem + past_mark + ("uk" if h=="back" else "ük"))
    fst.add_arc(0, s5, "P2PL", stem + past_mark + v_á + ("tok" if h=="back" else "tek"))
    fst.add_arc(0, s6, "P3PL", stem + past_mark + v_á + "k")
    return fst

def build_cond_indef_fst(lemma: str) -> FST:
    h = harmony_set(lemma)
    fst = FST(); s=[fst.add_state() for _ in range(6)]
    for t in s: fst.set_final(t)
    fst.add_arc(0, s[0], "P1SG", lemma + "nék")
    fst.add_arc(0, s[1], "P2SG", lemma + ("nál" if h=="back" else "nél"))
    fst.add_arc(0, s[2], "P3SG", lemma + ("na" if h=="back" else "ne"))
    fst.add_arc(0, s[3], "P1PL", lemma + ("nánk" if h=="back" else "nénk"))
    fst.add_arc(0, s[4], "P2PL", lemma + ("nátok" if h=="back" else "nétek"))
    fst.add_arc(0, s[5], "P3PL", lemma + ("nának" if h=="back" else "nének"))
    return fst

def build_cond_def_fst(lemma: str) -> FST:
    h = harmony_set(lemma)
    fst = FST(); s=[fst.add_state() for _ in range(6)]
    for t in s: fst.set_final(t)
    fst.add_arc(0, s[0], "P1SG", lemma + ("nám" if h=="back" else "ném"))
    fst.add_arc(0, s[1], "P2SG", lemma + ("nád" if h=="back" else "néd"))
    fst.add_arc(0, s[2], "P3SG", lemma + ("ná" if h=="back" else "né"))
    fst.add_arc(0, s[3], "P1PL", lemma + ("nánk" if h=="back" else "nénk"))
    fst.add_arc(0, s[4], "P2PL", lemma + ("nátok" if h=="back" else "nétek"))
    fst.add_arc(0, s[5], "P3PL", lemma + ("nák" if h=="back" else "nék"))
    return fst

# Future is analytic with "fog"
def future_form(definite: bool, person: int, number: str, lemma: str) -> str:
    def_ind = {
        ("Sing",1): ("fogok","fogom"),
        ("Sing",2): ("fogsz","fogod"),
        ("Sing",3): ("fog","fogja"),
        ("Plur",1): ("fogunk","fogjuk"),
        ("Plur",2): ("fogtok","fogjátok"),
        ("Plur",3): ("fognak","fogják"),
    }[(number,person)]
    aux = def_ind[1] if definite else def_ind[0]
    return f"{aux} {infinitive(lemma)}"


# ============================ IRREGULAR LEXICON ============================

IRREGULARS: Dict[str, Dict[Tuple[str,str,bool,int,str], str]] = {}
def add_irregular(lemma: str, forms: Dict[Tuple[str,str,bool,int,str], str]): IRREGULARS[lemma] = forms

# lenni (van)
l={}
for d in [False,True]:
    l[("Pres","Ind",d,1,"Sing")] = "vagyok"
    l[("Pres","Ind",d,2,"Sing")] = "vagy"
    l[("Pres","Ind",d,3,"Sing")] = "van"
    l[("Pres","Ind",d,1,"Plur")] = "vagyunk"
    l[("Pres","Ind",d,2,"Plur")] = "vagytok"
    l[("Pres","Ind",d,3,"Plur")] = "vannak"
    l[("Past","Ind",d,1,"Sing")] = "voltam"
    l[("Past","Ind",d,2,"Sing")] = "voltál"
    l[("Past","Ind",d,3,"Sing")] = "volt"
    l[("Past","Ind",d,1,"Plur")] = "voltunk"
    l[("Past","Ind",d,2,"Plur")] = "voltatok"
    l[("Past","Ind",d,3,"Plur")] = "voltak"
    l[("Pres","Cnd",d,1,"Sing")] = "lennék"
    l[("Pres","Cnd",d,2,"Sing")] = "lennél"
    l[("Pres","Cnd",d,3,"Sing")] = "lenne"
    l[("Pres","Cnd",d,1,"Plur")] = "lennénk"
    l[("Pres","Cnd",d,2,"Plur")] = "lennétek"
    l[("Pres","Cnd",d,3,"Plur")] = "lennének"
add_irregular("van", l)

# megy, jön, eszik, iszik, tesz, vesz, hisz, visz, hoz
def add_sz_irregular(base, pres_1sg, pres_2sg, pres_3sg, pres_1pl, pres_2pl, pres_3pl, past_3sg_stem, cond_stem):
    f={}
    for d in [False,True]:
        f[("Pres","Ind",d,1,"Sing")] = pres_1sg
        f[("Pres","Ind",d,2,"Sing")] = pres_2sg
        f[("Pres","Ind",d,3,"Sing")] = pres_3sg
        f[("Pres","Ind",d,1,"Plur")] = pres_1pl
        f[("Pres","Ind",d,2,"Plur")] = pres_2pl
        f[("Pres","Ind",d,3,"Plur")] = pres_3pl
        f[("Past","Ind",d,1,"Sing")] = past_3sg_stem + "em"
        f[("Past","Ind",d,2,"Sing")] = past_3sg_stem + "ed"
        f[("Past","Ind",d,3,"Sing")] = past_3sg_stem
        f[("Past","Ind",d,1,"Plur")] = past_3sg_stem + "ük"
        f[("Past","Ind",d,2,"Plur")] = past_3sg_stem + "étek"
        f[("Past","Ind",d,3,"Plur")] = past_3sg_stem + "ék"
        f[("Pres","Cnd",d,1,"Sing")] = cond_stem + "nék"
        f[("Pres","Cnd",d,2,"Sing")] = cond_stem + "nél"
        f[("Pres","Cnd",d,3,"Sing")] = cond_stem + "ne"
        f[("Pres","Cnd",d,1,"Plur")] = cond_stem + "nénk"
        f[("Pres","Cnd",d,2,"Plur")] = cond_stem + "nétek"
        f[("Pres","Cnd",d,3,"Plur")] = cond_stem + "nének"
    add_irregular(base, f)

# Populate irregulars
# megy
m={}
for d in [False,True]:
    m[("Pres","Ind",d,1,"Sing")]="megyek"; m[("Pres","Ind",d,2,"Sing")]="mész"; m[("Pres","Ind",d,3,"Sing")]="megy"
    m[("Pres","Ind",d,1,"Plur")]="megyünk"; m[("Pres","Ind",d,2,"Plur")]="mentek"; m[("Pres","Ind",d,3,"Plur")]="mennek"
    m[("Past","Ind",d,1,"Sing")]="mentem"; m[("Past","Ind",d,2,"Sing")]="mentél"; m[("Past","Ind",d,3,"Sing")]="ment"
    m[("Past","Ind",d,1,"Plur")]="mentünk"; m[("Past","Ind",d,2,"Plur")]="mentetek"; m[("Past","Ind",d,3,"Plur")]="mentek"
    m[("Pres","Cnd",d,1,"Sing")]="mennék"; m[("Pres","Cnd",d,2,"Sing")]="mennél"; m[("Pres","Cnd",d,3,"Sing")]="menne"
    m[("Pres","Cnd",d,1,"Plur")]="mennénk"; m[("Pres","Cnd",d,2,"Plur")]="mennétek"; m[("Pres","Cnd",d,3,"Plur")]="mennének"
add_irregular("megy", m)
# jön
j={}
for d in [False,True]:
    j[("Pres","Ind",d,1,"Sing")]="jövök"; j[("Pres","Ind",d,2,"Sing")]="jössz"; j[("Pres","Ind",d,3,"Sing")]="jön"
    j[("Pres","Ind",d,1,"Plur")]="jövünk"; j[("Pres","Ind",d,2,"Plur")]="jöttök"; j[("Pres","Ind",d,3,"Plur")]="jönnek"
    j[("Past","Ind",d,1,"Sing")]="jöttem"; j[("Past","Ind",d,2,"Sing")]="jöttél"; j[("Past","Ind",d,3,"Sing")]="jött"
    j[("Past","Ind",d,1,"Plur")]="jöttünk"; j[("Past","Ind",d,2,"Plur")]="jöttetek"; j[("Past","Ind",d,3,"Plur")]="jöttek"
    j[("Pres","Cnd",d,1,"Sing")]="jönnék"; j[("Pres","Cnd",d,2,"Sing")]="jönnél"; j[("Pres","Cnd",d,3,"Sing")]="jönne"
    j[("Pres","Cnd",d,1,"Plur")]="jönnénk"; j[("Pres","Cnd",d,2,"Plur")]="jönnétek"; j[("Pres","Cnd",d,3,"Plur")]="jönnének"
add_irregular("jön", j)
# eszik
e={}
for d in [False,True]:
    e[("Pres","Ind",d,1,"Sing")]="eszem"; e[("Pres","Ind",d,2,"Sing")]="eszel"; e[("Pres","Ind",d,3,"Sing")]="eszik"
    e[("Pres","Ind",d,1,"Plur")]="eszünk"; e[("Pres","Ind",d,2,"Plur")]="esztek"; e[("Pres","Ind",d,3,"Plur")]="esznek"
    e[("Past","Ind",d,1,"Sing")]="ettem"; e[("Past","Ind",d,2,"Sing")]="ettél"; e[("Past","Ind",d,3,"Sing")]="evett"
    e[("Past","Ind",d,1,"Plur")]="ettünk"; e[("Past","Ind",d,2,"Plur")]="ettetek"; e[("Past","Ind",d,3,"Plur")]="ettek"
    e[("Pres","Cnd",d,1,"Sing")]="ennék"; e[("Pres","Cnd",d,2,"Sing")]="ennél"; e[("Pres","Cnd",d,3,"Sing")]="enne"
    e[("Pres","Cnd",d,1,"Plur")]="ennénk"; e[("Pres","Cnd",d,2,"Plur")]="ennétek"; e[("Pres","Cnd",d,3,"Plur")]="ennének"
add_irregular("eszik", e)
# iszik
i={}
for d in [False,True]:
    i[("Pres","Ind",d,1,"Sing")]="iszom"; i[("Pres","Ind",d,2,"Sing")]="iszol"; i[("Pres","Ind",d,3,"Sing")]="iszik"
    i[("Pres","Ind",d,1,"Plur")]="iszunk"; i[("Pres","Ind",d,2,"Plur")]="isztok"; i[("Pres","Ind",d,3,"Plur")]="isznak"
    i[("Past","Ind",d,1,"Sing")]="ittam"; i[("Past","Ind",d,2,"Sing")]="ittál"; i[("Past","Ind",d,3,"Sing")]="ivott"
    i[("Past","Ind",d,1,"Plur")]="ittunk"; i[("Past","Ind",d,2,"Plur")]="ittatok"; i[("Past","Ind",d,3,"Plur")]="ittak"
    i[("Pres","Cnd",d,1,"Sing")]="innék"; i[("Pres","Cnd",d,2,"Sing")]="innál"; i[("Pres","Cnd",d,3,"Sing")]="inna"
    i[("Pres","Cnd",d,1,"Plur")]="innánk"; i[("Pres","Cnd",d,2,"Plur")]="innátok"; i[("Pres","Cnd",d,3,"Plur")]="innának"
add_irregular("iszik", i)
# tesz, vesz, hisz, visz, hoz
add_sz_irregular("tesz","teszek","teszel","tesz","teszünk","tesztek","tesznek","tett","ten")
add_sz_irregular("vesz","veszek","veszel","vesz","veszünk","vesztek","vesznek","vett","ven")
add_sz_irregular("hisz","hiszek","hiszel","hisz","hiszünk","hisztek","hisznek","hitt","hin")
add_sz_irregular("visz","viszek","viszel","visz","viszünk","visztek","visznek","vitt","vin")
add_sz_irregular("hoz","hozok","hozol","hoz","hozunk","hoztok","hoznak","hozott","hoz")


# ============================ NOUN FST ============================

def pluralize(noun: str) -> str:
    if noun.endswith("a"):
        return noun[:-1] + "á" + "k"
    if noun.endswith("e"):
        return noun[:-1] + "é" + "k"
    if noun[-1].lower() in ALL_VOWELS:
        return noun + "k"
    h = harmony_set(noun)
    return noun + ("ok" if h == "back" else "ök" if h == "front_r" else "ek")

def noun_case_fst(noun: str, case: str, number: str) -> str:
    # Generate via simple deterministic FST made on the fly.
    class _N:
        pass
    fst = FST(); s1=fst.add_state(); fst.set_final(s1)
    base = noun if number=="Sing" else pluralize(noun)
    h = harmony_set(base)
    if case=="Nominative":
        fst.add_arc(0, s1, "ACC", base); return fst.transduce(["ACC"])
    if case=="Accusative":
        if base.endswith("a"):
            out = base[:-1] + "á" + "t"
        elif base.endswith("e"):
            out = base[:-1] + "é" + "t"
        elif base[-1].lower() in ALL_VOWELS:
            out = base + "t"
        else:
            last = base[-1].lower()
            bigram = base[-2:].lower() if len(base)>=2 else ""
            sibil = last in {"s","z","c"} or bigram in {"sz","zs","cs","dz","dzs"}
            dental = last in {"t","d"}
            plural_k = last == "k"
            use_link = sibil or dental or plural_k
            lv = "a" if last_vowel(base) in {"a","á"} else "e" if last_vowel(base) in {"e","é"} else ("o" if h=="back" else "ö" if h=="front_r" else "e")
            out = base + (lv + "t" if use_link else "t")
        fst.add_arc(0, s1, "ACC", out); return fst.transduce(["ACC"])
    if case=="Dative":
        fst.add_arc(0, s1, "DAT", base + ("nak" if h=="back" else "nek")); return fst.transduce(["DAT"])
    if case=="Inessive":
        fst.add_arc(0, s1, "INE", base + ("ban" if h=="back" else "ben")); return fst.transduce(["INE"])
    if case=="Superessive":
        if base[-1].lower() in ALL_VOWELS and number=="Sing":
            out = base + "n"
        else:
            out = base + ("on" if h=="back" else "ön" if h=="front_r" else "en")
        fst.add_arc(0, s1, "SUP", out); return fst.transduce(["SUP"])
    if case=="Adessive":
        fst.add_arc(0, s1, "ADE", base + ("nál" if h=="back" else "nél")); return fst.transduce(["ADE"])
    if case=="Illative":
        fst.add_arc(0, s1, "ILL", base + ("ba" if h=="back" else "be")); return fst.transduce(["ILL"])
    if case=="Sublative":
        fst.add_arc(0, s1, "SUB", base + ("ra" if h=="back" else "re")); return fst.transduce(["SUB"])
    if case=="Allative":
        add = ("hoz" if h=="back" else "höz" if h=="front_r" else "hez"); fst.add_arc(0, s1, "ALL", base + add); return fst.transduce(["ALL"])
    if case=="Instrumental":
        if base[-1].lower() in ALL_VOWELS:
            out = base + ("val" if h=="back" else "vel")
        else:
            ending = "al" if h=="back" else "el"
            if base.endswith(("sz","zs","cs","gy","ny","ty","ly")):
                out = base + base[-2:] + ending
            else:
                out = base + base[-1] + ending
        fst.add_arc(0, s1, "INS", out); return fst.transduce(["INS"])
    if case=="Genitive":
        if number=="Sing":
            if noun.endswith("a"): out = noun[:-1] + "á" + "é"
            elif noun.endswith("e"): out = noun[:-1] + "é" + "é"
            else: out = noun + "é"
        else:
            out = pluralize(noun) + "é"
        fst.add_arc(0, s1, "GEN", out); return fst.transduce(["GEN"])
    return base


# ============================ GENERATION API (FST-DRIVEN) ============================

PRONOUNS_HU = {("Sing",1):"én",("Sing",2):"te",("Sing",3):"ő",("Plur",1):"mi",("Plur",2):"ti",("Plur",3):"ők"}
CASE_TO_UD = {"Nominative":"Nom","Accusative":"Acc","Dative":"Dat","Inessive":"Ine","Superessive":"Sup","Adessive":"Ade","Illative":"Ill","Sublative":"Sub","Allative":"All","Instrumental":"Ins","Genitive":"Gen"}

def parse_verb_mode(mode: str) -> Tuple[str,str,bool]:
    if mode.startswith("Present"): return "Ind","Pres","Definite" in mode
    if mode.startswith("Past"):    return "Ind","Past","Definite" in mode
    if mode.startswith("Conditional"): return "Cnd","Pres","Definite" in mode
    return "Ind","Fut","Definite" in mode

def make_ud_key_for_verb(mood: str, tense: str, definite: bool, person: int, number: str) -> str:
    dval = "Def" if definite else "Ind"
    return f"VERB VerbForm=Fin|Mood={mood}|Tense={tense}|Person={person}|Number={'Sing' if number=='Sing' else 'Plur'}|Definite={dval}"

def make_ud_key_for_noun(case: str, number: str) -> str:
    return f"NOUN Case={CASE_TO_UD[case]}|Number={'Sing' if number=='Sing' else 'Plur'}"

def choose_person_number() -> Tuple[int,str]:
    return random.choice([1,2,3]), random.choice(["Sing","Plur"])

def fst_generate_verb(lemma: str, mood: str, tense: str, definite: bool, person: int, number: str, is_ik: bool) -> str:
    irr = IRREGULARS.get(lemma, {}).get((tense, mood, definite, person, number))
    if irr: return irr

    # Build an FST for the selected slot and transduce one-symbol input encoding person/number.
    if tense=="Pres" and mood=="Ind":
        fst = build_present_def_fst(lemma) if definite else build_present_indef_fst(lemma, is_ik)
    elif tense=="Past" and mood=="Ind":
        fst = build_past_def_fst(lemma) if definite else build_past_indef_fst(lemma)
    elif tense=="Pres" and mood=="Cnd":
        fst = build_cond_def_fst(lemma) if definite else build_cond_indef_fst(lemma)
    elif tense=="Fut" and mood=="Ind":
        return future_form(definite, person, number, lemma)
    else:
        return lemma

    inp = f"P{person}{'SG' if number=='Sing' else 'PL'}"
    return fst.transduce([inp])

def fst_generate_noun(noun: str, case: str, number: str) -> str:
    return noun_case_fst(noun, case, number)


# ============================ CORPUS & STATE ============================

REQUIRED_COLS = {"pos","lemma","english"}
def validate_corpus(df: pd.DataFrame) -> Tuple[bool,str]:
    missing = REQUIRED_COLS - set(df.columns)
    if missing: return False, f"Missing required columns: {', '.join(sorted(missing))}"
    return True, "ok"

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

if "df" not in st.session_state: st.session_state.df = None
if df is not None:
    ok, msg = validate_corpus(df)
    if ok: st.session_state.df = df.copy()
    else: st.error(msg)

for key, val in [("score",0),("total",0),("current",None),("solution",""),("kind",""),("feedback",""),("tts_last_audio",None),("checked",False)]:
    st.session_state.setdefault(key, val)


# ============================ TASK LOGIC ============================

def next_task(df: pd.DataFrame) -> Tuple[str, dict, str]:
    scope = []
    if st.session_state.want_verbs and verb_modes: scope.append("verb")
    if st.session_state.want_nouns and noun_modes and noun_numbers: scope.append("noun")
    if not scope:
        st.warning("Enable at least one practice scope in the sidebar."); return "none", {}, ""

    which = random.choice(scope)
    if which == "verb":
        sub = df[df["pos"].str.upper().eq("VERB")]
        if sub.empty:
            st.warning("CSV contains no verbs."); return "none", {}, ""
        row = sub.sample(1).iloc[0]
        mode_choice = random.choice(verb_modes)
        mood, tense, definite = parse_verb_mode(mode_choice)
        person, number = choose_person_number()
        ud_key = make_ud_key_for_verb(mood, tense, definite, person, number)
        is_ik = is_ik_verb(str(row["lemma"]), get_is_ik_flag(row))
        task = VerbTask(str(row["lemma"]), str(row["english"]), mood, tense, definite, person, number, is_ik, ud_key)
        sol = fst_generate_verb(task.lemma, task.mood, task.tense, task.definite, task.person, task.number, task.is_ik)
        return "verb", task.__dict__, sol

    sub = df[df["pos"].str.upper().eq("NOUN")]
    if sub.empty:
        st.warning("CSV contains no nouns."); return "none", {}, ""
    row = sub.sample(1).iloc[0]
    case = random.choice(noun_modes)
    number = "Sing" if random.choice(noun_numbers) == "Singular" else "Plur"
    task = NounTask(str(row["lemma"]), str(row["english"]), case, number, make_ud_key_for_noun(case, number))
    sol = fst_generate_noun(task.lemma, task.case, task.number)
    return "noun", task.__dict__, sol


def new_question():
    st.session_state.feedback = ""
    st.session_state.tts_last_audio = None
    st.session_state.checked = False
    if st.session_state.df is None:
        st.info("Use the sidebar to upload your corpus or load it from GitHub, then click Next.")
        return
    kind, payload, solution = next_task(st.session_state.df)
    st.session_state.kind = kind
    st.session_state.current = payload
    st.session_state.solution = solution


# ============================ TTS ============================

def tts_speak_hu(text: str, rate: float) -> Optional[bytes]:
    provider = st.session_state.tts_provider
    if not text or provider == "Off": return None
    try:
        if provider.startswith("gTTS"):
            if not _HAS_GTTS: st.error("gTTS is not installed."); return None
            t = gTTS(text, lang="hu"); buf = BytesIO(); t.write_to_fp(buf); buf.seek(0); return buf.read()
        if provider.startswith("Google"):
            if not _HAS_GOOGLE_TTS: st.error("google-cloud-texttospeech is not installed."); return None
            sa = st.secrets.get("GOOGLE_TTS_SERVICE_ACCOUNT_JSON", None)
            creds = service_account.Credentials.from_service_account_info(sa) if isinstance(sa, dict) else None
            client = texttospeech.TextToSpeechClient(credentials=creds) if creds else texttospeech.TextToSpeechClient()
            synthesis_input = texttospeech.SynthesisInput(text=text)
            voice = texttospeech.VoiceSelectionParams(language_code="hu-HU")
            audio_config = texttospeech.AudioConfig(audio_encoding=texttospeech.AudioEncoding.MP3, speaking_rate=float(st.session_state.tts_rate))
            response = client.synthesize_speech(input=synthesis_input, voice=voice, audio_config=audio_config)
            return response.audio_content
    except Exception as e:
        st.error(f"TTS failed: {e}")
    return None


# ============================ MAIN UI ============================

colL, colR = st.columns([2,1])
with colL:
    if st.session_state.current is None and st.session_state.df is not None:
        new_question()
    elif st.session_state.df is None:
        st.info("Use the sidebar to upload your corpus or load it from GitHub, then click Next.")

    if st.button("Next", use_container_width=True):
        new_question()

    if st.session_state.current and st.session_state.kind in {"verb","noun"}:
        c = st.session_state.current
        if st.session_state.kind == "verb":
            pron = PRONOUNS_HU[(c["number"], c["person"])] if st.session_state.show_hu_pronouns else ""
            if c["person"] == 2:
                expected = "te" if c["number"]=="Sing" else "ti"
                if st.session_state.show_hu_pronouns and pron != expected:
                    pron = expected
            mode_map = {("Ind","Pres"):"present", ("Ind","Past"):"past", ("Cnd","Pres"):"conditional present", ("Ind","Fut"):"future"}
            conj = f"{'definite' if c['definite'] else 'indefinite'} {mode_map[(c['mood'], c['tense'])]}"
            st.markdown(
                f"""
                <div class="prompt-card">
                  <div><span class="pill">Verb</span><span class="pill">{conj}</span></div>
                  <div class="mono" style="font-size:1.25rem;margin-top:.25rem;"><b>{c["lemma"]}</b></div>
                  <div class="muted">Meaning: {c["gloss"]}</div>
                  <div class="muted">Pronoun: {pron if pron else f'person {c["person"]}, {c["number"]}'}</div>
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

        c_answer = st.text_input("Type the correct form")

        colA, colB = st.columns([1,1])
        with colA:
            if st.button("Check", disabled=st.session_state.checked or not c_answer.strip()):
                user = normalize_answer(c_answer, st.session_state.ignore_accents)
                gold = normalize_answer(st.session_state.solution, st.session_state.ignore_accents)
                st.session_state.total += 1
                if user == gold and len(gold) > 0:
                    st.session_state.score += 1
                    st.session_state.feedback = f"<span class='good'>Correct.</span> {st.session_state.solution}"
                    if st.session_state.auto_say_answer:
                        audio = tts_speak_hu(st.session_state.solution, st.session_state.tts_rate)
                        if audio: st.session_state.tts_last_audio = audio
                else:
                    st.session_state.feedback = f"<span class='bad'>Not quite.</span> Expected: <b>{st.session_state.solution}</b>"
                st.session_state.checked = True
        with colB:
            if st.session_state.allow_reveal and st.button("Reveal"):
                st.session_state.feedback = f"Answer: <b>{st.session_state.solution}</b>"
                if st.session_state.auto_say_answer:
                    audio = tts_speak_hu(st.session_state.solution, st.session_state.tts_rate)
                    if audio: st.session_state.tts_last_audio = audio

        if st.session_state.feedback:
            st.markdown(st.session_state.feedback, unsafe_allow_html=True)

        colT1, colT2 = st.columns([1,1])
        with colT1:
            if st.button("🔊 Speak prompt"):
                audio = tts_speak_hu(c["lemma"], st.session_state.tts_rate)
                if audio: st.session_state.tts_last_audio = audio
        with colT2:
            if st.button("🔊 Speak correct form"):
                audio = tts_speak_hu(st.session_state.solution, st.session_state.tts_rate)
                if audio: st.session_state.tts_last_audio = audio

        if st.session_state.tts_last_audio:
            st.audio(st.session_state.tts_last_audio, format="audio/mp3")

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

st.caption("FST generation covers present, past, conditional present, and analytic future with fog + infinitive; nouns cover core cases. Built-in irregulars: van, megy, jön, eszik, iszik, tesz, vesz, hisz, visz, hoz. -ik verbs handled in present.")


# ============================ PUBLIC API FOR INTEGRATION ============================
# If you later install pynini or hfst, you can replace the tiny FST with a compiled transducer:
#   - Define a lexicon of stems with archiphonemes (V_back/V_front, etc.)
#   - Compile suffix transducers keyed by feature bundles
#   - Compose: lexical_form @ morphotactics @ phonology -> surface
# The app only needs fst_generate_verb / fst_generate_noun to produce strings.
