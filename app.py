# app.py
import os
import csv
import io
import random
import unicodedata
import streamlit as st

# ----------------------------
# Hugging Face cache locations
# ----------------------------
# These help Streamlit Cloud keep the model between runs.
os.environ.setdefault("TRANSFORMERS_CACHE", "/mount/data/transformers")
os.environ.setdefault("HF_HOME", "/mount/data/hf_home")
os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")

# ----------------------------
# Page and theme
# ----------------------------
st.set_page_config(page_title="Hungarian Morphology Trainer", page_icon="🇭🇺", layout="centered")
st.markdown("""
<style>
:root{
  --navy:#0a2540;
  --cream:#fff6e6;
  --ink:#0d1b2a;
  --muted:#5b6b83;
  --ok:#16a34a;
  --bad:#dc2626;
  --chip:#e8eef7;
}
html, body, [class*="css"]{font-family: system-ui, -apple-system, Segoe UI, Roboto, Helvetica, Arial;}
body{background: var(--cream);}
.block-container{padding-top:1.0rem; padding-bottom:1.25rem;}

h1,h2,h3{color:var(--navy);}

/* primary button with motion */
.stButton>button[kind="primary"]{
  background: linear-gradient(135deg, #0a2540, #1f6f8b);
  border:0; color:#fff; padding:.62rem 1.1rem; border-radius:12px;
  box-shadow:0 6px 18px rgba(10,37,64,.25);
  transition:transform .08s ease-out, box-shadow .15s ease;
}
.stButton>button[kind="primary"]:hover{ transform: translateY(-1px); }
.stButton>button:not([kind="primary"]){
  background:#fff; border:1px solid #e6eaf0; color:#0a2540;
  padding:.55rem 1rem; border-radius:12px;
}

/* prompt chip */
.prompt-chip{
  display:inline-block; background:#eaf0ff; color:#0a2540;
  padding:.5rem .8rem; border-radius:12px; font-weight:700;
}
.case-chip{ background:#fff0da; }
.verb-chip{ background:#e6f5ff; }

/* feedback pills */
.feedback{ display:inline-block; padding:.52rem .9rem; border-radius:999px; font-weight:700;}
.feedback.ok{ background:var(--ok); color:#fff;}
.feedback.bad{ background:var(--bad); color:#fff;}

/* card shell */
.card{
  background:#fff;
  border-radius:18px;
  padding:1.0rem 1.1rem .9rem 1.1rem;
  box-shadow:0 10px 28px rgba(10,37,64,.08);
}
.header-wrap{ text-align:center; margin-bottom:.6rem;}
.header-title{ color:var(--muted); font-size:.95rem; }
.header-lemma{ color:var(--navy); font-size:1.9rem; font-weight:800; }
.underline{height:2px; width:78%; background:#0a2540; opacity:.2; margin:.6rem auto .5rem auto; border-radius:2px;}

/* input shaping */
input[type="text"]{ border-radius:12px !important; }
</style>
""", unsafe_allow_html=True)

# ----------------------------
# Utilities
# ----------------------------
PRONOUNS = {"1Sg":"én","2Sg":"te","3Sg":"ő","1Pl":"mi","2Pl":"ti","3Pl":"ők"}

def normalize(s: str) -> str:
    return unicodedata.normalize("NFC", s.strip().lower())

def strip_diacritics(s: str) -> str:
    return ''.join(ch for ch in unicodedata.normalize("NFD", s) if unicodedata.category(ch) != "Mn")

def answers_equal(a: str, b: str, strict_accents: bool) -> bool:
    a = normalize(a); b = normalize(b)
    if strict_accents:
        return a == b
    return strip_diacritics(a) == strip_diacritics(b)

# ----------------------------
# Model loader and generator
# ----------------------------
@st.cache_resource(show_spinner=False)
def load_generator():
    # model requires sentencepiece
    try:
        import sentencepiece  # noqa: F401
    except Exception:
        st.error("Missing dependency: sentencepiece. Add `sentencepiece>=0.1.99` to requirements.txt and redeploy.")
        st.stop()

    from transformers import pipeline
    # device_map="auto" chooses CPU on Streamlit Cloud
    return pipeline(
        task="text2text-generation",
        model="NYTK/morphological-generator-emmorph-mt5-hungarian",
        device_map="auto"
    )

def build_prompt(lemma: str, pos_code: str, tag: str) -> str:
    # Store pos_code as "/N" or "/V" without brackets, add exactly one bracketed POS here.
    pos = pos_code.strip()
    if pos.startswith("[") and pos.endswith("]"):
        pos = pos[1:-1]
    return f"morph: {lemma} [{pos}]{tag}"

@st.cache_data(show_spinner=False)
def generate_one(lemma: str, pos_code: str, tag: str) -> str:
    gen = load_generator()
    out = gen(build_prompt(lemma, pos_code, tag), max_new_tokens=8)[0]["generated_text"]
    return out.strip()

# ----------------------------
# Sidebar controls
# ----------------------------
with st.sidebar:
    st.title("Practice settings")

    st.subheader("What to practice")
    c1, c2 = st.columns(2)
    with c1:
        opt_ine = st.checkbox("Inessive", value=True, help="Noun case: -ban / -ben")
        opt_prs_indef = st.checkbox("Present indefinite", value=True)
    with c2:
        opt_ade = st.checkbox("Adessive", value=True, help="Noun case: -nál / -nél")
        opt_prs_def = st.checkbox("Present definite", value=False)

    strict_accents = st.toggle("Require accents for answers", value=False, help="Turn on to require correct diacritics.")
    warm = st.button("Initialize model now", help="Loads the generator so the first check is instant.")
    if warm:
        _ = load_generator()
        st.success("Model ready.")

    st.divider()
    st.subheader("Lemmas")
    source = st.radio("Choose your list", ["Sample list", "Upload CSVs"])
    nouns_file = verbs_file = None
    if source == "Upload CSVs":
        with st.popover("How to format CSVs"):
            st.write("Upload two CSV files with a header named `lemma` and one lemma per row, UTF‑8 encoded.")
            st.code("lemma\nház\nkönyv\nember\n", language="text")
            st.code("lemma\nlát\nír\nszeret\n", language="text")
        nouns_file = st.file_uploader("Upload nouns.csv", type=["csv"])
        verbs_file = st.file_uploader("Upload verbs.csv", type=["csv"])

# ----------------------------
# Lemma loading
# ----------------------------
SAMPLE_NOUNS = ["ház","ablak","asztal","könyv","ember","város","kert","kút","madár","téma"]
SAMPLE_VERBS = ["lát","ír","olvas","mond","mos","tör","tanul","szeret","ad","vesz"]

def read_lemmas(upload) -> list[str]:
    text = upload.getvalue().decode("utf-8")
    # Try DictReader first for header "lemma"
    try:
        rows = list(csv.DictReader(io.StringIO(text)))
        if rows and "lemma" in rows[0]:
            return [normalize(r["lemma"]) for r in rows if r.get("lemma")]
    except Exception:
        pass
    # Fallback: first column
    return [normalize(r[0]) for r in csv.reader(io.StringIO(text)) if r]

def load_lemma_pools():
    nouns = SAMPLE_NOUNS[:]
    verbs = SAMPLE_VERBS[:]
    if nouns_file is not None:
        try:
            nouns = read_lemmas(nouns_file)
        except Exception:
            st.error("Could not read nouns.csv. Ensure it has a 'lemma' header and UTF‑8 encoding.")
    if verbs_file is not None:
        try:
            verbs = read_lemmas(verbs_file)
        except Exception:
            st.error("Could not read verbs.csv. Ensure it has a 'lemma' header and UTF‑8 encoding.")
    return sorted(set(nouns)), sorted(set(verbs))

nouns, verbs = load_lemma_pools()

# ----------------------------
# Tag pools using emMorph codes
# ----------------------------
NOUN_TAGS: list[tuple[str, str, str]] = []
if opt_ine: NOUN_TAGS.append(("/N", "[Ine]", "Inessive"))
if opt_ade: NOUN_TAGS.append(("/N", "[Ade]", "Adessive"))

VERB_TAGS: list[tuple[str, str, str]] = []
if opt_prs_indef:
    for p in ["1Sg","2Sg","3Sg","1Pl","2Pl","3Pl"]:
        VERB_TAGS.append(("/V", f"[Prs.NDef.{p}]", f"Present indefinite · {PRONOUNS[p]}"))
if opt_prs_def:
    for p in ["1Sg","2Sg","3Sg","1Pl","2Pl","3Pl"]:
        VERB_TAGS.append(("/V", f"[Prs.Def.{p}]", f"Present definite · {PRONOUNS[p]}"))

if not NOUN_TAGS and not VERB_TAGS:
    st.info("Choose at least one option in the sidebar to begin.")
    st.stop()

# ----------------------------
# Exercise pool and state
# ----------------------------
def build_pool():
    items = []
    for lemma in nouns:
        for pos, tag, label in NOUN_TAGS:
            items.append(("N", lemma, pos, tag, label))
    for lemma in verbs:
        for pos, tag, label in VERB_TAGS:
            items.append(("V", lemma, pos, tag, label))
    random.shuffle(items)
    return items

if "pool" not in st.session_state: st.session_state.pool = build_pool()
if "idx" not in st.session_state: st.session_state.idx = 0
if "score" not in st.session_state: st.session_state.score = 0
if "seen" not in st.session_state: st.session_state.seen = 0
if "current" not in st.session_state: st.session_state.current = None

def next_question():
    if st.session_state.idx >= len(st.session_state.pool):
        random.shuffle(st.session_state.pool)
        st.session_state.idx = 0
        st.balloons()
    kind, lemma, pos, tag, label = st.session_state.pool[st.session_state.idx]
    if kind == "V":
        p = tag.split(".")[-1].strip("]")
        pron = PRONOUNS.get(p, "")
        pretty = f"{pron} + {lemma}"
        chip_class = "verb-chip"
        sublabel = label
    else:
        pretty = f"{lemma} + {label}"
        chip_class = "case-chip"
        sublabel = "Case practice"
    st.session_state.current = dict(kind=kind, lemma=lemma, pos=pos, tag=tag, pretty=pretty, chip_class=chip_class, sublabel=sublabel)

# controls row
c1, c2, c3 = st.columns(3)
with c1:
    if st.button("Build or reset pool"):
        st.session_state.pool = build_pool()
        st.session_state.idx = 0
        st.session_state.score = 0
        st.session_state.seen = 0
        st.session_state.current = None
with c2:
    if st.button("Reshuffle remaining"):
        tail = st.session_state.pool[st.session_state.idx:]
        random.shuffle(tail)
        st.session_state.pool[st.session_state.idx:] = tail
with c3:
    st.caption(f"Items left this cycle: {max(0, len(st.session_state.pool) - st.session_state.idx)}")

if st.session_state.current is None:
    next_question()

# ----------------------------
# Card UI
# ----------------------------
st.markdown('<div class="card">', unsafe_allow_html=True)
st.markdown(f"""
<div class="header-wrap">
  <div class="header-title">{'verb' if st.session_state.current['kind']=='V' else 'noun'}</div>
  <div class="header-lemma">{st.session_state.current['lemma']}</div>
</div>
""", unsafe_allow_html=True)
st.markdown('<div class="underline"></div>', unsafe_allow_html=True)

st.write("Type the correct form for:")
st.markdown(f'<span class="prompt-chip {st.session_state.current["chip_class"]}">{st.session_state.current["pretty"]}</span>', unsafe_allow_html=True)
st.caption(st.session_state.current["sublabel"])
st.markdown('</div>', unsafe_allow_html=True)  # end card

# answer row
user = st.text_input("Your answer", value="", placeholder="type here, then Check")
col1, col2, col3 = st.columns([1,1,1])
with col1:
    check = st.button("Check", type="primary")
with col2:
    reveal = st.button("Show answer")
with col3:
    nxt = st.button("Next")

fb = st.empty()

def show_feedback(ok: bool, gold: str):
    if ok:
        st.toast("Correct!", icon="✅")
        fb.markdown(f'<span class="feedback ok">Correct: {gold}</span>', unsafe_allow_html=True)
    else:
        st.toast("Not quite", icon="❌")
        fb.markdown(f'<span class="feedback bad">Expected: {gold}</span>', unsafe_allow_html=True)

# ----------------------------
# Events
# ----------------------------
if check:
    try:
        gold = generate_one(st.session_state.current["lemma"], st.session_state.current["pos"], st.session_state.current["tag"])
    except Exception as e:
        st.exception(e)
        st.stop()
    st.session_state.seen += 1
    if answers_equal(user, gold, strict_accents):
        st.session_state.score += 1
        show_feedback(True, gold)
        st.session_state.idx += 1
        next_question()
    else:
        show_feedback(False, gold)

if reveal:
    try:
        gold = generate_one(st.session_state.current["lemma"], st.session_state.current["pos"], st.session_state.current["tag"])
    except Exception as e:
        st.exception(e)
        st.stop()
    show_feedback(False, gold)
    st.session_state.idx += 1
    next_question()

if nxt:
    st.session_state.idx += 1
    next_question()
    fb.empty()

# footer
st.write(f"Score: {st.session_state.score} / {st.session_state.seen}")
st.progress(0 if st.session_state.seen == 0 else min(1.0, st.session_state.score / max(1, st.session_state.seen)))
st.caption("Answers are generated with the emMorph tagset using NYTK’s Hungarian morphological generator on Hugging Face.")
