# app.py
import csv
import io
import random
import unicodedata
import streamlit as st

# --------------------
# Page & global style
# --------------------
st.set_page_config(page_title="Hungarian Morphology Trainer", page_icon="🇭🇺", layout="centered")

st.markdown('''
<style>
/* Base */
:root {
  --bg:#fffaf5;
  --card:#ffeedd;
  --ink:#2d2a26;
  --muted:#6a6a73;
  --accent:#7bd88f;       /* green chip */
  --chip:#e9e9ef;         /* gray chip */
  --brand1:#5b8def;
  --brand2:#6bd0a6;
}
html, body, [class*="css"]  {font-family: ui-sans-serif, system-ui, -apple-system, Segoe UI, Roboto, Helvetica, Arial;}
.block-container{padding-top:1rem; padding-bottom:2rem;}
/* App background */
body{background: var(--bg);}

/* Top lemma section */
.header-wrap{
  text-align:center;
  margin: 0 0 0.75rem 0;
}
.header-title{
  color: var(--muted);
  font-size: 0.95rem;
  margin-bottom: .25rem;
}
.header-lemma{
  font-weight: 800;
  font-size: 1.9rem;
  color: var(--ink);
}

/* Card */
.practice-zone{
  background: var(--card);
  padding: 1.1rem;
  border-radius: 18px;
  box-shadow: 0 10px 28px rgba(0,0,0,.06) inset;
}
.card{
  background: #fff;
  border-radius: 18px;
  padding: 1.1rem 1.1rem 0.7rem 1.1rem;
  box-shadow: 0 8px 22px rgba(0,0,0,.08);
}
.card-pron{
  text-align:center;
  font-size: 1.6rem;
  color: var(--ink);
  font-weight: 700;
  margin-top:.3rem;
}
.card-noun{
  text-align:center;
  font-size: 1.6rem;
  color: var(--ink);
  font-weight: 700;
  margin-top:.3rem;
}
.underline{
  height: 2px;
  width: 80%;
  background: #3b302a;
  opacity:.35;
  margin: .7rem auto 0.6rem auto;
  border-radius:2px;
}

/* Chips */
.chips{ display:flex; gap:.5rem; justify-content:center; margin:.1rem 0 .3rem 0;}
.chip{
  padding: .24rem .6rem;
  border-radius: 999px;
  font-weight:700;
  font-size:.78rem;
}
.chip-green{ background: var(--accent); color:#073b18;}
.chip-gray{ background: var(--chip); color:#333;}

/* Buttons */
.stButton>button[kind="primary"]{
  background: linear-gradient(135deg, var(--brand1), var(--brand2));
  border: 0;
  color: white;
  padding: 0.62rem 1.15rem;
  border-radius: 14px;
  transition: transform .08s ease-out, box-shadow .15s ease;
  box-shadow: 0 6px 18px rgba(0,0,0,.18);
}
.stButton>button[kind="primary"]:hover{ transform: translateY(-1px); }
.stButton>button[kind="secondary"]{
  background:#fff; border:1px solid #e7eaf3; border-radius:14px; padding:.55rem 1rem;
}

/* Feedback pills */
.feedback{
  display:inline-block; padding:.55rem .9rem; border-radius:999px; font-weight:700;
}
.feedback.ok{ background:#14b87a; color:white;}
.feedback.bad{ background:#f04444; color:white;}

/* Bottom nav (decorative) */
.navbar{
  display:flex; justify-content:space-around; align-items:center;
  padding:.35rem .75rem; margin-top:.8rem;
  color:#7b7b86; font-size:.85rem;
}
.navitem{ opacity:.8;}
.navitem .active{ color:var(--ink); font-weight:700;}
/* Text input soften */
input[type="text"]{ border-radius: 12px !important; }
</style>
''', unsafe_allow_html=True)

# --------------------
# Helpers
# --------------------
def normalize(s: str) -> str:
    return unicodedata.normalize("NFC", s.strip().lower())

@st.cache_resource(show_spinner=True)
def load_generator():
    from transformers import pipeline
    return pipeline(
        task="text2text-generation",
        model="NYTK/morphological-generator-emmorph-mt5-hungarian"
    )

def build_prompt(lemma: str, pos_code: str, tag: str) -> str:
    return f"morph: {lemma} [{pos_code}]{tag}"

def generate_form(lemma: str, pos_code: str, tag: str) -> str:
    gen = load_generator()
    out = gen(build_prompt(lemma, pos_code, tag), max_new_tokens=8)[0]["generated_text"]
    return out.strip()

PRONOUNS = {"1Sg":"én","2Sg":"te","3Sg":"ő","1Pl":"mi","2Pl":"ti","3Pl":"ők"}

# --------------------
# Sidebar – simple options
# --------------------
with st.sidebar:
    st.header("Settings")
    st.write("Choose what to practice. The app generates answers with the emMorph tagset.")

    c1, c2 = st.columns(2)
    with c1:
        opt_ine = st.checkbox("Inessive", value=True, help="Noun case: -ban/-ben")
        opt_prs_indef = st.checkbox("Present indefinite", value=True)
    with c2:
        opt_ade = st.checkbox("Adessive", value=True, help="Noun case: -nál/-nél")
        opt_prs_def = st.checkbox("Present definite", value=False)

    st.divider()
    st.subheader("Lemmas")
    src = st.radio("Pick your source", ["Sample list", "Upload CSVs"])
    nouns_up = verbs_up = None
    if src == "Upload CSVs":
        with st.popover("CSV format (click)"):
            st.write("Each file must contain a header named **lemma** and one lemma per row.")
            st.write("Nouns file example:")
            st.code("lemma\nház\nkönyv\nember\n", language="text")
            st.write("Verbs file example:")
            st.code("lemma\nlát\nír\nszeret\n", language="text")
            st.caption("Encoding: UTF-8. Only the lemma column is required.")
        nouns_up = st.file_uploader("Upload nouns.csv", type=["csv"])
        verbs_up = st.file_uploader("Upload verbs.csv", type=["csv"])

# --------------------
# Lemma loading
# --------------------
SAMPLE_NOUNS = ["ház","ablak","asztal","könyv","ember","város","kert","kút","madár","téma"]
SAMPLE_VERBS = ["lát","ír","olvas","mond","mos","tör","tanul","szeret","ad","vesz"]

def read_lemmas(upload):
    text = upload.getvalue().decode("utf-8")
    try:
        rows = list(csv.DictReader(io.StringIO(text)))
        if rows and "lemma" in rows[0]:
            return [normalize(r["lemma"]) for r in rows if r.get("lemma")]
    except Exception:
        pass
    return [normalize(r[0]) for r in csv.reader(io.StringIO(text)) if r]

def load_lemmas():
    nouns, verbs = SAMPLE_NOUNS[:], SAMPLE_VERBS[:]
    if nouns_up is not None: nouns = read_lemmas(nouns_up)
    if verbs_up is not None: verbs = read_lemmas(verbs_up)
    return sorted(set(nouns)), sorted(set(verbs))

nouns, verbs = load_lemmas()

# --------------------
# Build target pools (tags hidden from user)
# --------------------
NOUN_TAGS = []
if opt_ine: NOUN_TAGS.append(("[/N]","[Ine]","Inessive"))
if opt_ade: NOUN_TAGS.append(("[/N]","[Ade]","Adessive"))

VERB_TAGS = []
if opt_prs_indef:
    for p in ["1Sg","2Sg","3Sg","1Pl","2Pl","3Pl"]:
        VERB_TAGS.append(("[/V]", f"[Prs.NDef.{p}]", p, "Present indefinite"))
if opt_prs_def:
    for p in ["1Sg","2Sg","3Sg","1Pl","2Pl","3Pl"]:
        VERB_TAGS.append(("[/V]", f"[Prs.Def.{p}]", p, "Present definite"))

if not NOUN_TAGS and not VERB_TAGS:
    st.info("Pick at least one item to practice in the sidebar.")
    st.stop()

def build_pool():
    pool = []
    for n in nouns:
        for pos, tag, label in NOUN_TAGS:
            pool.append(("N", n, pos, tag, label, None))
    for v in verbs:
        for pos, tag, person, paradigm in VERB_TAGS:
            pool.append(("V", v, pos, tag, paradigm, person))
    random.shuffle(pool)
    return pool

if "pool" not in st.session_state: st.session_state.pool = build_pool()
if "i" not in st.session_state: st.session_state.i = 0
if "score" not in st.session_state: st.session_state.score = 0
if "seen" not in st.session_state: st.session_state.seen = 0

def current_item():
    if st.session_state.i >= len(st.session_state.pool):
        random.shuffle(st.session_state.pool)
        st.session_state.i = 0
        st.balloons()
    return st.session_state.pool[st.session_state.i]

kind, lemma, pos, tag, label, person = current_item()

# --------------------
# UI layout (mobile-style card)
# --------------------
st.markdown(f'''
<div class="header-wrap">
  <div class="header-title">{'verb' if kind=='V' else 'noun'}</div>
  <div class="header-lemma">{lemma}</div>
</div>
''', unsafe_allow_html=True)

st.markdown('<div class="practice-zone">', unsafe_allow_html=True)
st.markdown('<div class="card">', unsafe_allow_html=True)

if kind == "V":
    st.markdown(f'<div class="card-pron">{PRONOUNS[person]}</div>', unsafe_allow_html=True)
else:
    st.markdown(f'<div class="card-noun">{lemma}</div>', unsafe_allow_html=True)

st.markdown('<div class="underline"></div>', unsafe_allow_html=True)

if kind == "V":
    chip1 = "Present"
    chip2 = "Definite" if "Def" in tag else "Indefinite"
else:
    chip1 = label
    chip2 = "Case"

st.markdown(f'''
<div class="chips">
  <span class="chip chip-green">{chip1}</span>
  <span class="chip chip-gray">{chip2}</span>
</div>
''', unsafe_allow_html=True)

st.markdown('</div>', unsafe_allow_html=True)   # end .card
st.markdown('</div>', unsafe_allow_html=True)   # end .practice-zone

answer = st.text_input("Type your answer", value="", placeholder="type the form, then Check")

c1, c2, c3 = st.columns(3)
with c1:
    do_check = st.button("Check", type="primary")
with c2:
    do_show = st.button("Tap to show")
with c3:
    do_next = st.button("Next")

fb = st.empty()

def feedback(ok, gold):
    if ok:
        fb.markdown(f'<span class="feedback ok">Correct: {gold}</span>', unsafe_allow_html=True)
        st.session_state.score += 1
    else:
        fb.markdown(f'<span class="feedback bad">Expected: {gold}</span>', unsafe_allow_html=True)

def gold_form():
    return generate_form(lemma, pos, tag)

if do_check:
    st.session_state.seen += 1
    gold = gold_form()
    feedback(normalize(answer)==normalize(gold), gold)
    st.session_state.i += 1

if do_show:
    gold = gold_form()
    feedback(False, gold)
    st.session_state.i += 1

if do_next:
    st.session_state.i += 1
    fb.empty()

st.write(f"Score: {st.session_state.score} / {st.session_state.seen}")

st.markdown('''
<div class="navbar">
  <div class="navitem"><span class="active">Practice</span></div>
  <div class="navitem">Rhymes</div>
  <div class="navitem">Verbs</div>
  <div class="navitem">Settings</div>
</div>
''', unsafe_allow_html=True)
