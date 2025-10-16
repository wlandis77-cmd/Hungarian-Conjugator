# app.py
import csv
import io
import random
import unicodedata
import streamlit as st

# --------------
# Page + styling
# --------------
st.set_page_config(page_title="Hungarian Morphology Trainer", page_icon="🇭🇺", layout="centered")
st.markdown("""
<style>
html, body, [class*="css"]  {
  font-family: system-ui, -apple-system, Segoe UI, Roboto, Helvetica, Arial, "Apple Color Emoji", "Segoe UI Emoji";
}
/* Primary buttons with motion */
.stButton>button[kind="primary"]{
  background: linear-gradient(135deg, #5b8def, #6bd0a6);
  border: 0;
  color: white;
  padding: 0.6rem 1.1rem;
  border-radius: 12px;
  transition: transform .08s ease-out, box-shadow .15s ease;
  box-shadow: 0 4px 12px rgba(0,0,0,.15);
}
.stButton>button[kind="primary"]:hover{
  transform: translateY(-1px);
  box-shadow: 0 6px 18px rgba(0,0,0,.20);
}
.stButton>button[kind="primary"]:active{
  transform: translateY(0);
  box-shadow: 0 3px 8px rgba(0,0,0,.18);
}
/* Secondary buttons */
.stButton>button:not([kind="primary"]){
  background: #ffffff;
  border: 1px solid #e7eaf3;
  border-radius: 12px;
  padding: 0.5rem 1rem;
  transition: transform .08s ease-out, box-shadow .15s ease, background .2s ease;
}
.stButton>button:not([kind="primary"]):hover{
  background: #f7f9fc;
  transform: translateY(-1px);
}
/* Feedback pills */
.feedback{
  display:inline-block;
  padding:.55rem .9rem;
  border-radius: 999px;
  font-weight:600;
  border:0;
}
.feedback.ok{ background:#14b87a; color:white; }
.feedback.bad{ background:#f04444; color:white; }
/* Prompt chip */
.prompt-chip{
  display:inline-block;
  background:#eef2ff;
  color:#344;
  padding:.5rem .8rem;
  border-radius:12px;
  font-weight:600;
}
.case-chip{ background:#fff7ed; }
.verb-chip{ background:#ecfeff; }
/* Input */
input[type="text"]{ border-radius: 12px !important; }
</style>
""", unsafe_allow_html=True)

# --------------
# Helpers
# --------------
def normalize(s: str) -> str:
    return unicodedata.normalize("NFC", s.strip().lower())

@st.cache_resource(show_spinner=True)
def load_generator():
    # Guard against missing dependency that the model needs.
    try:
        import sentencepiece  # noqa: F401
    except Exception:
        st.error("Missing dependency: sentencepiece. Add `sentencepiece>=0.1.99` to requirements.txt and redeploy.")
        st.stop()

    from transformers import pipeline
    # mT5-based Hungarian morphological generator trained on emMorph tags
    return pipeline(
        task="text2text-generation",
        model="NYTK/morphological-generator-emmorph-mt5-hungarian"
    )

def build_prompt(lemma: str, pos_code: str, tag: str) -> str:
    # Accept "/N" or "/V" (without brackets) and add exactly one set of brackets.
    pos = pos_code.strip()
    if pos.startswith("[") and pos.endswith("]"):
        pos = pos[1:-1]
    return f"morph: {lemma} [{pos}]{tag}"

def generate_form(lemma: str, pos_code: str, tag: str) -> str:
    try:
        gen = load_generator()
        out = gen(build_prompt(lemma, pos_code, tag), max_new_tokens=8)[0]["generated_text"]
        return out.strip()
    except Exception as e:
        st.error(f"Generation failed: {e}")
        return ""

# Pronoun labels for UI
PRONOUNS = {
    "1Sg": "én",
    "2Sg": "te",
    "3Sg": "ő",
    "1Pl": "mi",
    "2Pl": "ti",
    "3Pl": "ők",
}

# --------------
# Sidebar (simple options)
# --------------
with st.sidebar:
    st.title("Practice settings")

    st.subheader("What to practice")
    colA, colB = st.columns(2)
    with colA:
        opt_ine = st.checkbox("Inessive", value=True, help="Noun case: -ban / -ben")
        opt_prs_indef = st.checkbox("Present indefinite", value=True, help="Verb: present tense, indefinite")
    with colB:
        opt_ade = st.checkbox("Adessive", value=True, help="Noun case: -nál / -nél")
        opt_prs_def = st.checkbox("Present definite", value=False, help="Verb: present tense, definite")

    st.divider()
    st.subheader("Lemmas")
    source = st.radio("Choose your list", ["Sample list", "Upload CSVs"])

    if source == "Upload CSVs":
        with st.popover("How to format CSVs"):
            st.write("Upload two CSV files: one for nouns, one for verbs.")
            st.write("Each file must include a header row with a single column named lemma.")
            st.code("lemma\nház\nkönyv\nember\n", language="text")
            st.code("lemma\nlát\nír\nszeret\n", language="text")
            st.caption("Encoding: UTF-8. One lemma per line. No extra columns are required.")
        nouns_file = st.file_uploader("Upload nouns.csv", type=["csv"])
        verbs_file = st.file_uploader("Upload verbs.csv", type=["csv"])
    else:
        nouns_file = verbs_file = None

# --------------
# Load lemma pools
# --------------
SAMPLE_NOUNS = ["ház", "ablak", "asztal", "könyv", "ember", "város", "kert", "kút", "madár", "téma"]
SAMPLE_VERBS = ["lát", "ír", "olvas", "mond", "mos", "tör", "tanul", "szeret", "ad", "vesz"]

def read_lemma_csv(upload) -> list[str]:
    text = upload.getvalue().decode("utf-8")
    # Try DictReader for header "lemma"
    try:
        rows = list(csv.DictReader(io.StringIO(text)))
        if rows and "lemma" in rows[0]:
            return [normalize(r["lemma"]) for r in rows if r.get("lemma")]
    except Exception:
        pass
    # Fallback: first column per row
    return [normalize(r[0]) for r in csv.reader(io.StringIO(text)) if r]

def load_lemmas():
    nouns = SAMPLE_NOUNS[:]
    verbs = SAMPLE_VERBS[:]
    if nouns_file is not None:
        try:
            nouns = read_lemma_csv(nouns_file)
        except Exception:
            st.error("Could not read nouns.csv. Ensure it has a 'lemma' header and UTF-8 encoding.")
    if verbs_file is not None:
        try:
            verbs = read_lemma_csv(verbs_file)
        except Exception:
            st.error("Could not read verbs.csv. Ensure it has a 'lemma' header and UTF-8 encoding.")
    return sorted(set(nouns)), sorted(set(verbs))

nouns, verbs = load_lemmas()

# --------------
# Build tag pools (store raw POS codes WITHOUT brackets)
# --------------
NOUN_TAGS = []
if opt_ine: NOUN_TAGS.append(("/N", "[Ine]", "Inessive"))
if opt_ade: NOUN_TAGS.append(("/N", "[Ade]", "Adessive"))

VERB_TAGS = []
if opt_prs_indef:
    for p in ["1Sg", "2Sg", "3Sg", "1Pl", "2Pl", "3Pl"]:
        VERB_TAGS.append(("/V", f"[Prs.NDef.{p}]", f"Present indefinite · {PRONOUNS[p]}"))
if opt_prs_def:
    for p in ["1Sg", "2Sg", "3Sg", "1Pl", "2Pl", "3Pl"]:
        VERB_TAGS.append(("/V", f"[Prs.Def.{p}]", f"Present definite · {PRONOUNS[p]}"))

if not NOUN_TAGS and not VERB_TAGS:
    st.info("Choose at least one option in the sidebar to begin.")
    st.stop()

# --------------
# Build exercise pool
# --------------
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

if "pool" not in st.session_state:
    st.session_state.pool = build_pool()
if "idx" not in st.session_state:
    st.session_state.idx = 0
if "score" not in st.session_state:
    st.session_state.score = 0
if "seen" not in st.session_state:
    st.session_state.seen = 0
if "current" not in st.session_state:
    st.session_state.current = None

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

# Top controls
c1, c2, c3 = st.columns(3)
with c1:
    if st.button("Build / Reset pool"):
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

st.write("Type the correct form for:")
st.markdown(f'<span class="prompt-chip {st.session_state.current["chip_class"]}">{st.session_state.current["pretty"]}</span>', unsafe_allow_html=True)
st.caption(st.session_state.current["sublabel"])

user = st.text_input("Your answer", value="", placeholder="type here and press Check")
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

if check:
    gold = generate_form(st.session_state.current["lemma"], st.session_state.current["pos"], st.session_state.current["tag"])
    st.session_state.seen += 1
    if normalize(user) == normalize(gold):
        st.session_state.score += 1
        show_feedback(True, gold)
        st.session_state.idx += 1
        next_question()
    else:
        show_feedback(False, gold)

if reveal:
    gold = generate_form(st.session_state.current["lemma"], st.session_state.current["pos"], st.session_state.current["tag"])
    show_feedback(False, gold)
    st.session_state.idx += 1
    next_question()

if nxt:
    st.session_state.idx += 1
    next_question()
    fb.empty()

st.write(f"Score: {st.session_state.score} / {st.session_state.seen}")
st.progress(0 if st.session_state.seen == 0 else min(1.0, st.session_state.score / max(1, st.session_state.seen)))
st.caption("Forms generated via the emMorph tagset using NYTK’s Hungarian morphological generator.")
