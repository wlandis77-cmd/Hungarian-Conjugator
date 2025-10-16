import io
import csv
import random
import unicodedata
import streamlit as st

# ----------------------------
# Utilities
# ----------------------------
def normalize(s: str) -> str:
    return unicodedata.normalize("NFC", s.strip().lower())

@st.cache_resource(show_spinner=True)
def load_generator():
    from transformers import pipeline
    # Hugging Face model trained for Hungarian generation with emMorph tags
    # Model card shows the "morph: " prefix and example: "morph: munka [/N][Acc]"
    # https://huggingface.co/NYTK/morphological-generator-emmorph-mt5-hungarian
    return pipeline(
        task="text2text-generation",
        model="NYTK/morphological-generator-emmorph-mt5-hungarian"
    )

def build_prompt(lemma: str, pos: str, tag: str) -> str:
    # pos is "/N" for nouns or "/V" for verbs; tag is something like "[Ine]" or "[Prs.NDef.1Sg]"
    return f"morph: {lemma} [{pos}]{tag}"

def generate_form(lemma: str, pos: str, tag: str) -> str:
    pipe = load_generator()
    prompt = build_prompt(lemma, pos, tag)
    out = pipe(prompt, max_new_tokens=8)[0]["generated_text"]
    return out.strip()

# ----------------------------
# App setup
# ----------------------------
st.set_page_config(page_title="Hungarian Morphology Trainer", page_icon="🇭🇺", layout="centered")
st.title("Hungarian Verb and Noun Trainer")
st.write("Pick what to practice, then type the correct surface form. Tags match the emMorph code list, so generated answers follow standard Hungarian morphology.")

# Sidebar controls
with st.sidebar:
    st.header("Practice settings")

    st.subheader("Noun cases")
    use_ine = st.checkbox("Inessive [Ine]  “-ban/-ben”", value=True)
    use_ade = st.checkbox("Adessive [Ade]  “-nál/-nél”", value=True)

    st.subheader("Verbs, present tense")
    use_prs_indef = st.checkbox("Present Indefinite [Prs.NDef.*]", value=True)
    use_prs_def   = st.checkbox("Present Definite [Prs.Def.*]", value=False)

    st.subheader("Lemma sources")
    source = st.radio("Pick your pool", ["Sample list", "Upload CSVs"])
    nouns_upload = None
    verbs_upload = None
    if source == "Upload CSVs":
        st.caption("Upload two simple CSV files. One lemma per line. No header. UTF 8.")
        nouns_upload = st.file_uploader("nouns.csv", type=["csv"])
        verbs_upload = st.file_uploader("verbs.csv", type=["csv"])

    st.caption("Tag reference comes from the official emMorph code list.")
    st.markdown("[Open emMorph code list](https://e-magyar.hu/en/textmodules/emmorph_codelist)")

# Sample lemma pools so the app runs even before you upload
SAMPLE_NOUNS = ["ház", "ablak", "asztal", "könyv", "ember", "város", "kert", "kút", "madár", "téma"]
SAMPLE_VERBS = ["lát", "ír", "olvas", "mond", "mos", "tör", "tanul", "szeret", "ad", "vesz"]

def load_lemmas():
    nouns = SAMPLE_NOUNS[:]
    verbs = SAMPLE_VERBS[:]
    if source == "Upload CSVs":
        if nouns_upload is not None:
            text = nouns_upload.getvalue().decode("utf-8")
            nouns = [normalize(row[0]) for row in csv.reader(io.StringIO(text)) if row]
        if verbs_upload is not None:
            text = verbs_upload.getvalue().decode("utf-8")
            verbs = [normalize(row[0]) for row in csv.reader(io.StringIO(text)) if row]
    # de duplicate and sort for a tidy UI
    return sorted(set(nouns)), sorted(set(verbs))

nouns, verbs = load_lemmas()

# Build tag sets from emMorph codes
NOUN_TAGS = []
if use_ine: NOUN_TAGS.append(("/N", "[Ine]"))
if use_ade: NOUN_TAGS.append(("/N", "[Ade]"))

VERB_TAGS = []
if use_prs_indef:
    VERB_TAGS += [("/V", f"[Prs.NDef.{p}]") for p in ["1Sg","2Sg","3Sg","1Pl","2Pl","3Pl"]]
if use_prs_def:
    VERB_TAGS += [("/V", f"[Prs.Def.{p}]") for p in ["1Sg","2Sg","3Sg","1Pl","2Pl","3Pl"]]

if not any([NOUN_TAGS, VERB_TAGS]):
    st.info("Choose at least one case or verb paradigm in the sidebar.")
    st.stop()

# Build a pool of exercises and avoid repeats within a cycle
def build_pool():
    items = []
    for lemma in nouns:
        for _, tag in NOUN_TAGS:
            items.append(("N", lemma, "/N", tag))
    for lemma in verbs:
        for _, tag in VERB_TAGS:
            items.append(("V", lemma, "/V", tag))
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
    kind, lemma, pos, tag = st.session_state.pool[st.session_state.idx]
    pretty = f"{lemma}  +  {tag.strip('[]')}  ({'noun' if kind=='N' else 'verb'})"
    st.session_state.current = dict(kind=kind, lemma=lemma, pos=pos, tag=tag, pretty=pretty)

# Controls to rebuild or reshuffle
colA, colB, colC = st.columns(3)
with colA:
    if st.button("Build or reset pool"):
        st.session_state.pool = build_pool()
        st.session_state.idx = 0
        st.session_state.score = 0
        st.session_state.seen = 0
        st.session_state.current = None
with colB:
    if st.button("Reshuffle remaining"):
        tail = st.session_state.pool[st.session_state.idx:]
        random.shuffle(tail)
        st.session_state.pool[st.session_state.idx:] = tail
with colC:
    st.caption(f"Items left this cycle: {max(0, len(st.session_state.pool) - st.session_state.idx)}")

if st.session_state.current is None:
    next_question()

st.subheader("Type the correct form for:")
st.markdown(f"### {st.session_state.current['pretty']}")

answer = st.text_input("Your answer", value="", placeholder="type here then click Check")
col1, col2, col3 = st.columns([1,1,1])
go = col1.button("Check", type="primary")
reveal = col2.button("Show answer")
skip = col3.button("Next")

fb = st.empty()

if go:
    gold = generate_form(st.session_state.current["lemma"], st.session_state.current["pos"], st.session_state.current["tag"])
    st.session_state.seen += 1
    if normalize(answer) == normalize(gold):
        st.session_state.score += 1
        fb.success(f"Correct: {gold}")
        st.session_state.idx += 1
        next_question()
    else:
        fb.error(f"Not quite. Expected: {gold}")

if reveal:
    gold = generate_form(st.session_state.current["lemma"], st.session_state.current["pos"], st.session_state.current["tag"])
    fb.info(f"Answer: {gold}")
    st.session_state.idx += 1
    next_question()

if skip:
    st.session_state.idx += 1
    next_question()
    fb.empty()

st.write(f"Score: {st.session_state.score} / {st.session_state.seen}")
st.progress(0 if st.session_state.seen == 0 else min(1.0, st.session_state.score / max(1, st.session_state.seen)))
st.caption("Forms are generated with the emMorph tagset and the NYTK mT5 model.")
