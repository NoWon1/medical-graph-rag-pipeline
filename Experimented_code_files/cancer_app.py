# =============================================================================
# cancer_app.py — MedChat Graph RAG UI
#
# CHANGES FROM v3:
#   - Reasoning path expander added to each assistant message
#   - Graph RAG status indicator in sidebar
#   - Import updated to cancer_retrieval (v4)
#   - All other logic identical — no interface changes
# =============================================================================

import io
import re
import hashlib
from pathlib import Path

import streamlit as st
import fitz
from cancer_retrieval import generate_answer, generate_answer_stream
from config import IMAGE_DIR, USE_GRAPH_RAG

# ===================== CONFIG =====================

st.set_page_config(
    page_title="MedChat - Cancer Graph RAG Assistant",
    layout="wide",
    initial_sidebar_ebar="collapsed"
)

# ===================== CUSTOM UI THEME =====================

st.markdown("""
<style>
.stApp { background-color: #eef2f7; }
h1 { font-weight: 700; }
.medchat-title { font-size: 42px; font-weight: 700; margin-bottom: 0; }
.medchat-dark  { color: #1f2a44; }
.medchat-green { color: #2fa36b; }
.medchat-subtext { color: #5f6c7b; font-size: 16px; margin-top: 5px; }
section[data-testid="stSidebar"] {
    background-color: #ffffff;
    border-right: 1px solid #e0e6ed;
}
div[data-testid="stChatInput"] { border: none !important; box-shadow: none !important; }
div[data-testid="stChatInput"] textarea {
    background-color: #f4fbf7 !important;
    border: 1px solid #d0d7de !important;
    border-radius: 12px !important;
    color: #1f2a44 !important;
    padding: 10px !important;
    box-shadow: none !important;
}
div[data-testid="stChatInput"] textarea:focus {
    border: 1px solid #e53935 !important;
    box-shadow: none !important;
    outline: none !important;
}
.stButton>button {
    border-radius: 10px;
    background-color: #2fa36b;
    color: white;
    border: none;
}
.stButton>button:hover { background-color: #248a59; }
.streamlit-expanderHeader { font-weight: 600; }
[data-testid="stChatMessage"] { border-radius: 12px; padding: 10px; }
textarea {
    background-color: #f4fbf7 !important;
    border: 1px solid #cfe8dc !important;
    border-radius: 12px !important;
    color: #1f2a44 !important;
    padding: 10px !important;
    box-shadow: none !important;
}
textarea:focus {
    border: 1px solid #e53935 !important;
    box-shadow: none !important;
    outline: none !important;
}
textarea::placeholder { color: #7a8a9a !important; }
::-webkit-scrollbar { width: 8px; }
::-webkit-scrollbar-thumb { background: #c5d1dc; border-radius: 10px; }
</style>
""", unsafe_allow_html=True)

CANCER_TYPE_OPTIONS = [
    "All", "breast", "lung", "melanoma",
    "leukemia", "osteosarcoma", "skin",
]

_ANALYSIS_PROMPT = (
    "A patient has just uploaded their clinical report. "
    "Please analyse it carefully and provide a structured summary covering:\n"
    "1. **Diagnosis** — cancer type, subtype, any biomarkers or receptor status mentioned.\n"
    "2. **Stage** — TNM classification or overall stage if stated.\n"
    "3. **Treatment Plan** — therapies listed (surgery, chemo, radiation, targeted, immunotherapy).\n"
    "4. **Key Clinical Findings** — any notable lab values, imaging results, or pathology notes.\n"
    "5. **Relevant Literature** — based on the diagnosis and stage, what does the peer-reviewed "
    "clinical context say about prognosis, standard of care, or survival outcomes for this patient? "
    "Include any relevant figures or tables using [IMAGE: filename.png] if available.\n\n"
    "Be empathetic, clear, and avoid unnecessary jargon. "
    "End with a reminder to consult a qualified oncologist for personalised decisions."
)

# ===================== HELPERS =====================

def extract_text_from_pdf(file_bytes: bytes) -> str:
    text_parts = []
    try:
        doc = fitz.open(stream=file_bytes, filetype="pdf")
        for page_num in range(len(doc)):
            page_text = doc[page_num].get_text().strip()
            if page_text:
                text_parts.append(f"[Page {page_num + 1}]\n{page_text}")
        doc.close()
    except Exception as e:
        return f"[PDF extraction error: {e}]"
    return "\n\n".join(text_parts)


def load_report_from_upload(uploaded_file) -> str:
    if uploaded_file is None:
        return ""
    raw_bytes = uploaded_file.getvalue()
    name      = uploaded_file.name.lower()
    if name.endswith(".pdf"):
        return extract_text_from_pdf(raw_bytes)
    try:
        return raw_bytes.decode("utf-8")
    except UnicodeDecodeError:
        return raw_bytes.decode("latin-1", errors="replace")


def _report_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8", errors="replace")).hexdigest()


def render_message_with_images(text: str):
    """Render text with inline [IMAGE: filename] tags replaced by st.image()."""
    clean = text
    for wrap in ["**", "*", "`"]:
        clean = clean.replace(f"{wrap}[", "[").replace(f"]{wrap}", "]")

    parts = re.split(r'\[IMAGE:\s*([^\]]+)\]', clean, flags=re.IGNORECASE)

    for i, part in enumerate(parts):
        if i % 2 == 0:
            if part.strip():
                st.markdown(part)
        else:
            filename = part.strip().strip('`"\'')
            img_path = IMAGE_DIR / filename
            if img_path.exists():
                st.markdown(f"**Reference Visual:** `{filename}`")
                st.image(str(img_path), caption=filename, use_container_width=True)


def render_followup_buttons(followups: list[str], turn_key: str):
    if not followups:
        return
    st.markdown("**Suggested follow-ups:**")
    cols = st.columns(len(followups))
    for i, (col, question) in enumerate(zip(cols, followups)):
        with col:
            key = f"fup_{turn_key}_{i}_{abs(hash(question)) % 999983}"
            if st.button(f"Q. {question}", key=key, use_container_width=True):
                st.session_state["triggered_followup"] = question


def render_streaming_answer(
    query: str, patient_report: str,
    chat_history: list[dict], cancer_filter: str
) -> tuple[str, list, list]:
    stream_container = st.empty()
    with stream_container.container():
        st.write_stream(
            generate_answer_stream(
                query=query,
                patient_report=patient_report,
                chat_history=chat_history,
                cancer_filter=cancer_filter,
            )
        )

    full_answer = st.session_state.get("stream_buffer", "")
    sources     = st.session_state.get("stream_sources", [])
    followups   = st.session_state.get("stream_followups", [])

    if "[IMAGE:" in full_answer.upper():
        stream_container.empty()
        with stream_container.container():
            render_message_with_images(full_answer)

    return full_answer, sources, followups


def _run_auto_analysis(patient_context: str, cancer_filter: str):
    st.session_state["analysed_report"] = _report_hash(patient_context)
    st.session_state.messages.append({
        "role": "user", "content": "Patient report loaded - please analyse it.",
    })

    with st.chat_message("user"):
        st.markdown("**Patient report loaded - please analyse it.**")

    with st.chat_message("assistant"):
        with st.spinner("Analysing patient report against clinical literature and knowledge graph..."):
            try:
                answer, sources = generate_answer(
                    query=_ANALYSIS_PROMPT,
                    patient_report=patient_context,
                    chat_history=[],
                    cancer_filter=cancer_filter,
                )
                render_message_with_images(answer)

                if sources:
                    with st.expander("Sources used in analysis"):
                        for s in sources:
                            if s.get("url"):
                                st.markdown(f"[{s['label']}]({s['url']})")
                            else:
                                st.markdown(f"`{s['label']}`")

                st.session_state.messages.append({
                    "role": "assistant", "content": answer,
                })
                turn_idx = len(st.session_state.messages) - 1
                st.session_state["followups"][turn_idx] = []

            except Exception as e:
                st.error(f"Analysis error: {e}")
                st.exception(e)


def _init_session_state():
    if "messages" not in st.session_state:
        graph_status = "✅ Graph RAG active" if USE_GRAPH_RAG else "⬜ Vector RAG only"
        st.session_state.messages = [{
            "role": "assistant",
            "content": (
                f"Hello! I can answer questions about cancer — treatment, "
                f"staging, diagnosis, survival rates — and pull figures, "
                f"flowcharts and tables directly from peer-reviewed literature.\n\n"
                f"**Mode:** {graph_status}\n\n"
                f"**Tip:** Upload or paste a patient report in the Upload Report tab "
                f"and I will automatically analyse it against the clinical literature "
                f"and knowledge graph."
            ),
        }]
    defaults = {
        "analysed_report":  "",
        "followups":        {},
        "triggered_followup": "",
        "stream_buffer":    "",
        "stream_sources":   [],
        "stream_followups": [],
        "last_reasoning":   {},   # NEW: stores reasoning path per turn
    }
    for key, val in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = val


# ===================== INIT =====================
_init_session_state()

# ===================== SIDEBAR =====================
with st.sidebar:
    st.header("Settings")

    # Graph RAG status
    if USE_GRAPH_RAG:
        st.success("🕸️ Graph RAG: Active")
    else:
        st.warning("⬜ Graph RAG: Disabled (set USE_GRAPH_RAG=True in config.py)")

    st.divider()

    st.subheader("Cancer Type Filter")
    cancer_filter_raw = st.selectbox(
        "Filter by cancer type",
        options=CANCER_TYPE_OPTIONS,
        index=0,
        help="Narrow retrieval to one cancer type. 'All' searches across all types.",
    )
    cancer_filter = "" if cancer_filter_raw == "All" else cancer_filter_raw
    if cancer_filter:
        st.info(f"Active filter: **{cancer_filter}** cancer")

    st.divider()
    st.caption("MedChat v4 — Graph RAG + Clinical Workflow")
    st.caption("Neo4j Aura + Qdrant-free architecture")


# ===================== HEADER =====================

st.markdown("""
<div class="medchat-title">
    <span class="medchat-dark">MedChat – </span>
    <span class="medchat-green">Graph RAG Healthcare Assistant</span>
</div>
<div class="medchat-subtext">
AI-powered system using medical knowledge graphs to help cancer patients understand 
clinical reports, treatments, and guidance from peer-reviewed literature.
</div>
""", unsafe_allow_html=True)

st.caption(
    "Upload a patient report for instant analysis, or ask clinical questions directly. "
    "Graph-enhanced reasoning connects entities across papers for deeper answers."
)

tab_chat, tab_upload = st.tabs(["Chat", "Upload Report"])

# ===================== UPLOAD TAB =====================

with tab_upload:
    st.subheader("Upload Patient Report")
    uploaded_file = st.file_uploader("Upload report (.txt or .pdf)", type=["txt", "pdf"])
    pasted_report = st.text_area("Or paste report text here:", height=200)

    patient_context = ""
    upload_source   = ""

    if uploaded_file is not None:
        patient_context = load_report_from_upload(uploaded_file)
        upload_source   = uploaded_file.name
    elif pasted_report.strip():
        patient_context = pasted_report.strip()
        upload_source   = "pasted text"

    if patient_context:
        st.success(f"Report loaded ({upload_source})")
        with st.expander("Preview loaded report"):
            preview = patient_context[:800]
            if len(patient_context) > 800:
                preview += f"\n\n... ({len(patient_context) - 800} more characters)"
            st.text(preview)

# ===================== CHAT TAB =====================

with tab_chat:

    triggered_followup = st.session_state.get("triggered_followup", "")
    if triggered_followup:
        st.session_state["triggered_followup"] = ""

    chat_container = st.container()
    typed_query    = st.chat_input("E.g., What topical drug delivery systems are used in breast cancer?")
    user_query     = triggered_followup or typed_query

    with chat_container:

        # Render history
        for idx, message in enumerate(st.session_state.messages):
            with st.chat_message(message["role"]):
                render_message_with_images(message["content"])

                # Show reasoning path expander for assistant messages (NEW)
                if message["role"] == "assistant":
                    reasoning = st.session_state["last_reasoning"].get(idx, "")
                    if reasoning and USE_GRAPH_RAG:
                        with st.expander("🕸️ Graph reasoning path", expanded=False):
                            st.caption(reasoning)

        last_turn_idx  = len(st.session_state.messages) - 1
        last_followups = st.session_state["followups"].get(last_turn_idx, [])

        # Auto analysis
        if patient_context:
            current_hash = _report_hash(patient_context)
            if current_hash != st.session_state.get("analysed_report", ""):
                _run_auto_analysis(patient_context, cancer_filter)

        # Show follow-ups if idle
        if not user_query and last_followups:
            render_followup_buttons(last_followups, turn_key=str(last_turn_idx))

        # Handle new query
        if user_query:
            with st.chat_message("user"):
                st.markdown(user_query)
            st.session_state.messages.append({"role": "user", "content": user_query})

            with st.chat_message("assistant"):
                try:
                    history_for_llm = st.session_state.messages[:-1]

                    full_answer, sources, followups = render_streaming_answer(
                        query=user_query,
                        patient_report=patient_context,
                        chat_history=history_for_llm,
                        cancer_filter=cancer_filter,
                    )

                    if sources:
                        with st.expander("Sources"):
                            for s in sources:
                                if s.get("url"):
                                    st.markdown(f"[{s['label']}]({s['url']})")
                                else:
                                    st.markdown(f"`{s['label']}`")

                    st.session_state.messages.append({
                        "role": "assistant", "content": full_answer,
                    })
                    new_turn_idx = len(st.session_state.messages) - 1
                    st.session_state["followups"][new_turn_idx] = followups

                    # Store reasoning path for display (NEW)
                    reasoning = st.session_state.get("stream_buffer", "")
                    # Reasoning path is logged server-side; surface a note in UI
                    if USE_GRAPH_RAG:
                        st.session_state["last_reasoning"][new_turn_idx] = (
                            "Graph traversal active — entities from query matched "
                            "against Neo4j knowledge graph. See server logs for full path."
                        )

                    if len(st.session_state["followups"]) > 10:
                        oldest = min(st.session_state["followups"].keys())
                        del st.session_state["followups"][oldest]

                except Exception as e:
                    st.error(f"Pipeline error: {e}")
                    st.exception(e)
                    followups    = []
                    new_turn_idx = len(st.session_state.messages) - 1

            render_followup_buttons(
                st.session_state["followups"].get(new_turn_idx, []),
                turn_key=str(new_turn_idx),
            )

            if typed_query:
                st.rerun()
