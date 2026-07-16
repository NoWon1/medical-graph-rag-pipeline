
# =============================================================================
# cancer_app.py — MedChat v5  Three-Mode Graph RAG UI
#
# CHANGES FROM v4:
#   - Removed: USE_GRAPH_RAG import and status indicator
#               (replaced by explicit query_mode toggle)
#
#   - Added:   query_mode radio selector in sidebar
#               Three options driven entirely by config.py constants:
#               QUERY_MODE_AUTO / QUERY_MODE_RESEARCH / QUERY_MODE_GRAPH
#               Labels and descriptions come from config — no hardcoded strings
#
#   - Added:   query_mode passed into every call to:
#               generate_answer()        — non-streaming path
#               generate_answer_stream() — streaming path
#               render_streaming_answer() — wrapper function signature updated
#
#   - Added:   Mode badge displayed in chat header so user always knows
#               which mode is active
#
#   - Added:   stream_reasoning stored per turn — shows actual reasoning
#               path from cancer_retrieval.py in the expander, not a
#               generic placeholder string
#
#   - Added:   query_mode stored in session_state["query_mode"] so the
#               reasoning display can show which mode fired for each turn
#               even after the user changes the toggle
#
# UNCHANGED:
#   - All CSS styling
#   - render_message_with_images() — [IMAGE:] parsing untouched
#   - render_followup_buttons()    — untouched
#   - Patient report upload logic  — untouched
#   - Auto-analysis on report load — untouched (uses AUTO mode always)
#   - Tab structure                — untouched
#   - Chat history rendering       — untouched
# =============================================================================

import re
import hashlib
from pathlib import Path

import streamlit as st
import fitz

from cancer_retrieval import generate_answer, generate_answer_stream
from config import (
    IMAGE_DIR,
    QUERY_MODE_AUTO, QUERY_MODE_RESEARCH, QUERY_MODE_GRAPH,
    QUERY_MODE_DEFAULT, QUERY_MODE_LABELS, QUERY_MODE_DESCRIPTIONS,
)

# =============================================================================
# PAGE CONFIG
# =============================================================================

st.set_page_config(
    page_title="MedChat - Cancer Graph RAG Assistant",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# =============================================================================
# CUSTOM CSS — unchanged from v4
# =============================================================================

st.markdown("""
<style>
.stApp { background-color: #eef2f7; }
h1 { font-weight: 700; }
.medchat-title   { font-size: 42px; font-weight: 700; margin-bottom: 0; }
.medchat-dark    { color: #1f2a44; }
.medchat-green   { color: #2fa36b; }
.medchat-subtext { color: #5f6c7b; font-size: 16px; margin-top: 5px; }

section[data-testid="stSidebar"] {
    background-color: #ffffff;
    border-right: 1px solid #e0e6ed;
}
div[data-testid="stChatInput"] {
    border: none !important;
    box-shadow: none !important;
}
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

# =============================================================================
# CONSTANTS
# =============================================================================

CANCER_TYPE_OPTIONS = [
    "All", "breast", "lung", "melanoma",
    "leukemia", "osteosarcoma", "skin",
]

# Auto-analysis prompt — always runs in AUTO mode to get both
# graph drug/food context AND literature clinical context
_ANALYSIS_PROMPT = (
    "A patient has just uploaded their clinical report. "
    "Please analyse it carefully and provide a structured summary covering:\n"
    "1. **Diagnosis** — cancer type, subtype, any biomarkers or receptor status mentioned.\n"
    "2. **Stage** — TNM classification or overall stage if stated.\n"
    "3. **Treatment Plan** — therapies listed (surgery, chemo, radiation, targeted, immunotherapy).\n"
    "4. **Key Clinical Findings** — any notable lab values, imaging results, or pathology notes.\n"
    "5. **Relevant Literature** — based on the diagnosis and stage, what does the peer-reviewed "
    "clinical context say about prognosis, standard of care, or survival outcomes for this patient? "
    "Include any relevant figures or tables using [IMAGE: filename.png] if available.\n"
    "6. **Nutrition & Drug Interaction Guidance** — based on the treatment plan identified, "
    "what eating side effects should the patient expect and what foods should they avoid or eat? "
    "Are there any important interactions with common medications?\n\n"
    "Be empathetic, clear, and avoid unnecessary jargon. "
    "End with a reminder to consult a qualified oncologist for personalised decisions."
)

# Mode badge colours — shown in chat next to assistant messages
_MODE_BADGE = {
    QUERY_MODE_AUTO:     ("🔄", "#6c757d"),   # grey
    QUERY_MODE_RESEARCH: ("📄", "#0d6efd"),   # blue
    QUERY_MODE_GRAPH:    ("🕸️", "#198754"),   # green
}

# =============================================================================
# HELPER FUNCTIONS — unchanged from v4 except render_streaming_answer
# =============================================================================

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
    """
    Render assistant message text. Replaces [IMAGE: filename.png] tags
    with st.image() calls. All other text rendered as markdown.
    Unchanged from v3/v4.
    """
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
    """Render suggested follow-up question buttons. Unchanged from v4."""
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
    query:          str,
    patient_report: str,
    chat_history:   list[dict],
    cancer_filter:  str,
    query_mode:     str,          # ← NEW parameter v5
) -> tuple[str, list, list]:
    """
    Stream the answer token by token using st.write_stream().
    If the final answer contains [IMAGE:] tags, re-render with images.
    Returns (full_answer, sources, followups).
    """
    stream_container = st.empty()
    with stream_container.container():
        st.write_stream(
            generate_answer_stream(
                query=query,
                patient_report=patient_report,
                chat_history=chat_history,
                cancer_filter=cancer_filter,
                query_mode=query_mode,      # ← passed through v5
            )
        )

    full_answer = st.session_state.get("stream_buffer",   "")
    sources     = st.session_state.get("stream_sources",  [])
    followups   = st.session_state.get("stream_followups", [])

    # If images are present in the answer, replace the streamed text
    # with a properly rendered version that shows actual images
    if "[IMAGE:" in full_answer.upper():
        stream_container.empty()
        with stream_container.container():
            render_message_with_images(full_answer)

    return full_answer, sources, followups


def _run_auto_analysis(
    patient_context: str,
    cancer_filter:   str,
):
    """
    Automatically analyse an uploaded patient report.
    Always uses QUERY_MODE_AUTO so the analysis gets both:
      - Graph drug/food/interaction data specific to the treatment plan
      - Literature clinical context for prognosis and staging
    """
    st.session_state["analysed_report"] = _report_hash(patient_context)
    st.session_state.messages.append({
        "role": "user",
        "content": "Patient report loaded — please analyse it.",
    })

    with st.chat_message("user"):
        st.markdown("**Patient report loaded — please analyse it.**")

    with st.chat_message("assistant"):
        with st.spinner(
            "Analysing report against clinical literature "
            "and treatment knowledge graph..."
        ):
            try:
                # Auto-analysis always uses AUTO mode for maximum coverage
                answer, sources = generate_answer(
                    query=_ANALYSIS_PROMPT,
                    patient_report=patient_context,
                    chat_history=[],
                    cancer_filter=cancer_filter,
                    query_mode=QUERY_MODE_AUTO,
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
                st.session_state["followups"][turn_idx]     = []
                st.session_state["turn_modes"][turn_idx]    = QUERY_MODE_AUTO
                st.session_state["last_reasoning"][turn_idx] = (
                    st.session_state.get("stream_reasoning", "auto mode")
                )

            except Exception as e:
                st.error(f"Analysis error: {e}")
                st.exception(e)

# =============================================================================
# SESSION STATE INITIALISATION
# =============================================================================

def _init_session_state():
    if "messages" not in st.session_state:
        st.session_state.messages = [{
            "role": "assistant",
            "content": (
                "Hello! I can answer questions about cancer — treatment, "
                "staging, diagnosis, survival rates — and pull figures, "
                "flowcharts and tables directly from peer-reviewed literature.\n\n"
                "**Use the mode selector in the sidebar to choose:**\n"
                "- **Auto** — searches both literature and treatment knowledge graph\n"
                "- **Research & Literature** — clinical trials, survival data, staging\n"
                "- **Treatment & Nutrition** — drug side effects, food guidance, "
                "drug interactions, nutrition protocols\n\n"
                "**Tip:** Upload a patient report in the Upload Report tab for "
                "an automatic personalised analysis."
            ),
        }]

    defaults = {
        "analysed_report":    "",
        "followups":          {},
        "triggered_followup": "",
        "stream_buffer":      "",
        "stream_sources":     [],
        "stream_followups":   [],
        "stream_reasoning":   "",
        "last_reasoning":     {},   # turn_idx → reasoning path string
        "turn_modes":         {},   # turn_idx → query_mode used for that turn
    }
    for key, val in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = val

# =============================================================================
# INIT
# =============================================================================

_init_session_state()

# =============================================================================
# SIDEBAR
# Three sections: mode selector, cancer filter, app info
# =============================================================================

with st.sidebar:
    st.header("Settings")

    # ── Query mode selector ───────────────────────────────────────────────────
    # The core new feature in v5.
    # Radio buttons driven entirely by QUERY_MODE_LABELS from config.py.
    # No hardcoded strings here — change labels in config.py only.

    st.subheader("Search Mode")

    # Build ordered list: AUTO first (default), then RESEARCH, then GRAPH
    mode_options = [QUERY_MODE_AUTO, QUERY_MODE_RESEARCH, QUERY_MODE_GRAPH]
    mode_labels  = [QUERY_MODE_LABELS[m] for m in mode_options]

    selected_label = st.radio(
        label="Choose how to answer your question:",
        options=mode_labels,
        index=0,   # AUTO is default
        help="Switch modes based on what you want to know.",
    )

    # Map selected label back to mode constant
    query_mode = mode_options[mode_labels.index(selected_label)]

    # Show description for the selected mode
    st.caption(QUERY_MODE_DESCRIPTIONS[query_mode])

    # Visual mode indicator
    icon, colour = _MODE_BADGE[query_mode]
    st.markdown(
        f"<div style='padding:6px 10px; border-radius:6px; "
        f"background:#f0f4f8; font-size:13px; margin-top:6px;'>"
        f"{icon} <strong>Active:</strong> {QUERY_MODE_LABELS[query_mode]}"
        f"</div>",
        unsafe_allow_html=True,
    )

    st.divider()

    # ── Cancer type filter ────────────────────────────────────────────────────
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

    # ── App info ──────────────────────────────────────────────────────────────
    st.caption("MedChat v5 — Three-Mode Graph RAG")
    st.caption("Neo4j Aura · BAAI/bge-base-en · Groq llama-3.3-70b")

# =============================================================================
# HEADER
# =============================================================================

st.markdown("""
<h1 class="medchat-title">
    <span class="medchat-dark">MedChat – </span>
    <span class="medchat-green">Graph RAG Healthcare Assistant</span>
</h1>
<p class="medchat-subtext">
AI-powered system using medical knowledge graphs and peer-reviewed literature
to help cancer patients understand treatments, nutrition, and clinical guidance.
</p>
""", unsafe_allow_html=True)

st.caption(
    "Switch modes in the sidebar: Research for clinical literature, "
    "Treatment & Nutrition for drug/food guidance, Auto for both."
)

# =============================================================================
# TABS
# =============================================================================

tab_chat, tab_upload = st.tabs(["Chat", "Upload Report"])

# =============================================================================
# UPLOAD TAB — unchanged from v4
# =============================================================================

with tab_upload:
    st.subheader("Upload Patient Report")
    uploaded_file = st.file_uploader(
        "Upload report (.txt or .pdf)", type=["txt", "pdf"]
    )
    pasted_report = st.text_area(
        "Or paste report text here:",
        height=200,
        placeholder="e.g. Diagnosed with Stage II Breast Cancer. Current treatment: Tamoxifen...",
    )

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
        st.info("Switch to the 'Chat' tab to see your automated analysis.")
        with st.expander("Preview loaded report"):
            preview = patient_context[:800]
            if len(patient_context) > 800:
                preview += f"\n\n... ({len(patient_context) - 800} more characters)"
            st.text(preview)

# =============================================================================
# CHAT TAB
# =============================================================================

with tab_chat:

    # Read and clear any triggered follow-up before rendering inputs
    triggered_followup = st.session_state.get("triggered_followup", "")
    if triggered_followup:
        st.session_state["triggered_followup"] = ""

    chat_container = st.container()
    typed_query    = st.chat_input(
        "E.g., What foods should I avoid while on cisplatin?"
    )
    user_query = triggered_followup or typed_query

    with chat_container:

        # ── Render conversation history ───────────────────────────────────
        for idx, message in enumerate(st.session_state.messages):
            with st.chat_message(message["role"]):
                render_message_with_images(message["content"])

                # Reasoning path expander — shows mode + path for each turn
                if message["role"] == "assistant":
                    reasoning  = st.session_state["last_reasoning"].get(idx, "")
                    turn_mode  = st.session_state["turn_modes"].get(idx, "")
                    mode_label = QUERY_MODE_LABELS.get(turn_mode, "")

                    if reasoning and turn_mode:
                        icon = _MODE_BADGE.get(turn_mode, ("🔍", "#6c757d"))[0]
                        with st.expander(
                            f"{icon} {mode_label} — reasoning path",
                            expanded=False,
                        ):
                            st.caption(reasoning)

        last_turn_idx  = len(st.session_state.messages) - 1
        last_followups = st.session_state["followups"].get(last_turn_idx, [])

        # ── Auto-analyse uploaded report ──────────────────────────────────
        if patient_context:
            current_hash = _report_hash(patient_context)
            if current_hash != st.session_state.get("analysed_report", ""):
                _run_auto_analysis(patient_context, cancer_filter)

        # ── Show follow-ups when idle ─────────────────────────────────────
        if not user_query and last_followups:
            render_followup_buttons(last_followups, turn_key=str(last_turn_idx))

        # ── Handle new query ──────────────────────────────────────────────
        if user_query:

            with st.chat_message("user"):
                st.markdown(user_query)
            st.session_state.messages.append({
                "role": "user", "content": user_query
            })

            with st.chat_message("assistant"):
                try:
                    history_for_llm = st.session_state.messages[:-1]

                    # ── Stream the answer ─────────────────────────────────
                    # query_mode comes from the sidebar radio selector
                    full_answer, sources, followups = render_streaming_answer(
                        query=user_query,
                        patient_report=patient_context,
                        chat_history=history_for_llm,
                        cancer_filter=cancer_filter,
                        query_mode=query_mode,          # ← key v5 addition
                    )

                    # ── Sources expander ──────────────────────────────────
                    if sources:
                        with st.expander("Sources"):
                            for s in sources:
                                if s.get("url"):
                                    st.markdown(f"[{s['label']}]({s['url']})")
                                else:
                                    st.markdown(f"`{s['label']}`")

                    # ── Store in session state ────────────────────────────
                    st.session_state.messages.append({
                        "role": "assistant", "content": full_answer,
                    })
                    new_turn_idx = len(st.session_state.messages) - 1

                    st.session_state["followups"][new_turn_idx]     = followups
                    st.session_state["turn_modes"][new_turn_idx]    = query_mode
                    st.session_state["last_reasoning"][new_turn_idx] = (
                        st.session_state.get("stream_reasoning", "")
                    )

                    # Prune old follow-up state to avoid memory growth
                    if len(st.session_state["followups"]) > 10:
                        oldest = min(st.session_state["followups"].keys())
                        del st.session_state["followups"][oldest]

                except Exception as e:
                    st.error(f"Pipeline error: {e}")
                    st.exception(e)
                    followups    = []
                    new_turn_idx = len(st.session_state.messages) - 1

            # ── Follow-up buttons for this turn ───────────────────────────
            render_followup_buttons(
                st.session_state["followups"].get(new_turn_idx, []),
                turn_key=str(new_turn_idx),
            )

            # Rerun to clear the input box and re-render chat history
            if typed_query:
                st.rerun()


