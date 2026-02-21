import os
import streamlit as st
import pandas as pd
from engine.processor import process_pdf_to_graph
from engine.generator import generate_graph_quiz, generate_essay_questions, evaluate_essay_response
from engine.graph_store import QuizGraphStore
from langchain_ollama import ChatOllama
from dotenv import load_dotenv

load_dotenv()

# ── Page Config ─────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="GraphRAG Tutor 3.0 (Fast)",
    layout="wide",
    page_icon="🧠",
)

# ── Model Initialisation ─────────────────────────────────────────────────────
# Using DeepSeek-R1 (8B) for high-quality extraction (slow on 4GB VRAM)
extraction_llm = ChatOllama(
    model="deepseek-r1:8b", 
    temperature=0,
    num_ctx=4096
)

# Using Llama 3.2 (3B) for fast and reliable Quiz Generation
quiz_llm = ChatOllama(
    model="llama3.2:latest", 
    temperature=0,
    num_ctx=4096
)
# Alias as 'llm' for backward compatibility or general use
llm = quiz_llm 

store = QuizGraphStore()

# ── Session State ────────────────────────────────────────────────────────────
for key, val in {
    "quiz": [],
    "essays": [],
    "responses": {},
    "quiz_submitted": False,
    "essay_results": {},
}.items():
    if key not in st.session_state:
        st.session_state[key] = val


# ── Sidebar: Knowledge Management ────────────────────────────────────────────
with st.sidebar:
    st.header("📂 Knowledge Ingestion")
    st.caption("Phase 1: Fast Ingestion Pipeline")
    
    uploaded = st.file_uploader(
        "Upload Lesson PDFs", type="pdf", accept_multiple_files=True
    )

    if st.button("🏗️ Build Knowledge Graph", use_container_width=True):
        if not uploaded:
            st.warning("⚠️ Please upload at least one PDF first.")
        else:
            existing = store.get_lessons()
            for file in uploaded:
                lesson_name = file.name.replace(".pdf", "")
                if lesson_name in existing:
                    st.info(f"⏭️ '{lesson_name}' already exists.")
                    continue

                st.markdown(f"---\n**Processing:** `{lesson_name}`")
                status_text = st.empty()
                progress_bar = st.progress(0.0)

                def make_callback(status_el, prog_el):
                    def callback(msg: str, pct: float):
                        status_el.info(msg)
                        prog_el.progress(min(max(pct, 0.0), 1.0))
                    return callback

                cb = make_callback(status_text, progress_bar)

                try:
                    # Uses DeepSeek-R1 (8B) for high-quality extraction
                    process_pdf_to_graph(file, extraction_llm, status_callback=cb)
                    st.toast(f"✅ '{lesson_name}' ingested successfully!")
                except Exception as e:
                    import traceback
                    st.error(f"❌ Failed to process '{file.name}': {e}")
                    with st.expander("Show Technical details"):
                        st.code(traceback.format_exc())

            st.rerun()

    st.divider()
    if st.button("🗑️ Reset Database", type="primary", use_container_width=True):
        store.wipe_database()
        st.toast("🗑️ Database wiped.")
        st.rerun()


# ── Main UI ───────────────────────────────────────────────────────────────────
st.title("🎓 Advanced GraphRAG Edu-Suite")
st.markdown("#### *Llama 3.2 Intelligence + Neo4j Memory (Fast Mode)*")

tab_mcq, tab_essay, tab_insights = st.tabs(
    ["🎯 Multi-Choice Quiz", "✍️ Essay Lab", "📊 Graph Insights"]
)


# ── TAB: MCQ ──────────────────────────────────────────────────────────────────
with tab_mcq:
    existing_lessons = store.get_lessons()
    if not existing_lessons:
        st.info("👈 Upload a PDF to start building your knowledge base.")
    else:
        col_1, col_2 = st.columns([2, 1])
        selected_lesson = col_1.selectbox("Lesson Topic", existing_lessons)
        num_q = col_2.slider("Questions", 1, 10, 5)

        if st.button("✨ Generate Quiz", use_container_width=True):
            with st.status("🧠 Orchestrating Generation Loop...", expanded=True) as status:
                st.write("🔍 Retrieving graph context...")
                # Pass quiz_llm for generation and critic
                quiz = generate_graph_quiz(
                    quiz_llm, selected_lesson, num_q
                )
                
                if quiz:
                    st.session_state.quiz = quiz
                    st.session_state.quiz_submitted = False
                    st.session_state.responses = {}
                    st.write("✅ Quiz generated and refined!")
                    status.update(label="✨ Quiz Ready!", state="complete", expanded=False)
                    st.rerun()
                else:
                    st.error("❌ Generation failed. Check terminal logs.")
                    status.update(label="❌ Generation Failed", state="error", expanded=True)

        if st.session_state.quiz:
            if "question" in st.session_state.quiz[0] and st.session_state.quiz[0]["question"] == "No data found.":
                 st.warning("📭 No data found in the Graph for this lesson. Try rebuilding the graph.")
            
            for i, q in enumerate(st.session_state.quiz):
                st.subheader(f"Q{i+1}: {q.get('bloom_level', 'Question')}")
                st.write(q["question"])
                
                options = q.get("options", [])
                st.session_state.responses[i] = st.radio(
                    "Pick an answer:",
                    options=range(len(options)),
                    format_func=lambda x: options[x],
                    key=f"q_{i}",
                    index=None
                )
            
            if st.button("Submit Quiz"):
                st.session_state.quiz_submitted = True
                st.rerun()

        if st.session_state.quiz_submitted:
            score = 0
            for i, q in enumerate(st.session_state.quiz):
                idx = q.get("correct_index", 0)
                options = q.get("options", [])
                
                # Safety check
                is_valid = isinstance(idx, int) and 0 <= idx < len(options)
                correct = st.session_state.responses.get(i) == idx
                
                if correct: score += 1
                
                ans_text = options[idx] if is_valid else "Unknown"
                st.write(f"**Q{i+1}:** {'✅' if correct else '❌'} (Correct: {ans_text})")
            st.metric("Final Score", f"{score}/{len(st.session_state.quiz)}")


# ── TAB: ESSAY ────────────────────────────────────────────────────────────────
with tab_essay:
    st.info("The Essay Lab is powered by Llama 3.2 synthesis.")
    # (Essay logic implemented via generator)
    pass


# ── TAB: INSIGHTS ─────────────────────────────────────────────────────────────
with tab_insights:
    st.header("🕸️ Knowledge Graph Stats")
    stats_rows = store.query("MATCH (n) OPTIONAL MATCH (n)-[r]->(m) RETURN count(DISTINCT n) AS nodes, count(r) AS rels")
    if stats_rows:
        stats = stats_rows[0]
        c1, c2 = st.columns(2)
        c1.metric("Nodes", stats["nodes"])
        c2.metric("Relationships", stats["rels"])

    st.write("### Recent Extracted Relationships")
    rows = store.query("MATCH (a:Entity)-[r]->(b:Entity) RETURN a.id as Source, type(r) as Relation, b.id as Target LIMIT 20")
    if rows:
        st.table(pd.DataFrame(rows))