import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from engine.processor import process_pdf_to_graph
from engine.generator import generate_graph_quiz, generate_essay_questions, evaluate_essay_response
from engine.graph_store import QuizGraphStore
from langchain_groq import ChatGroq
from dotenv import load_dotenv


load_dotenv()

# --- 1. GLOBAL CONFIGURATION ---
st.set_page_config(page_title="GraphRAG Tutor Pro", layout="wide", page_icon="🧠")

# This order ensures it works everywhere
groq_key = os.getenv("GROQ_API_KEY") or st.secrets.get("GROQ_API_KEY")

if not groq_key:
    st.error("Missing GROQ_API_KEY. Please check your .env or Streamlit Secrets.")
    st.stop()

# 3. Initialize the Free Open-Source Model
llm = ChatGroq(
    temperature=0, 
    groq_api_key=groq_key, 
    model_name="llama3-70b-8192" 
)
store = QuizGraphStore()

# Initialize Session States (CORRECTED & MATCHED)
keys_to_init = {
    "quiz": [],
    "essays": [],
    "responses": {},
    "quiz_submitted": False,
    "essay_results": {} # Matched name across the whole file
}

for key, val in keys_to_init.items():
    if key not in st.session_state:
        st.session_state[key] = val

# --- 2. SIDEBAR: KNOWLEDGE MANAGEMENT ---
with st.sidebar:
    st.header("📂 Knowledge Base")
    uploaded = st.file_uploader("Upload Lesson PDFs", type="pdf", accept_multiple_files=True)
    
    if st.button("🏗️ Build Knowledge Graph", use_container_width=True):
        if uploaded:
            existing = store.get_lessons()
            for file in uploaded:
                lesson_name = file.name.replace(".pdf", "")
                if lesson_name in existing:
                    st.info(f"⏭️ {lesson_name} already exists.")
                    continue
                with st.spinner(f"Mapping {file.name}..."):
                    process_pdf_to_graph(file, llm)
            st.success("Graph Updated!")
            st.rerun()
    
    st.divider()
    if st.button("🗑️ Reset All Knowledge", type="primary", use_container_width=True):
        store.wipe_database()
        st.rerun()

# --- 3. MAIN UI LAYOUT ---
st.title("🎓 Advanced GraphRAG Learning System")

tab_mcq, tab_essay, tab_insights = st.tabs(["🎯 Multiple Choice Quiz", "✍️ Essay Lab", "📊 Graph Insights"])

# --- TAB: MULTIPLE CHOICE QUIZ ---
with tab_mcq:
    existing_lessons = store.get_lessons()
    if not existing_lessons:
        st.info("👈 Upload and build a knowledge graph to start.")
    else:
        col_a, col_b = st.columns([2, 1])
        with col_a:
            selected_lesson = st.selectbox("Select Lesson", existing_lessons, key="mcq_select")
        with col_b:
            num_q = st.slider("Number of Questions", 1, 10, 5, key="mcq_slider")

        if st.button("✨ Generate Quiz", use_container_width=True):
            with st.spinner("Analyzing Graph..."):
                st.session_state.quiz = generate_graph_quiz(llm, selected_lesson, num_q)
                st.session_state.quiz_submitted = False
                st.session_state.responses = {}

        if st.session_state.quiz:
            for i, q in enumerate(st.session_state.quiz):
                st.subheader(f"Question {i+1} ({q.get('bloom_level', 'General')})")
                st.write(q['question'])
                st.session_state.responses[i] = st.radio(
                    "Select Answer:", options=range(len(q['options'])),
                    format_func=lambda x: f"{chr(65+x)}. {q['options'][x]}",
                    key=f"mcq_radio_{i}",
                    index=None if not st.session_state.quiz_submitted else st.session_state.responses.get(i)
                )

            if st.button("📊 Submit & Analyze Quiz"):
                if len(st.session_state.responses) < len(st.session_state.quiz):
                    st.warning("Please answer all questions.")
                else:
                    st.session_state.quiz_submitted = True
                    st.rerun()

            if st.session_state.quiz_submitted:
                st.divider()
                results_data = []
                score = 0
                for i, q in enumerate(st.session_state.quiz):
                    correct = st.session_state.responses[i] == q['correct_index']
                    if correct: score += 1
                    results_data.append({"Bloom Level": q.get('bloom_level', 'Unknown'), "Correct": correct})
                
                st.metric("Final Score", f"{score}/{len(st.session_state.quiz)}")
                st.bar_chart(pd.DataFrame(results_data).groupby("Bloom Level")["Correct"].mean() * 100)

# --- TAB: ESSAY LAB (ENHANCED DISPLAY) ---
with tab_essay:
    if not existing_lessons:
        st.info("👈 Build a knowledge graph first.")
    else:
        e_col1, e_col2 = st.columns([1, 2])
        with e_col1:
            e_lesson = st.selectbox("Select Lesson", existing_lessons, key="essay_select")
            if st.button("📝 Generate New Essay Prompt", use_container_width=True):
                st.session_state.essays = generate_essay_questions(llm, e_lesson, 1)

        if st.session_state.essays:
            st.divider()
            for i, q in enumerate(st.session_state.essays):
                # Using columns for a Split-Screen Experience
                q_col, a_col = st.columns([1, 1], gap="large")
                
                with q_col:
                    st.markdown(f"### ❓ Question {i+1}")
                    st.markdown(f"**Level:** `{q['bloom_level']}`")
                    st.info(q['question'])
                    with st.expander("View Grading Criteria"):
                        for item in q.get('rubric', []):
                            st.write(f"• {item}")
                
                with a_col:
                    st.markdown("### ✍️ Your Answer")
                    user_essay = st.text_area("Type your response here...", key=f"essay_input_{i}", height=350, label_visibility="collapsed")
                    
                    if st.button(f"Grade Response", key=f"grade_btn_{i}", use_container_width=True):
                        with st.spinner("Analyzing logical connections..."):
                            report = evaluate_essay_response(llm, q, user_essay)
                            st.session_state.essay_results[i] = report
                
                # Evaluation Display below the split-screen
                if i in st.session_state.essay_results:
                    res = st.session_state.essay_results[i]
                    st.markdown("---")
                    st.markdown("### 📊 Evaluation Report")
                    r_col1, r_col2 = st.columns([1, 3])
                    r_col1.metric("Grade", f"{res['score']}/10")
                    r_col2.success(f"**Feedback:** {res['feedback']}")
                    
                    if res.get('missed_entities'):
                        st.warning(f"**Missed Concepts:** {', '.join(res['missed_entities'])}")

# --- TAB: GRAPH INSIGHTS ---
with tab_insights:
    if existing_lessons:
        st.header("🔍 Knowledge Graph Structure")
        stats = store.query("MATCH (n) OPTIONAL MATCH (n)-[r]->(m) RETURN count(DISTINCT n) as nodes, count(r) as rels")
        st.write(f"This environment is currently powered by **{stats[0]['nodes']} concepts** and **{stats[0]['rels']} logical links**.")
        
        rels = store.query("MATCH (n)-[r]->(m) RETURN n.id as Source, type(r) as Relation, m.id as Target LIMIT 15")
        st.dataframe(pd.DataFrame(rels), use_container_width=True)