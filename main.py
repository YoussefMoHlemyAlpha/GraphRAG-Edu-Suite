import os
import streamlit as st
import pandas as pd
from engine.processor import process_pdf_to_graph
from engine.generator import generate_graph_quiz, generate_essay_questions, evaluate_essay_response
from engine.graph_store import QuizGraphStore
from engine.rag_metrics import MetricsStore
from langchain_ollama import ChatOllama
import plotly.express as px
import plotly.graph_objects as go
from dotenv import load_dotenv

load_dotenv()

# ── Helper Functions ─────────────────────────────────────────────────────────

def check_ollama_models():
    """Check if Ollama models are available and warm them up."""
    try:
        import subprocess
        result = subprocess.run(['ollama', 'list'], capture_output=True, text=True, timeout=5)
        if result.returncode != 0:
            return False, "Ollama is not running"
        
        # Check if models exist
        output = result.stdout
        has_deepseek = 'deepseek-r1:1.5b' in output or 'deepseek-r1' in output
        has_llama = 'llama3.2' in output
        
        if not has_deepseek or not has_llama:
            return False, "Required models not found. Run: ollama pull deepseek-r1:1.5b && ollama pull llama3.2:latest"
        
        return True, "Models available"
    except Exception as e:
        return False, f"Error checking Ollama: {str(e)}"


def warmup_model(llm, model_name, timeout=10):
    """Warm up a model with a simple query."""
    try:
        import time
        start = time.time()
        response = llm.invoke("hi")
        elapsed = time.time() - start
        return True, f"Ready ({elapsed:.1f}s)"
    except Exception as e:
        error_msg = str(e)
        if "timeout" in error_msg.lower():
            return False, "Timeout - model may be loading"
        elif "connection" in error_msg.lower():
            return False, "Connection failed - restart Ollama"
        else:
            return False, f"Error: {error_msg[:100]}"

# ── Page Config ─────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="GraphRAG Edu-Suite with Quality Control",
    layout="wide",
    page_icon="🎓",
)

# ── Model Initialisation ─────────────────────────────────────────────────────
# Using Gemma3 (4B) for high-quality extraction and generation
extraction_llm = ChatOllama(
    model="gemma3:4b", 
    temperature=0,
    num_ctx=4096,
    timeout=90  # 90 second timeout for extraction
)

# Using Gemma3 (4B) for fast and reliable Quiz Generation with better factual accuracy
quiz_llm = ChatOllama(
    model="gemma3:4b",
    temperature=0,
    num_ctx=4096,
    timeout=60  # 60 second timeout
)

# Using Llama 3.2 for critique and review
critic_llm = ChatOllama(
    model="llama3.2:latest",
    temperature=0,
    num_ctx=4096,
    timeout=60
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
    "enable_metrics": False,  # NEW: Toggle for metrics
    "quiz_metrics": None,  # NEW: Store aggregate quiz metrics
    "essay_metrics": None,  # NEW: Store aggregate essay metrics
}.items():
    if key not in st.session_state:
        st.session_state[key] = val


# ── Sidebar: Knowledge Management ────────────────────────────────────────────
with st.sidebar:
    st.header("📂 Knowledge Ingestion")
    st.caption("Gemma3 High-Quality Extraction with Context Validation")
    
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

                # Skip existing lessons
                existing_lessons = store.get_lessons()
                if lesson_name in existing_lessons:
                    st.info(f"⏭️ Skipping '{lesson_name}': Already in Knowledge Graph.")
                    continue

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
st.title("🎓 GraphRAG Edu-Suite with Quality Control")
st.markdown("#### *Gemma3 Generation + Llama 3.2 Validation + RAG Metrics + Neo4j Knowledge Graph*")

# Metrics toggle in sidebar
with st.sidebar:
    st.divider()
    st.subheader("📊 RAG Metrics")
    st.session_state.enable_metrics = st.checkbox(
        "Enable Quality Metrics",
        value=st.session_state.enable_metrics,
        help="Calculate BLEU, Groundedness, and Hallucination Rate for generated content"
    )
    if st.session_state.enable_metrics:
        st.caption("✅ Metrics will be calculated and stored")
        
        # Hallucination threshold
        if "hallucination_threshold" not in st.session_state:
            st.session_state.hallucination_threshold = 0.40
        
        st.session_state.hallucination_threshold = st.slider(
            "Auto-Regenerate Threshold",
            min_value=0.0,
            max_value=1.0,
            value=st.session_state.hallucination_threshold,
            step=0.05,
            help="Automatically regenerate if hallucination rate exceeds this threshold"
        )
        st.caption(f"⚠️ Will regenerate if hallucination > {st.session_state.hallucination_threshold:.0%}")

tab_mcq, tab_essay, tab_insights, tab_analytics, tab_metrics = st.tabs(
    ["🎯 Multi-Choice Quiz (Quality Controlled)", "✍️ Essay Lab (Quality Controlled)", "📊 Graph Insights", "📈 Analytics", "📊 RAG Metrics"]
)


# ── TAB: MCQ ──────────────────────────────────────────────────────────────────
with tab_mcq:
    st.subheader("🎯 Multi-Choice Quiz Generator")
    st.caption("Gemma3 Generation + Llama 3.2 Validation + Coverage Filtering + Per-Question Metrics")
    existing_lessons = store.get_lessons()
    if not existing_lessons:
        st.info("👈 Upload a PDF to start building your knowledge base.")
    else:
        col_1, col_2 = st.columns([2, 1])
        selected_lesson = col_1.selectbox("Lesson Topic", existing_lessons)
        num_q = col_2.slider("Questions", 6, 10, 6)

        if st.button("✨ Generate Quiz", use_container_width=True):
            # Check Ollama status first
            ollama_ok, ollama_msg = check_ollama_models()
            if not ollama_ok:
                st.error(f"❌ Ollama Error: {ollama_msg}")
                st.info("💡 Make sure Ollama is running and models are installed")
                st.stop()
            
            with st.status("🧠 Orchestrating Generation Loop...", expanded=True) as status:
                st.write("🔍 Retrieving graph context...")
                
                # Optional warmup - don't fail if it doesn't work
                st.write("🔥 Warming up models...")
                success, message = warmup_model(quiz_llm, "Gemma3")
                if success:
                    st.write(f"✅ Gemma3 {message}")
                else:
                    st.warning(f"⚠️ Warmup skipped: {message}")
                    st.write("Continuing anyway...")
                
                max_attempts = 3  # Maximum regeneration attempts
                attempt = 0
                
                while attempt < max_attempts:
                    try:
                        if attempt > 0:
                            st.write(f"🔄 Regenerating (Attempt {attempt + 1}/{max_attempts})...")
                        
                        # Dual-Model Loop: Gemma3 for Gen, Llama 3.2 for Critic
                        st.write("📝 Generating questions...")
                        result = generate_graph_quiz(
                            quiz_llm, selected_lesson, num_q, 
                            critic_llm=critic_llm,
                            enable_metrics=st.session_state.enable_metrics
                        )
                        
                        st.write("✅ Generation complete, processing results...")
                        
                        # Handle return value (can be tuple or list)
                        if isinstance(result, tuple):
                            quiz, _ = result  # Ignore the flag
                        else:
                            quiz = result
                        
                        st.write(f"📊 Received {len(quiz) if quiz else 0} questions")
                        
                        if not quiz:
                            st.error("❌ No questions generated")
                            st.info("💡 This usually means:")
                            st.info("   • No data in knowledge graph for this lesson")
                            st.info("   • Context retrieval failed")
                            st.info("   • LLM generation returned empty")
                            status.update(label="❌ No Questions Generated", state="error", expanded=True)
                            break
                        
                        if quiz:
                            # Quiz is acceptable
                            st.session_state.quiz = quiz
                            st.session_state.quiz_submitted = False
                            st.session_state.responses = {}
                            
                            if st.session_state.enable_metrics:
                                # Count questions with metrics
                                with_metrics = sum(1 for q in quiz if 'metrics' in q)
                                st.write(f"✅ Quiz generated with quality metrics on {with_metrics}/{len(quiz)} questions")
                            else:
                                st.write("✅ Quiz generated and refined!")
                            
                            status.update(label="✨ Quiz Ready!", state="complete", expanded=False)
                            st.rerun()
                            break
                        else:
                            st.error("❌ Generation failed. Check terminal logs.")
                            status.update(label="❌ Generation Failed", state="error", expanded=True)
                            break
                            
                    except Exception as e:
                        import traceback
                        error_details = traceback.format_exc()
                        st.error(f"❌ Generation failed: {str(e)}")
                        
                        # Provide specific error guidance
                        if "Server disconnected" in str(e) or "Connection" in str(e):
                            st.error("🔌 Ollama server connection lost")
                            st.info("💡 Try: Restart Ollama and try again")
                        elif "timeout" in str(e).lower():
                            st.error("⏱️ Generation timed out")
                            st.info("💡 Try: Reduce number of questions or restart Ollama")
                        elif "No data found" in str(e):
                            st.error("📭 No data in knowledge graph")
                            st.info("💡 Try: Rebuild the knowledge graph for this lesson")
                        else:
                            st.error("Check terminal logs for details:")
                            with st.expander("Show Error Details"):
                                st.code(error_details)
                        
                        status.update(label="❌ Generation Failed", state="error", expanded=True)
                        break

        if st.session_state.quiz:
            if "question" in st.session_state.quiz[0] and st.session_state.quiz[0]["question"] == "No data found.":
                 st.warning("📭 No data found in the Graph for this lesson. Try rebuilding the graph.")
            
            for i, q in enumerate(st.session_state.quiz):
                st.subheader(f"Q{i+1}: {q.get('bloom_level', 'Question')}")
                st.write(q["question"])
                
                # Show per-question metrics if available
                if "metrics" in q:
                    col1, col2 = st.columns(2)
                    groundedness = q['metrics']['groundedness']
                    hallucination = q['metrics']['hallucination_rate']
                    
                    # Color code based on quality
                    ground_color = "normal" if groundedness >= 0.7 else "inverse"
                    hall_color = "inverse" if hallucination <= 0.33 else "normal"
                    
                    col1.metric(
                        "Groundedness", 
                        f"{groundedness:.1%}",
                        delta="Good" if groundedness >= 0.7 else "Low",
                        delta_color=ground_color
                    )
                    col2.metric(
                        "Hallucination", 
                        f"{hallucination:.1%}",
                        delta="Low" if hallucination <= 0.33 else "High",
                        delta_color=hall_color
                    )
                
                options = q.get("options", [])
                st.session_state.responses[i] = st.radio(
                    "Pick an answer:",
                    options=range(len(options)),
                    format_func=lambda x: options[x],
                    key=f"q_{i}",
                    index=None
                )
            
            # Display average hallucination rate at the bottom
            if any("metrics" in q for q in st.session_state.quiz):
                st.divider()
                st.subheader("📊 Overall Quiz Quality")
                
                # Calculate averages
                all_groundedness = [q['metrics']['groundedness'] for q in st.session_state.quiz if 'metrics' in q]
                all_hallucination = [q['metrics']['hallucination_rate'] for q in st.session_state.quiz if 'metrics' in q]
                
                if all_groundedness and all_hallucination:
                    avg_groundedness = sum(all_groundedness) / len(all_groundedness)
                    avg_hallucination = sum(all_hallucination) / len(all_hallucination)
                    
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        st.metric(
                            "Average Groundedness",
                            f"{avg_groundedness:.1%}",
                            help="How well questions are supported by source material"
                        )
                    
                    with col2:
                        hall_color = "🟢" if avg_hallucination <= 0.33 else "🔴"
                        st.metric(
                            "Average Hallucination",
                            f"{hall_color} {avg_hallucination:.1%}",
                            help="Average ratio of content not found in retrieved documents"
                        )
                    
                    with col3:
                        quality_score = (avg_groundedness * 0.5) + ((1 - avg_hallucination) * 0.5)
                        quality_label = "Excellent" if quality_score >= 0.8 else "Good" if quality_score >= 0.6 else "Fair"
                        st.metric(
                            "Overall Quality",
                            f"{quality_score:.1%}",
                            delta=quality_label,
                            help="Combined quality score"
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
    st.subheader("✍️ The Essay Lab")
    st.caption("Gemma3 Generation + Llama 3.2 Validation + Concept Verification + Per-Essay Metrics")
    
    existing_lessons = store.get_lessons()
    if not existing_lessons:
        st.info("👈 Build your knowledge graph first.")
    else:
        col1, col2 = st.columns([2, 1])
        essay_lesson = col1.selectbox("Choose a Theme", existing_lessons, key="essay_sel")
        essay_num = col2.slider("Question Count", 2, 5, 2, key="essay_num")
        
        if st.button("🎭 Generate Essay Prompts", use_container_width=True):
            # Check Ollama status first
            ollama_ok, ollama_msg = check_ollama_models()
            if not ollama_ok:
                st.error(f"❌ Ollama Error: {ollama_msg}")
                st.info("💡 Make sure Ollama is running and models are installed")
                st.stop()
            
            with st.status("🏗️ Synthesizing Deep Questions...", expanded=True) as status:
                # Optional warmup - don't fail if it doesn't work
                st.write("🔥 Warming up models...")
                success, message = warmup_model(quiz_llm, "Gemma3")
                if success:
                    st.write(f"✅ Gemma3 {message}")
                else:
                    st.warning(f"⚠️ Warmup skipped: {message}")
                    st.write("Continuing anyway...")
                
                max_attempts = 3  # Maximum regeneration attempts
                attempt = 0
                
                while attempt < max_attempts:
                    try:
                        if attempt > 0:
                            st.write(f"🔄 Regenerating (Attempt {attempt + 1}/{max_attempts})...")
                        
                        result = generate_essay_questions(
                            quiz_llm, essay_lesson, essay_num, 
                            critic_llm=critic_llm,
                            enable_metrics=st.session_state.enable_metrics
                        )
                        
                        # Handle return value (can be tuple or list)
                        if isinstance(result, tuple):
                            essays, _ = result  # Ignore the flag, we have per-essay metrics now
                        else:
                            essays = result
                        
                        if essays:
                            # Essays are acceptable
                            st.session_state.essays = essays
                            st.session_state.essay_results = {}
                            
                            if st.session_state.enable_metrics:
                                # Count essays with metrics
                                with_metrics = sum(1 for e in essays if 'metrics' in e)
                                st.write(f"✅ Essays generated with quality metrics on {with_metrics}/{len(essays)} questions")
                            else:
                                st.write("✅ Essays generated!")
                            
                            status.update(label="✅ Prompts Ready!", state="complete")
                            st.rerun()
                            break
                        else:
                            st.error("❌ Generation failed. Check terminal logs.")
                            status.update(label="❌ Generation Failed", state="error")
                            break
                            
                    except Exception as e:
                        import traceback
                        error_details = traceback.format_exc()
                        st.error(f"❌ Generation failed: {str(e)}")
                        
                        # Provide specific error guidance
                        if "Server disconnected" in str(e) or "Connection" in str(e):
                            st.error("🔌 Ollama server connection lost")
                            st.info("💡 Try: Restart Ollama and try again")
                        elif "timeout" in str(e).lower():
                            st.error("⏱️ Generation timed out")
                            st.info("💡 Try: Reduce number of questions or restart Ollama")
                        elif "No data found" in str(e):
                            st.error("📭 No data in knowledge graph")
                            st.info("💡 Try: Rebuild the knowledge graph for this lesson")
                        else:
                            st.error("Check terminal logs for details:")
                            with st.expander("Show Error Details"):
                                st.code(error_details)
                        
                        status.update(label="❌ Generation Failed", state="error")
                        break
        
        if st.session_state.essays:
            for i, essay in enumerate(st.session_state.essays):
                b_level = essay.get("bloom_level", "Synthesis")
                diff = essay.get("difficulty", "Challenge")
                
                with st.expander(f"Prompt {i+1}: {b_level} ({diff})", expanded=True):
                    # Show per-essay metrics if available (like MCQs)
                    if "metrics" in essay:
                        col1, col2 = st.columns(2)
                        groundedness = essay['metrics']['groundedness']
                        hallucination = essay['metrics']['hallucination_rate']
                        
                        # Color code based on quality
                        ground_color = "normal" if groundedness >= 0.7 else "inverse"
                        hall_color = "inverse" if hallucination <= 0.33 else "normal"
                        
                        col1.metric(
                            "Groundedness", 
                            f"{groundedness:.1%}",
                            delta="Good" if groundedness >= 0.7 else "Low",
                            delta_color=ground_color
                        )
                        col2.metric(
                            "Hallucination", 
                            f"{hallucination:.1%}",
                            delta="Low" if hallucination <= 0.33 else "High",
                            delta_color=hall_color
                        )
                    
                    st.write(f"**{essay['question']}**")
                    concepts = ", ".join(essay.get("expected_concepts", []))
                    st.caption(f"Focus on: {concepts}")
                    
                    answer = st.text_area("Your response:", key=f"ans_{i}", height=200)
                    
                    if st.button(f"🚀 Submit Answer {i+1}", key=f"sub_{i}"):
                        with st.spinner("⚖️ AI Grade Auditor in progress..."):
                            result = evaluate_essay_response(quiz_llm, essay, answer)
                            st.session_state.essay_results[i] = result
                    
                    if i in st.session_state.essay_results:
                        res = st.session_state.essay_results[i]
                        st.success(f"**Score: {res['score']}/10**")
                        st.info(f"**Feedback:** {res['feedback']}")
            
            # Display average metrics at the bottom (like MCQs)
            if any("metrics" in essay for essay in st.session_state.essays):
                st.divider()
                st.subheader("📊 Overall Essay Quality")
                
                # Calculate averages
                all_groundedness = [essay['metrics']['groundedness'] for essay in st.session_state.essays if 'metrics' in essay]
                all_hallucination = [essay['metrics']['hallucination_rate'] for essay in st.session_state.essays if 'metrics' in essay]
                
                if all_groundedness and all_hallucination:
                    avg_groundedness = sum(all_groundedness) / len(all_groundedness)
                    avg_hallucination = sum(all_hallucination) / len(all_hallucination)
                    
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        st.metric(
                            "Average Groundedness",
                            f"{avg_groundedness:.1%}",
                            help="How well essays are supported by source material"
                        )
                    
                    with col2:
                        hall_color = "🟢" if avg_hallucination <= 0.33 else "🔴"
                        st.metric(
                            "Average Hallucination",
                            f"{hall_color} {avg_hallucination:.1%}",
                            help="Average ratio of content not found in retrieved documents"
                        )
                    
                    with col3:
                        quality_score = (avg_groundedness * 0.5) + ((1 - avg_hallucination) * 0.5)
                        quality_label = "Excellent" if quality_score >= 0.8 else "Good" if quality_score >= 0.6 else "Fair"
                        st.metric(
                            "Overall Quality",
                            f"{quality_score:.1%}",
                            delta=quality_label,
                            help="Combined quality score"
                        )
        
        # Remove old aggregate metrics display (replaced by per-essay metrics above)
        # if st.session_state.essays and st.session_state.essay_metrics:
        #     ... (old code removed)


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


# ── TAB: ANALYTICS ────────────────────────────────────────────────────────────
with tab_analytics:
    st.subheader("📈 Performance Analytics")
    st.caption("Factual Mastery across Cognitive Dimensions")
    
    if not st.session_state.quiz_submitted or not st.session_state.quiz:
        st.info("Complete and submit a quiz to see your performance profile.")
    else:
        # Prepare data for Plotly
        data = []
        for i, q in enumerate(st.session_state.quiz):
            level = q.get("bloom_level", "Synthesis")
            is_correct = st.session_state.responses.get(i) == q.get("correct_index")
            data.append({"Level": level, "Correct": 1 if is_correct else 0, "Total": 1})
            
        df = pd.DataFrame(data).groupby("Level").sum().reset_index()
        df["Accuracy (%)"] = (df["Correct"] / df["Total"] * 100).round(1)
        
        # Sort by Revised Bloom's Order
        bloom_order = ["Remember", "Understand", "Apply", "Analyze", "Evaluate", "Create"]
        df["Level"] = pd.Categorical(df["Level"], categories=bloom_order, ordered=True)
        df = df.sort_values("Level")

        # 1. Bar Chart: Accuracy by Level
        fig_bar = px.bar(
            df, x="Level", y="Accuracy (%)", 
            color="Accuracy (%)", 
            color_continuous_scale="RdYlGn",
            range_y=[0, 100],
            title="Mastery by Bloom's Level"
        )
        st.plotly_chart(fig_bar, use_container_width=True)
        
        # 2. Radar Chart: Cognitive Profile
        if len(df) >= 3:
            fig_radar = px.line_polar(
                df, r="Accuracy (%)", theta="Level", 
                line_close=True,
                title="Cognitive Mastery Profile"
            )
            fig_radar.update_traces(fill='toself')
            st.plotly_chart(fig_radar, use_container_width=True)
        
        st.write("### 💡 Learning Recommendations")
        weak_levels = df[df["Accuracy (%)"] < 70]["Level"].tolist()
        if not weak_levels:
            st.success("🌟 Excellent! You've mastered all cognitive levels for this lesson.")
        else:
            levels_str = ", ".join([str(l) for l in weak_levels])
            st.warning(f"Focus on improving your **{levels_str}** skills. Try another quiz with more questions in these areas.")


# ── TAB: RAG METRICS ──────────────────────────────────────────────────────────
with tab_metrics:
    st.header("📊 RAG Quality Metrics Dashboard")
    st.caption("Track Groundedness, Hallucination Rate, and Overall Quality across all generated content")
    
    if not st.session_state.enable_metrics:
        st.info("👈 Enable 'Quality Metrics' in the sidebar to start tracking RAG performance.")
    else:
        metrics_store = MetricsStore()
        
        col1, col2 = st.columns([2, 1])
        with col1:
            filter_lesson = st.selectbox(
                "Filter by Lesson",
                ["All Lessons"] + store.get_lessons(),
                key="metrics_lesson_filter"
            )
        with col2:
            if st.button("🔄 Refresh Metrics", use_container_width=True):
                st.rerun()
        
        # Get summary
        lesson_filter = None if filter_lesson == "All Lessons" else filter_lesson
        summary = metrics_store.get_summary(lesson_name=lesson_filter)
        
        if summary['count'] == 0:
            st.warning("No metrics data available yet. Generate some quizzes or essays with metrics enabled!")
        else:
            # Display summary statistics
            st.subheader(f"📈 Summary Statistics ({summary['count']} evaluations)")
            
            col1, col2, col3 = st.columns(3)
            
            # Groundedness
            if summary['groundedness']['mean'] is not None:
                with col1:
                    st.metric(
                        "Avg Groundedness",
                        f"{summary['groundedness']['mean']:.1%}",
                        help="How well generated content is supported by retrieved sources"
                    )
                    st.caption(f"Range: {summary['groundedness']['min']:.1%} - {summary['groundedness']['max']:.1%}")
            
            # Hallucination Rate
            if summary['hallucination_rate']['mean'] is not None:
                with col2:
                    st.metric(
                        "Avg Hallucination Rate",
                        f"{summary['hallucination_rate']['mean']:.1%}",
                        delta=f"-{summary['hallucination_rate']['mean']:.1%}",
                        delta_color="inverse",
                        help="Ratio of words not found in retrieved documents"
                    )
                    st.caption(f"Range: {summary['hallucination_rate']['min']:.1%} - {summary['hallucination_rate']['max']:.1%}")
            
            # BLEU Score
            if summary['bleu']['mean'] is not None:
                with col3:
                    st.metric(
                        "Avg BLEU Score",
                        f"{summary['bleu']['mean']:.3f}",
                        help="Similarity to reference texts (when available)"
                    )
                    st.caption(f"Range: {summary['bleu']['min']:.3f} - {summary['bleu']['max']:.3f}")
            
            st.divider()
            
            # Visualizations
            metrics_list = metrics_store.load_metrics()
            if lesson_filter:
                metrics_list = [m for m in metrics_list if m.metadata.get('lesson_name') == lesson_filter]
            
            if metrics_list:
                # Prepare data for visualization
                viz_data = []
                for i, m in enumerate(metrics_list):
                    viz_data.append({
                        'Index': i + 1,
                        'Groundedness': m.groundedness,
                        'Hallucination Rate': m.hallucination_rate,
                        'BLEU': m.bleu if m.bleu is not None else 0,
                        'Question Type': m.metadata.get('question_type', 'unknown'),
                        'Bloom Level': m.metadata.get('bloom_level', 'Unknown')
                    })
                
                viz_df = pd.DataFrame(viz_data)
                
                # Groundedness over time
                st.subheader("📊 Groundedness Trend")
                fig_ground = px.line(
                    viz_df, x='Index', y='Groundedness',
                    title='Groundedness Score Over Evaluations',
                    markers=True
                )
                fig_ground.add_hline(y=0.7, line_dash="dash", line_color="green", 
                                    annotation_text="Target: 70%")
                st.plotly_chart(fig_ground, use_container_width=True)
                
                # Hallucination rate over time
                st.subheader("🚨 Hallucination Rate Trend")
                fig_hall = px.line(
                    viz_df, x='Index', y='Hallucination Rate',
                    title='Hallucination Rate Over Evaluations',
                    markers=True,
                    color_discrete_sequence=['red']
                )
                fig_hall.add_hline(y=0.3, line_dash="dash", line_color="orange",
                                  annotation_text="Warning: 30%")
                st.plotly_chart(fig_hall, use_container_width=True)
                
                # Metrics by Bloom's Level
                if 'Bloom Level' in viz_df.columns:
                    st.subheader("🎯 Metrics by Bloom's Taxonomy Level")
                    bloom_metrics = viz_df.groupby('Bloom Level').agg({
                        'Groundedness': 'mean',
                        'Hallucination Rate': 'mean'
                    }).reset_index()
                    
                    fig_bloom = go.Figure()
                    fig_bloom.add_trace(go.Bar(
                        name='Groundedness',
                        x=bloom_metrics['Bloom Level'],
                        y=bloom_metrics['Groundedness'],
                        marker_color='lightblue'
                    ))
                    fig_bloom.add_trace(go.Bar(
                        name='Hallucination Rate',
                        x=bloom_metrics['Bloom Level'],
                        y=bloom_metrics['Hallucination Rate'],
                        marker_color='lightcoral'
                    ))
                    fig_bloom.update_layout(
                        title='Average Metrics by Cognitive Level',
                        barmode='group',
                        yaxis_title='Score'
                    )
                    st.plotly_chart(fig_bloom, use_container_width=True)
            
            st.divider()
            
            # Export options
            st.subheader("💾 Export Data")
            col1, col2 = st.columns(2)
            with col1:
                if st.button("📥 Export to CSV", use_container_width=True):
                    output_file = "rag_metrics_export.csv"
                    metrics_store.export_to_csv(output_file)
                    st.success(f"✅ Exported to {output_file}")
            
            with col2:
                if st.button("🗑️ Clear Metrics", type="secondary", use_container_width=True):
                    import os
                    if os.path.exists("rag_metrics.jsonl"):
                        os.remove("rag_metrics.jsonl")
                        st.success("✅ Metrics cleared")
                        st.rerun()

