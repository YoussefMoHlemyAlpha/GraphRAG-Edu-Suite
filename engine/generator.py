"""
generator.py — Phase 3: Retrieval & Reasoning (Simplified Llama 3.2)

🧠 Phase 3
    Step 7: Contextual Retrieval (Refined BFS)
    Step 8: Prompt Augmentation (Llama 3.2:latest)
    Step 9: Logic & Reasoning (Llama 3.2:latest)
    Step 9.5: Simple Critic loop (Optional/Integrated)
    Step 10: Final Output
"""

from __future__ import annotations
import json
import re
from typing import Callable, List, Dict, Any

from engine.graph_store import QuizGraphStore


# ─────────────────────────────────────────────
# Utilities
# ─────────────────────────────────────────────

def _emit(callback: Callable | None, msg: str, pct: float = 0.0):
    if callback:
        callback(msg, pct)
    else:
        print(msg)


def _parse_json(text: str, is_list: bool = True):
    """Robust extraction of the outermost JSON block."""
    start_char = '[' if is_list else '{'
    end_char = ']' if is_list else '}'
    
    # 1. Try markdown fences first
    fenced = re.search(r'```(?:json)?\s*([\s\S]*?)```', text, re.IGNORECASE)
    if fenced:
        try:
            return json.loads(fenced.group(1).strip())
        except: pass

    # 2. Greedy search for the outermost array/object
    try:
        start_idx = text.find(start_char)
        end_idx = text.rfind(end_char)
        if start_idx != -1 and end_idx != -1:
            candidate = text[start_idx:end_idx+1]
            return json.loads(candidate)
    except:
        pass
        
    return [] if is_list else {}


# ─────────────────────────────────────────────
# Step 7 — Contextual Retrieval (BFS)
# ─────────────────────────────────────────────

def _retrieve_graph_context(store: QuizGraphStore, lesson_name: str) -> tuple[str, list[str]]:
    query = """
    MATCH (les:Lesson {name: $lesson})<-[:PART_OF]-(e:Entity)
    OPTIONAL MATCH (e)-[r]->(related)
    RETURN 
        e.id AS entity, 
        e.type AS type, 
        e.properties AS props, 
        type(r) AS relation, 
        related.id AS target
    LIMIT 500
    """
    rows = store.query(query, {"lesson": lesson_name})

    if not rows:
        return "", []

    facts: list[str] = []
    entity_degree: dict[str, int] = {}

    for row in rows:
        ent     = row.get("entity") or ""
        etype   = row.get("type")   or ""
        props   = row.get("props")  or ""
        rel     = row.get("relation")
        target  = row.get("target")

        if not ent:
            continue

        entity_degree[ent] = entity_degree.get(ent, 0) + (1 if rel else 0)

        if props:
            facts.append(f"[{etype}] {ent}: {props}")
        if rel and target:
            facts.append(f"({ent})-[:{rel}]->({target})")

    sorted_entities = sorted(entity_degree.items(), key=lambda x: x[1], reverse=True)
    high_value      = [e for e, _ in sorted_entities[:10]]

    unique_facts = list(dict.fromkeys(facts))
    context_text = "\n".join(unique_facts)
    return context_text, high_value


# ─────────────────────────────────────────────
# Pipelines
# ─────────────────────────────────────────────

def generate_graph_quiz(
    llm,                 # Primary LLM (Llama 3.2)
    lesson_name: str,
    n: int,
    status_callback: Callable | None = None,
    **kwargs             # Optional critic_llm
) -> list[dict]:
    store = QuizGraphStore()
    critic_llm = kwargs.get("critic_llm") or llm

    from engine.vram_util import stop_ollama_model
    stop_ollama_model("deepseek-r1:8b") # Clear VRAM for Llama 3.2
    
    _emit(status_callback, "🔍 Step 7 — Retrieving context…", 0.10)
    context, high_value = _retrieve_graph_context(store, lesson_name)
    print(f"--- RETRIEVED CONTEXT ---\n{context}\n------------------------")

    if not context:
        return [{"question": "No data found.", "options": ["N/A"], "correct_index": 0}]

    # ── Phase 1: Generation ───────────────────────────────────────────
    _emit(status_callback, "🧠 Phase 1: Generating Initial Quiz…", 0.30)
    gen_prompt = f"""Use the provided context to generate a {n}-question MCQ quiz based on Bloom's Taxonomy.

Bloom's Levels to use: Knowledge, Comprehension, Application, Analysis, Synthesis, Evaluation.

CONTEXT:
{context}

STRICT RULES:
1. Each question must have EXACTLY 4 options.
2. Only one option must be correct.
3. Map each question to a Bloom's Taxonomy level.
4. Return ONLY a JSON array of objects.

FORMAT:
{{
  "question": "...",
  "options": ["A", "B", "C", "D"],
  "correct_index": 0,
  "bloom_level": "Analysis",
  "source_fact": "..."
}}
"""
    try:
        raw_gen = llm.invoke(gen_prompt).content
        initial_quiz = _parse_json(raw_gen)
        if not initial_quiz:
             return []
    except Exception as e:
        print(f"⚠️ Generation failed: {e}")
        return []

    # ── Phase 2: Critic Review ────────────────────────────────────────
    _emit(status_callback, "⚖️ Phase 2: Critic Review & Refinement…", 0.60)
    critic_prompt = f"""Review and correct the following quiz. 
ENSURE:
1. Exactly 4 options per question.
2. The 'correct_index' is mathematically correct (0 to 3).
3. The questions are grounded in the context provided.
4. Each question correctly reflects its assigned Bloom's Level.

CONTEXT:
{context}

QUIZ DATA:
{json.dumps(initial_quiz)}

Return ONLY the corrected JSON array.
"""
    try:
        raw_refined = critic_llm.invoke(critic_prompt).content
        print(f"--- QUIZ RAW RESPONSE ---\n{raw_refined}\n------------------------")
        refined_quiz = _parse_json(raw_refined)
        
        # Final validation and cleanup
        final_quiz = []
        for q in (refined_quiz or initial_quiz):
            if isinstance(q, dict) and "options" in q and len(q["options"]) == 4:
                # Force index sanity
                idx = q.get("correct_index", 0)
                if not isinstance(idx, int) or idx < 0 or idx >= 4:
                    q["correct_index"] = 0
                final_quiz.append(q)
        
        _emit(status_callback, "🎉 Step 10 — Ready!", 1.0)
        return final_quiz
    except Exception as e:
        print(f"⚠️ Critic failed: {e}")
        return initial_quiz[:n]
    
def _parse_json(text: str, is_list: bool = True):
    """Robust extraction of the outermost JSON block, cleaned for Llama/DeepSeek noise."""
    text = re.sub(r'<think>[\s\S]*?</think>', '', text).strip()
    start_char = '[' if is_list else '{'
    end_char = ']' if is_list else '}'
    
    try:
        start_idx = text.find(start_char)
        end_idx = text.rfind(end_char)
        if start_idx != -1 and end_idx != -1:
            candidate = text[start_idx:end_idx+1]
            return json.loads(candidate)
    except:
        pass
    return [] if is_list else {}


def generate_essay_questions(llm, lesson_name: str, n: int, **kwargs) -> list[dict]:
    store = QuizGraphStore()
    context, _ = _retrieve_graph_context(store, lesson_name)
    if not context: return []
    
    prompt = f"Using this context:\n{context}\n\nGenerate {n} essay questions. Return JSON list."
    try:
        raw = llm.invoke(prompt).content
        return _parse_json(raw, is_list=True)
    except:
        return []


def evaluate_essay_response(llm, question_obj: dict, student_text: str) -> dict:
    prompt = f"QUESTION: {question_obj['question']}\nANSWER: {student_text}\n\nGrade 0-10. Return JSON: {{'score': 0, 'feedback': ''}}"
    try:
        raw = llm.invoke(prompt).content
        return _parse_json(raw, is_list=False) or {"score": 0, "feedback": "Parse error."}
    except:
        return {"score": 0, "feedback": "Error."}
