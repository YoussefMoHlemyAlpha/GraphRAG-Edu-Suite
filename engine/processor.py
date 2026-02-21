"""
processor.py — Phase 1 & 2 (Optimized & Stable)

🔍 Phase 1 — Ingestion & Extraction
    Step 1: Document Pre-processing (PDF → Text)
    Step 2: Context Batching (3500 chars/chunk for 4GB VRAM)
    Step 3: Sequential Extraction (Stable for limited VRAM)
    Step 4: Cypher Generation
"""

from __future__ import annotations
import json
import re
from typing import Callable, List, Dict, Any

import fitz  # PyMuPDF
from engine.graph_store import QuizGraphStore

# ─────────────────────────────────────────────
# Utility
# ─────────────────────────────────────────────

def _emit(callback: Callable | None, msg: str, pct: float = 0.0):
    if callback:
        try:
            callback(msg, pct)
        except:
            pass # Ignore UI sync issues
    else:
        print(msg)


def _extract_json(text: str) -> Any:
    """Robust JSON extraction, handling DeepSeek <think> tags."""
    # Remove <think>...</think> blocks if present
    text = re.sub(r'<think>[\s\S]*?</think>', '', text).strip()
    
    # 1. Try markdown fences
    fenced = re.search(r'```(?:json)?\s*([\s\S]*?)```', text, re.IGNORECASE)
    if fenced:
        try:
            return json.loads(fenced.group(1).strip())
        except json.JSONDecodeError:
            pass
    
    # ... (rest of the greedy logic remains same)

    # 2. Try the largest valid JSON block in the string
    arrays = re.findall(r'\[[\s\S]*\]', text)
    if arrays:
        for candidate in sorted(arrays, key=len, reverse=True):
            try:
                return json.loads(candidate)
            except json.JSONDecodeError:
                continue

    objects = re.findall(r'\{[\s\S]*\}', text)
    if objects:
        for candidate in sorted(objects, key=len, reverse=True):
            try:
                return json.loads(candidate)
            except json.JSONDecodeError:
                continue
    
    return None


# ─────────────────────────────────────────────
# Phase 1: Ingestion & Extraction
# ─────────────────────────────────────────────

def extract_text_from_pdf(pdf_file) -> str:
    pdf_file.seek(0)
    pdf_bytes = pdf_file.read()
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    text = ""
    for page in doc:
        text += page.get_text() + "\n"
    doc.close()
    return text.strip()


def perform_normal_chunking(text: str, target_size: int = 3500) -> List[str]:
    """Balanced chunk size for hardware with 4GB VRAM (Fast processing)."""
    paragraphs = re.split(r'\n\s*\n', text)
    chunks = []
    current_chunk = ""
    for p in paragraphs:
        if len(current_chunk) + len(p) > target_size and current_chunk:
            chunks.append(current_chunk.strip())
            current_chunk = p
        else:
            current_chunk += "\n\n" + p if current_chunk else p
    if current_chunk:
        chunks.append(current_chunk.strip())
    return chunks


def extract_graph_data(llm, chunk: str) -> Dict[str, Any]:
    # High-Yield Extraction Prompt (Optimized for DeepSeek & Llama 3.2)
    prompt = f"""Use the provided text to identify the most important concepts and their relationships for a Knowledge Graph.

TEXT:
{chunk}

TASK:
1. Extract 5-10 Entities (Concepts, Facts, Definitions).
2. Extract Relationships between these entities.
3. Return ONLY a JSON object.

FORMAT:
{{
  "entities": [
    {{"name": "...", "type": "...", "properties": ["..."]}}
  ],
  "relationships": [
    {{"source": "...", "relation": "...", "target": "..."}}
  ]
}}
"""
    try:
        response = llm.invoke(prompt).content
        print(f"--- LLM RAW RESPONSE ---\n{response}\n------------------------")
        data = _extract_json(response)
        if data:
            return data
    except Exception as e:
        print(f"⚠️ LLM Error: {e}")
    
    return {"entities": [], "relationships": []}


def generate_cypher_queries(extraction: Dict[str, Any], doc_id: str, lesson_name: str) -> List[str]:
    """Deterministic Cypher builder (Python) to avoid LLM hallucination in queries."""
    queries = []
    
    # Process Entities
    for ent in extraction.get("entities", []):
        name = ent.get("name", "").replace("'", "\\'")
        etype = ent.get("type", "Concept").replace(" ", "_")
        props = json.dumps(ent.get("properties", []))
        
        # Create Entity and link to Document
        queries.append(f"MERGE (e:Entity {{id: '{name}'}}) ON CREATE SET e.type='{etype}', e.properties={props}")
        queries.append(f"MATCH (e:Entity {{id: '{name}'}}), (d:Document {{id: '{doc_id}'}}) MERGE (e)-[:MENTIONED_IN]->(d)")

    # Process Relationships
    for rel in extraction.get("relationships", []):
        src = rel.get("source", "").replace("'", "\\'")
        tgt = rel.get("target", "").replace("'", "\\'")
        rtype = rel.get("relation", "RELATED_TO").upper().replace(" ", "_")
        
        if src and tgt:
            queries.append(f"MATCH (a:Entity {{id: '{src}'}}), (b:Entity {{id: '{tgt}'}}) MERGE (a)-[:{rtype}]->(b)")
            
    return queries


def validate_graph_schema(store: QuizGraphStore, lesson_name: str):
    store.query("""
    MATCH (doc:Document {lesson: $lesson})<-[:MENTIONED_IN]-(e:Entity)
    MATCH (les:Lesson {name: $lesson})
    MERGE (e)-[:PART_OF]->(les)
    """, {"lesson": lesson_name})


# ─────────────────────────────────────────────
# Execution Flow
# ─────────────────────────────────────────────

def process_pdf_to_graph(
    pdf_file,
    llm,
    vision_llm=None,
    status_callback: Callable | None = None,
) -> str:
    store = QuizGraphStore()
    lesson_name = pdf_file.name.replace(".pdf", "")
    doc_id = f"doc_{lesson_name.replace(' ', '_')}"

    from engine.vram_util import stop_ollama_model
    stop_ollama_model("llama3.2:latest") # Clear VRAM for DeepSeek
    
    _emit(status_callback, "📄 Extracting text…", 0.10)
    full_text = extract_text_from_pdf(pdf_file)

    if not full_text.strip():
        _emit(status_callback, "❌ Error: PDF extracted text is empty.", 0.15)
        raise ValueError("The PDF contains no machine-readable text. It might be a scan or images-only.")

    _emit(status_callback, "✂️ Batching context…", 0.20)
    chunks = perform_normal_chunking(full_text)
    
    _emit(status_callback, f"👁️ Extracting data from {len(chunks)} batches…", 0.30)
    all_data = {"entities": [], "relationships": []}
    
    for i, chunk in enumerate(chunks):
        pct = 0.30 + (0.40 * (i / len(chunks)))
        _emit(status_callback, f"  Processing batch {i+1}/{len(chunks)}…", pct)
        data = extract_graph_data(llm, chunk)
        all_data["entities"].extend(data.get("entities", []))
        all_data["relationships"].extend(data.get("relationships", []))

    if not all_data["entities"] and not all_data["relationships"]:
        _emit(status_callback, "❌ Error: Could not extract any knowledge.", 0.60)
        raise ValueError("No information could be extracted. Try a different PDF.")

    queries = generate_cypher_queries(all_data, doc_id, lesson_name)

    _emit(status_callback, f"💾 Building Knowledge Graph ({len(queries)} facts)…", 0.85)
    store.query("MERGE (l:Lesson {name: $name})", {"name": lesson_name})
    store.query("MERGE (d:Document {id: $id}) SET d.lesson = $name", {"id": doc_id, "name": lesson_name})
    store.query("MATCH (d:Document {id: $id}), (l:Lesson {name: $name}) MERGE (d)-[:BELONGS_TO]->(l)", {"id": doc_id, "name": lesson_name})
    for q in queries:
        try: store.query(q)
        except: pass

    _emit(status_callback, "🔍 Finalizing graph…", 0.95)
    validate_graph_schema(store, lesson_name)

    _emit(status_callback, f"🎉 '{lesson_name}' complete!", 1.0)
    return lesson_name
