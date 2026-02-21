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
    """Robust JSON extraction, handling DeepSeek <think> tags and malformed output."""
    # Remove <think>...</think> blocks if present
    text = re.sub(r'<think>[\s\S]*?</think>', '', text).strip()
    
    # 1. Try markdown fences (most reliable)
    fenced = re.search(r'```(?:json)?\s*([\s\S]*?)```', text, re.IGNORECASE)
    if fenced:
        try:
            return json.loads(fenced.group(1).strip())
        except json.JSONDecodeError:
            pass
    
    # 2. Extract anything between [ ] or { } greedily
    # First, try to find a list [] (useful for chunking results)
    arrays = re.findall(r'\[[\s\S]*\]', text)
    if arrays:
        for candidate in sorted(arrays, key=len, reverse=True):
            try:
                # Clean up any potential markdown or trailing text inside the candidate
                cleaned = re.sub(r'^.*?\[', '[', candidate, flags=re.DOTALL)
                cleaned = re.sub(r'\].*?$', ']', cleaned, flags=re.DOTALL)
                return json.loads(cleaned)
            except json.JSONDecodeError:
                continue

    # Then try to find an object {}
    objects = re.findall(r'\{[\s\S]*\}', text)
    if objects:
        for candidate in sorted(objects, key=len, reverse=True):
            try:
                cleaned = re.sub(r'^.*?\{', '{', candidate, flags=re.DOTALL)
                cleaned = re.sub(r'\}.*?$', '}', cleaned, flags=re.DOTALL)
                return json.loads(cleaned)
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


def perform_thematic_chunking(llm, text: str, target_size: int = 4000) -> List[str]:
    """Uses DeepSeek-R1 to identify logical/thematic shifts for semantic chunking."""
    # First, do a rough split into segments of ~8000 chars to avoid overwhelming the context
    paragraphs = re.split(r'\n\s*\n', text)
    segments = []
    curr = ""
    for p in paragraphs:
        if len(curr) + len(p) > 8000 and curr:
            segments.append(curr)
            curr = p
        else:
            curr += "\n\n" + p if curr else p
    if curr: segments.append(curr)

    final_chunks = []
    print(f"🧩 Starting semantic chunking on {len(text)} characters...")
    for i, seg in enumerate(segments):
        prompt = f"""Analyze the text below and identify logical 'Thematic Shifts'. 
A thematic shift is where the topic transitions (e.g., from 'Introduction' to 'Mechanism' or 'Examples').

TEXT:
{seg}

TASK:
1. Identify the character index or unique short phrases where the topic changes.
2. Return the text split into semantically coherent chunks based on these shifts.
3. Each chunk should ideally be between 2000-4000 characters.
4. Return ONLY a JSON list of strings representing the chunks.

JSON FORMAT:
[
  "chunk 1 text...",
  "chunk 2 text..."
]
"""
        try:
            response = llm.invoke(prompt).content
            chunks = _extract_json(response)
            if chunks and isinstance(chunks, list):
                print(f"  ✅ Segment {i+1}: Identified {len(chunks)} thematic chunks.")
                for j, c in enumerate(chunks):
                    snippet = str(c)[:50].replace('\n', ' ')
                    print(f"     Sub-chunk {j+1}: {snippet}...")
                final_chunks.extend([str(c) for c in chunks])
            else:
                print(f"  ⚠️ Segment {i+1}: AI failed to return valid JSON chunks. Falling back to length-based split.")
                final_chunks.extend(perform_normal_chunking(seg, target_size))
        except Exception as e:
            print(f"  ❌ Segment {i+1} Error: {e}")
            final_chunks.extend(perform_normal_chunking(seg, target_size))
    
    print(f"🎯 Total semantic chunks created: {len(final_chunks)}")
    return final_chunks


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
    """High-Precision Extraction for Neo4j Knowledge Graphs."""
    prompt = f"""You are an expert knowledge engineer. Extract key concepts and their semantic relationships from the provided text for a biological/educational Knowledge Graph.

TEXT:
{chunk}

TASK:
1. **Identify Entities**: Key terms, definitions, components, or facts. 
   - Use canonical names (singular, consistent casing).
   - Assign a specific 'type' (e.g., 'Organelle', 'Process', 'Component', 'Definition').
   - Include a 'description' or 'properties' for each.
2. **Identify Relationships**: How these entities interact.
   - Use meaningful relation types (e.g., 'PART_OF', 'INHIBITS', 'RESPONSIBLE_FOR', 'LOCATED_IN').
   - Ensure 'source' and 'target' match the entity 'name' exactly.

OUTPUT FORMAT (STRICT JSON):
{{
  "entities": [
    {{"name": "Mitochondria", "type": "Organelle", "description": "Produces ATP through cellular respiration"}}
  ],
  "relationships": [
    {{"source": "Mitochondria", "relation": "PRODUCES", "target": "ATP"}}
  ]
}}
"""
    try:
        response = llm.invoke(prompt).content
        # Log response for debugging if needed (keeping it quiet in prod)
        # print(f"--- EXTRACTION RAW ---\n{response}\n----------------------")
        data = _extract_json(response)
        if data and ("entities" in data or "relationships" in data):
            return data
    except Exception as e:
        print(f"⚠️ Extraction Error: {e}")
    
    return {"entities": [], "relationships": []}


def generate_cypher_queries(extraction: Dict[str, Any], doc_id: str, lesson_name: str) -> List[str]:
    """Deterministic Cypher builder (Python) to avoid LLM hallucination in queries."""
    queries = []
    
    # Process Entities
    for ent in extraction.get("entities", []):
        name = ent.get("name", "").replace("'", "\\'")
        etype = ent.get("type", "Concept").replace(" ", "_")
        
        # New: Support both 'properties' and 'description'
        desc = ent.get("description", "")
        props = ent.get("properties", [])
        if desc and not props: props = [desc]
        props_json = json.dumps(props)
        
        # Create Entity and link to Document
        queries.append(f"MERGE (e:Entity {{id: '{name}'}}) ON CREATE SET e.type='{etype}', e.properties={props_json}")
        queries.append(f"MATCH (e:Entity {{id: '{name}'}}), (d:Document {{id: '{doc_id}'}}) MERGE (e)-[:MENTIONED_IN]->(d)")

    # Process Relationships
    for rel in extraction.get("relationships", []):
        src = rel.get("source", "").replace("'", "\\'")
        tgt = rel.get("target", "").replace("'", "\\'")
        rtype = rel.get("relation", "RELATED_TO").upper().replace(" ", "_").replace("-", "_")
        
        if src and tgt:
            # We want to ensure nodes exist before relating them, though MERGE above handles entities
            # Using MERGE on relationship avoids duplicates
            queries.append(f"""
            MATCH (a:Entity {{id: '{src}'}}), (b:Entity {{id: '{tgt}'}})
            MERGE (a)-[:{rtype}]->(b)
            """)
            
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

    _emit(status_callback, "✂️ Semantic Thematic Chunking…", 0.20)
    # Use thematic chunking with DeepSeek-R1
    chunks = perform_thematic_chunking(llm, full_text)
    
    if not chunks:
        _emit(status_callback, "⚠️ Semantic chunking failed, falling back to normal split.", 0.25)
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
