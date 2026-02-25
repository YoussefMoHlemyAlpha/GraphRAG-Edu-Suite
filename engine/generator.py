"""
generator.py — Phase 3: Retrieval, Reasoning & Quality Control

🧠 Phase 3: Enhanced Generation with Quality Validation
    Step 7: Contextual Retrieval (Enhanced BFS with Bidirectional Relationships)
    Step 8: Prompt Augmentation (Gemma3:4b with Strict Context Rules)
    Step 9: Logic & Reasoning (Gemma3:4b Generation)
    Step 9.5: Critic Loop (Llama 3.2 Validation with Answer Verification)
    Step 9.6: Coverage Validation (Question & Answer Coverage Checks)
    Step 9.7: Quality Filtering (Reject Low-Coverage Questions)
    Step 10: RAG Metrics Calculation (Per-Question Groundedness & Hallucination)
    Step 11: Final Output with Quality Scores
"""

from __future__ import annotations
import json
import re
from typing import Callable, List, Dict, Any, Optional

from engine.graph_store import QuizGraphStore
from engine.rag_metrics import evaluate_rag_response, MetricsStore, tokenize, STOP_WORDS


# ─────────────────────────────────────────────
# Utilities
# ─────────────────────────────────────────────

def _emit(callback: Callable | None, msg: str, pct: float = 0.0):
    if callback:
        callback(msg, pct)
    else:
        print(msg)


def _parse_json(text: str, is_list: bool = True):
    """Robust extraction of the outermost JSON block with aggressive repair."""
    print(f"📝 _parse_json called with {len(text) if text else 0} chars, is_list={is_list}")
    
    if not text:
        return [] if is_list else {}
    
    # Remove common Gemma3 formatting artifacts
    text = re.sub(r'^Here is.*?:\s*', '', text, flags=re.IGNORECASE | re.MULTILINE)
    text = re.sub(r'^---+\s*', '', text, flags=re.MULTILINE)
    text = re.sub(r'\*\*Question:\*\*', '"question":', text)
    text = re.sub(r'\*\*Options:\*\*', '"options":', text)
    
    start_char = '[' if is_list else '{'
    end_char = ']' if is_list else '}'
    
    print(f"Looking for JSON between '{start_char}' and '{end_char}'")
    
    # 1. Try markdown fences first
    fenced = re.search(r'```(?:json)?\s*([\s\S]*?)```', text, re.IGNORECASE)
    if fenced:
        print("Found markdown fence")
        try:
            return json.loads(fenced.group(1).strip())
        except: 
            print("Markdown fence parse failed")
            pass

    # 2. Greedy search for the outermost array/object
    try:
        start_idx = text.find(start_char)
        end_idx = text.rfind(end_char)
        print(f"Found start at {start_idx}, end at {end_idx}")
        
        if start_idx != -1 and end_idx != -1 and end_idx > start_idx:
            candidate = text[start_idx:end_idx+1]
            print(f"Extracted candidate: {len(candidate)} chars")
            print(f"First 200 chars: {candidate[:200]}")
            
            # Try to parse as-is first
            try:
                parsed = json.loads(candidate)
                print(f"✅ Successfully parsed JSON: {len(parsed) if isinstance(parsed, list) else 1} items")
                return parsed
            except json.JSONDecodeError as e:
                # Try to repair common issues
                print(f"⚠️ JSON parse failed: {e}, attempting repair...")
                repaired = candidate
                
                # Fix 1: Missing opening braces: ["question" -> [{"question"
                if repaired.startswith('["'):
                    repaired = '[{' + repaired[1:]
                    print("  - Added opening brace")
                
                # Fix 2: Missing commas and braces between objects: "]["question" -> },{"question"
                repaired = repaired.replace(']["', '},{"')
                repaired = repaired.replace('"}{"', '"},{"')
                print("  - Fixed object separators")
                
                # Fix 3: Missing closing brace before final bracket
                if repaired.endswith(']') and not repaired.endswith('}]'):
                    repaired = repaired[:-1] + '}]'
                    print("  - Added closing brace")
                
                print(f"Repaired first 200 chars: {repaired[:200]}")
                
                try:
                    parsed = json.loads(repaired)
                    print(f"✅ Successfully parsed repaired JSON: {len(parsed) if isinstance(parsed, list) else 1} items")
                    return parsed
                except json.JSONDecodeError as e:
                    print(f"⚠️ JSON repair failed: {e}")
                    print(f"Repaired version: {repaired[:300]}...")
                    
    except Exception as e:
        print(f"⚠️ Unexpected error parsing JSON: {e}")
        import traceback
        traceback.print_exc()
        
    print("❌ All parsing attempts failed, returning empty")
    return [] if is_list else {}


# ─────────────────────────────────────────────
# Step 7 — Contextual Retrieval (BFS)
# ─────────────────────────────────────────────

def _retrieve_graph_context(store: QuizGraphStore, lesson_name: str) -> tuple[str, list[str]]:
    """Enhanced context retrieval with comprehensive detail and structured formatting."""
    
    # Query: Get all entities with their properties and bidirectional relationships
    query = """
    MATCH (les:Lesson {name: $lesson})<-[:PART_OF]-(e:Entity)
    OPTIONAL MATCH (e)-[r]->(related:Entity)
    OPTIONAL MATCH (e)<-[r_in]-(incoming:Entity)
    RETURN 
        e.id AS entity, 
        e.type AS type, 
        e.properties AS props,
        collect(DISTINCT {
            direction: 'outgoing',
            relation: type(r),
            target: related.id,
            target_type: related.type,
            target_props: related.properties
        }) AS outgoing_rels,
        collect(DISTINCT {
            direction: 'incoming',
            relation: type(r_in),
            source: incoming.id,
            source_type: incoming.type,
            source_props: incoming.properties
        }) AS incoming_rels
    LIMIT 1500
    """
    rows = store.query(query, {"lesson": lesson_name})

    if not rows:
        return "", []

    facts: list[str] = []
    entity_degree: dict[str, int] = {}
    entity_details: dict[str, dict] = {}

    # First pass: collect comprehensive entity details
    for row in rows:
        ent = row.get("entity") or ""
        etype = row.get("type") or ""
        props = row.get("props") or ""
        
        if not ent:
            continue
            
        # Store entity details
        if ent not in entity_details:
            entity_details[ent] = {
                'type': etype,
                'properties': props,
                'outgoing': [],
                'incoming': []
            }
        
        # Track outgoing relationships
        for rel_info in row.get("outgoing_rels", []):
            if rel_info.get('relation') and rel_info.get('target'):
                entity_degree[ent] = entity_degree.get(ent, 0) + 1
                entity_details[ent]['outgoing'].append(rel_info)
        
        # Track incoming relationships
        for rel_info in row.get("incoming_rels", []):
            if rel_info.get('relation') and rel_info.get('source'):
                entity_degree[ent] = entity_degree.get(ent, 0) + 1
                entity_details[ent]['incoming'].append(rel_info)

    # Second pass: build structured, comprehensive context
    for ent, details in entity_details.items():
        etype = details['type']
        props = details['properties']
        
        # Add entity definition with full details
        if props:
            if isinstance(props, list):
                props_str = "; ".join(str(p) for p in props if p)
            else:
                props_str = str(props)
            facts.append(f"ENTITY: {ent} (Type: {etype})")
            facts.append(f"  Definition: {props_str}")
        else:
            facts.append(f"ENTITY: {ent} (Type: {etype})")
        
        # Add outgoing relationships with full context
        for rel_info in details['outgoing']:
            rel = rel_info.get('relation')
            target = rel_info.get('target')
            target_type = rel_info.get('target_type')
            target_props = rel_info.get('target_props')
            
            if rel and target:
                # Rich relationship format with target definition
                if target_props:
                    if isinstance(target_props, list):
                        target_desc = "; ".join(str(p) for p in target_props if p)
                    else:
                        target_desc = str(target_props)
                    facts.append(f"  {ent} --[{rel}]--> {target} (Type: {target_type}, Definition: {target_desc})")
                else:
                    facts.append(f"  {ent} --[{rel}]--> {target} (Type: {target_type})")
        
        # Add incoming relationships for complete picture
        for rel_info in details['incoming']:
            rel = rel_info.get('relation')
            source = rel_info.get('source')
            source_type = rel_info.get('source_type')
            
            if rel and source:
                facts.append(f"  {source} (Type: {source_type}) --[{rel}]--> {ent}")
        
        # Add separator for readability
        facts.append("")

    sorted_entities = sorted(entity_degree.items(), key=lambda x: x[1], reverse=True)
    high_value = [e for e, _ in sorted_entities[:15]]

    unique_facts = list(dict.fromkeys(facts))
    context_text = "\n".join(unique_facts)
    
    print(f"📊 Retrieved {len(unique_facts)} structured facts from knowledge graph")
    print(f"📊 Top {len(high_value)} high-value entities: {', '.join(high_value[:5])}")
    print(f"📊 Context size: {len(context_text)} characters")
    
    return context_text, high_value


# ─────────────────────────────────────────────
# Pipelines
# ─────────────────────────────────────────────

def generate_graph_quiz(
    llm,                 # Primary LLM (Llama 3.2)
    lesson_name: str,
    n: int,
    status_callback: Callable | None = None,
    enable_metrics: bool = False,  # NEW: Enable metrics calculation
    **kwargs             # Optional critic_llm
) -> list[dict]:
    store = QuizGraphStore()
    critic_llm = kwargs.get("critic_llm") or llm
    from engine.vram_util import stop_ollama_model

    _emit(status_callback, "🔍 Step 7 — Retrieving graph knowledge…", 0.10)
    context, high_value = _retrieve_graph_context(store, lesson_name)
    print(f"--- RETRIEVED CONTEXT ---\n{context}\n------------------------")

    if not context:
        no_data_quiz = [{"question": "No data found.", "options": ["N/A"], "correct_index": 0}]
        if enable_metrics:
            return no_data_quiz, None
        return no_data_quiz

    # ── Phase 1: Generation (Gemma3) ───────────────────────────────
    _emit(status_callback, "📟 Optimizing VRAM for Gemma3...", 0.20)
    stop_ollama_model("llama3.2:latest") # Free VRAM for Gemma3
    _emit(status_callback, "🧠 Phase 1: Generating Bloom-Balanced Quiz…", 0.35)
    
    bloom_distribution = ""
    if n >= 6:
        bloom_distribution = "MANDATORY: Generate exactly ONE question for each of the 6 levels: Remember, Understand, Apply, Analyze, Evaluate, Create."
    elif n > 1:
        bloom_distribution = f"Generate {n} questions with DIFFERENT Bloom's levels. Do NOT repeat levels."
    
    gen_prompt = f"""You MUST generate EXACTLY {n} multiple choice questions based STRICTLY on the context below.
{bloom_distribution}

CONTEXT (YOUR ONLY SOURCE OF TRUTH):
{context}

CRITICAL RULES:
1. Generate EXACTLY {n} questions - no more, no less
2. ONLY use information that appears in the context above
3. Every question must be answerable using ONLY the context
4. Every option must be verifiable or falsifiable from the context
5. The correct answer MUST be explicitly stated in the context
6. Do NOT add external knowledge or assumptions

QUESTION REQUIREMENTS:
1. Return ONLY valid JSON - no explanations, no markdown, no extra text
2. Start your response with [ and end with ]
3. Each question must have EXACTLY 4 options
4. Only one option must be correct (correct_index: 0, 1, 2, or 3)
5. Use these Bloom's levels: Remember, Understand, Apply, Analyze, Evaluate, Create
6. Include "source_fact" field showing which context fact supports the answer

REQUIRED JSON FORMAT - YOU MUST GENERATE {n} QUESTIONS:
[
  {{
    "question": "Your question text here?",
    "options": ["Option A", "Option B", "Option C", "Option D"],
    "correct_index": 0,
    "bloom_level": "Remember",
    "source_fact": "Exact fact from context that supports this answer"
  }},
  ... (continue until you have {n} questions total)
]

IMPORTANT: Count your questions before submitting. You MUST have EXACTLY {n} questions in the array.

VALIDATION BEFORE SUBMITTING:
- Do I have EXACTLY {n} questions? If NO, add more
- Can each question be answered from the context? If NO, rewrite it
- Is the correct_index pointing to the right answer? Double-check
- Are all terms in the question defined in the context? If NO, rewrite

DO NOT include any text before or after the JSON array.
DO NOT use markdown formatting like ``` or ---.
START your response with [ and END with ].
"""
    try:
        print("🎯 Invoking LLM for generation...")
        raw_gen = llm.invoke(gen_prompt).content
        print(f"--- RAW GENERATION ({len(raw_gen)} chars) ---")
        print(raw_gen)
        print("------------------------")
        
        if not raw_gen or len(raw_gen) < 10:
            print("⚠️ LLM returned empty or very short response")
            if enable_metrics:
                return [], None
            return []
        
        print("🔍 Attempting to parse JSON...")
        initial_quiz = _parse_json(raw_gen)
        print(f"--- PARSED QUIZ ({len(initial_quiz) if initial_quiz else 0} questions) ---")
        
        if initial_quiz:
            print(f"✅ Got {len(initial_quiz)} questions (requested: {n})")
            if len(initial_quiz) < n:
                print(f"⚠️ WARNING: Only generated {len(initial_quiz)}/{n} questions!")
                print(f"   This may indicate context is too limited or LLM is being too strict")
            for i, q in enumerate(initial_quiz[:2]):  # Show first 2
                print(f"  Q{i+1}: {q.get('question', 'NO QUESTION')[:50]}...")
        
        if not initial_quiz:
            print("⚠️ No questions parsed from generation")
            print(f"First 500 chars: {raw_gen[:500]}")
            
            # Try manual extraction as last resort
            print("🔧 Attempting manual extraction...")
            import re
            # Find all question objects
            pattern = r'"question":\s*"([^"]+)"'
            questions_found = re.findall(pattern, raw_gen)
            print(f"Found {len(questions_found)} question strings in raw text")
            
            if enable_metrics:
                return [], None
            return []
    except Exception as e:
        print(f"⚠️ Generation failed with exception: {e}")
        import traceback
        traceback.print_exc()
        if enable_metrics:
            return [], None
        return []

    # ── Phase 2: Critic Review (Llama 3.2) ──────────────────────────
    _emit(status_callback, "⚖️ Phase 2: Review with Llama 3.2…", 0.60)
    
    # SWAP MODELS
    stop_ollama_model("gemma3:4b") # Free VRAM for Llama 3.2
    
    critic_prompt = f"""You are a STRICT fact-checker. Review and correct this quiz with EXTREME RIGOR.
DO NOT remove questions unless absolutely necessary - FIX them instead.

CONTEXT (GROUND TRUTH - ALL ANSWERS MUST BE VERIFIABLE HERE):
{context}

QUIZ TO VALIDATE ({len(initial_quiz)} questions):
{json.dumps(initial_quiz, indent=2)}

MANDATORY VALIDATION CHECKLIST (Check EVERY question):

1. FACTUAL CORRECTNESS:
   - Read the question carefully
   - Find the EXACT answer in the context above
   - Verify the correct_index (0-3) points to the RIGHT answer
   - If the correct answer is NOT at correct_index, FIX IT
   - If NO correct answer exists in options, REWRITE the options

2. CONTEXT GROUNDING:
   - Every term in the question MUST appear in the context
   - If a term is missing, REWRITE the question using context terms
   - Every option MUST be verifiable or falsifiable from context
   - ONLY remove questions if they cannot be rewritten to use context

3. OPTION QUALITY:
   - All 4 options must be distinct and plausible
   - Only ONE option can be correct
   - Wrong options should be related but clearly incorrect
   - No duplicate or near-duplicate options
   - FIX bad options, don't remove the question

4. ANSWER VERIFICATION (CRITICAL):
   - For EACH question, manually verify:
     * What is the question asking?
     * What does the context say about this?
     * Which option matches the context?
     * Does correct_index point to that option?
   - If correct_index is WRONG, change it to the right value

5. BLOOM'S LEVEL:
   - Ensure variety across cognitive levels
   - Match question complexity to stated Bloom level
   - Adjust level if needed, don't remove question

EXAMPLE VALIDATION:
Question: "What does X do?"
Context says: "X performs function Y"
Options: [A: "Y", B: "Z", C: "W", D: "Q"]
Correct answer: A (index 0)
Check: Does correct_index = 0? If not, FIX IT.

IMPORTANT: Try to keep all {len(initial_quiz)} questions by FIXING issues rather than removing.
Only remove a question if it's completely unverifiable and cannot be rewritten.

OUTPUT REQUIREMENTS:
- Return ONLY valid JSON array
- Keep as many questions as possible (target: {len(initial_quiz)} questions)
- Fix incorrect correct_index values
- Rewrite bad options
- Only remove truly unverifiable questions

Return the corrected quiz as a JSON array with {len(initial_quiz)} questions (or close to it):
"""
    try:
        raw_refined = critic_llm.invoke(critic_prompt).content
        print(f"--- CRITIC RAW RESPONSE ---\n{raw_refined[:500]}...\n------------------------")
        refined_quiz = _parse_json(raw_refined)
        
        print(f"📊 Critic results: {len(initial_quiz)} questions → {len(refined_quiz) if refined_quiz else 0} questions")
        if refined_quiz and len(refined_quiz) < len(initial_quiz):
            removed = len(initial_quiz) - len(refined_quiz)
            print(f"⚠️ Critic removed {removed} questions")
        
        # Final validation and cleanup with STRICT correct_index verification
        final_quiz = []
        for i, q in enumerate(refined_quiz or initial_quiz):
            if not isinstance(q, dict) or "options" not in q or len(q["options"]) != 4:
                print(f"⚠️ Q{i+1}: Skipping - invalid format")
                continue
            
            # Force index sanity
            idx = q.get("correct_index", 0)
            if not isinstance(idx, int) or idx < 0 or idx >= 4:
                print(f"⚠️ Q{i+1}: Invalid correct_index {idx}, defaulting to 0")
                q["correct_index"] = 0
            
            # Additional validation: Check if question and answer are in context
            question_text = q.get("question", "").lower()
            correct_answer = q["options"][q["correct_index"]].lower()
            context_lower = context.lower()
            
            # Extract key terms from question (remove common words)
            question_terms = set(tokenize(question_text)) - STOP_WORDS
            answer_terms = set(tokenize(correct_answer)) - STOP_WORDS
            context_terms = set(tokenize(context_lower)) - STOP_WORDS
            
            # Check if key terms exist in context
            question_coverage = len(question_terms & context_terms) / len(question_terms) if question_terms else 0
            answer_coverage = len(answer_terms & context_terms) / len(answer_terms) if answer_terms else 0
            
            # STRICT FILTERING: Skip questions with very low coverage
            if question_coverage < 0.5:
                print(f"⚠️ Q{i+1}: REJECTED - Question coverage too low ({question_coverage:.1%})")
                print(f"   Question: {question_text[:80]}")
                print(f"   Missing terms: {question_terms - context_terms}")
                continue
            
            if answer_coverage < 0.4:
                print(f"⚠️ Q{i+1}: REJECTED - Answer coverage too low ({answer_coverage:.1%})")
                print(f"   Answer: {correct_answer[:80]}")
                print(f"   Missing terms: {answer_terms - context_terms}")
                continue
            
            # VERIFY CORRECT_INDEX: Check if the marked answer is actually correct
            # by comparing all options' coverage scores
            all_options_coverage = []
            for opt_idx, option in enumerate(q["options"]):
                opt_terms = set(tokenize(option.lower())) - STOP_WORDS
                opt_coverage = len(opt_terms & context_terms) / len(opt_terms) if opt_terms else 0
                all_options_coverage.append((opt_idx, opt_coverage, option))
            
            # Sort by coverage (highest first)
            sorted_options = sorted(all_options_coverage, key=lambda x: x[1], reverse=True)
            best_option_idx = sorted_options[0][0]
            best_coverage = sorted_options[0][1]
            
            # If the current correct_index has much lower coverage than the best option, warn
            current_coverage = all_options_coverage[q["correct_index"]][1]
            if best_option_idx != q["correct_index"] and best_coverage > current_coverage + 0.2:
                print(f"⚠️ Q{i+1}: SUSPICIOUS correct_index!")
                print(f"   Current answer (index {q['correct_index']}): {q['options'][q['correct_index']]} (coverage: {current_coverage:.1%})")
                print(f"   Best option (index {best_option_idx}): {q['options'][best_option_idx]} (coverage: {best_coverage:.1%})")
                print(f"   Consider: The answer may be wrong!")
            
            # Warnings for moderate coverage
            if question_coverage < 0.7:
                print(f"⚠️ Q{i+1}: Low question coverage ({question_coverage:.1%}) - question may not be fully grounded in context")
            
            if answer_coverage < 0.6:
                print(f"⚠️ Q{i+1}: Low answer coverage ({answer_coverage:.1%}) - answer may be partially hallucinated")
            
            # Add coverage metrics to question for debugging
            q['_validation'] = {
                'question_coverage': question_coverage,
                'answer_coverage': answer_coverage
            }
            
            final_quiz.append(q)
        
        print(f"--- FINAL QUIZ: {len(final_quiz)} questions after validation (started with {len(refined_quiz or initial_quiz)}) ---")
        
        # If we filtered out too many questions, warn the user
        if len(final_quiz) < n * 0.7:  # Less than 70% of requested
            print(f"⚠️ WARNING: Only {len(final_quiz)}/{n} questions passed validation!")
            print(f"   This indicates context quality issues or overly strict filtering")
            print(f"   Consider: 1) Rebuilding knowledge graph, 2) Using a different lesson, 3) Relaxing coverage thresholds")
        
        # ── Calculate Metrics (if enabled) ────────────────────────────
        if enable_metrics:
            _emit(status_callback, "📊 Calculating RAG metrics…", 0.90)
            metrics_store = MetricsStore()
            
            # Calculate metrics for each question
            for i, q in enumerate(final_quiz):
                question_text = q.get("question", "")
                
                # Evaluate metrics for this question
                metrics = evaluate_rag_response(
                    generated=question_text,
                    retrieved_context=context,
                    references=None,
                    metadata={
                        'lesson_name': lesson_name,
                        'question_type': 'mcq',
                        'bloom_level': q.get('bloom_level', 'Unknown'),
                        'question_index': i,
                        'question_coverage': q.get('_validation', {}).get('question_coverage', 0),
                        'answer_coverage': q.get('_validation', {}).get('answer_coverage', 0)
                    }
                )
                
                # Store metrics
                metrics_store.add_metrics(metrics)
                
                # Attach metrics to question
                q['metrics'] = {
                    'groundedness': metrics.groundedness,
                    'hallucination_rate': metrics.hallucination_rate
                }
                
                print(f"✅ Q{i+1} Metrics - Groundedness: {metrics.groundedness:.2%}, Hallucination: {metrics.hallucination_rate:.2%}, Q-Coverage: {q['_validation']['question_coverage']:.2%}, A-Coverage: {q['_validation']['answer_coverage']:.2%}")
        
        _emit(status_callback, "🎉 Step 10 — Ready!", 1.0)
        return final_quiz, None if not enable_metrics else True  # Return flag instead of metrics object
    except Exception as e:
        print(f"⚠️ Critic failed: {e}")
        # Return initial quiz with proper format
        if enable_metrics:
            return initial_quiz[:n], None
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


def generate_essay_questions(
    llm,                 # Primary (Llama 3.2)
    lesson_name: str, 
    n: int, 
    status_callback: Callable | None = None,
    enable_metrics: bool = False,  # NEW: Enable metrics calculation
    **kwargs
) -> list[dict]:
    store = QuizGraphStore()
    critic_llm = kwargs.get("critic_llm") or llm
    from engine.vram_util import stop_ollama_model

    _emit(status_callback, "🔍 Retrieving graph context…", 0.10)
    context, _ = _retrieve_graph_context(store, lesson_name)
    if not context:
        if enable_metrics:
            return [], None
        return []

    # ── Phase 1: Generation (Qwen3) ───────────────────────────────
    _emit(status_callback, "📟 Optimizing VRAM for Qwen3...", 0.25)
    stop_ollama_model("llama3.2:latest")
    _emit(status_callback, "✍️ Drafting high-reasoning prompts…", 0.45)
    
    gen_prompt = f"""Generate EXACTLY {n} complex essay questions based STRICTLY on the context below.

CONTEXT (YOUR ONLY SOURCE OF TRUTH):
{context}

CRITICAL RULES:
1. Generate EXACTLY {n} essay questions - no more, no less
2. ONLY use information that appears in the context above
3. Every question must be answerable using ONLY the context
4. Expected concepts MUST be verifiable from the context
5. Do NOT add external knowledge or assumptions

BLOOM'S TAXONOMY LEVELS:
If multiple questions: Cover Remember, Understand, Apply, Analyze, Evaluate, Create

REQUIRED JSON FORMAT - YOU MUST GENERATE {n} QUESTIONS:
[
  {{
    "question": "Your essay question here?",
    "difficulty": "Easy/Med/Hard",
    "bloom_level": "One of the 6 Bloom levels",
    "expected_concepts": ["concept1", "concept2", "concept3"]
  }},
  ... (continue until you have {n} questions total)
]

VALIDATION BEFORE SUBMITTING:
- Do I have EXACTLY {n} questions? If NO, add more
- Can each question be answered from the context? If NO, rewrite it
- Are all expected concepts defined in the context? If NO, remove or replace them
- Are all terms in the question defined in the context? If NO, rewrite

Return ONLY valid JSON - no explanations, no markdown, no extra text.
START your response with [ and END with ].
"""
    try:
        raw_gen = llm.invoke(gen_prompt).content
        initial_essays = _parse_json(raw_gen)
        if not initial_essays:
            if enable_metrics:
                return [], None
            return []
    except:
        if enable_metrics:
            return [], None
        return []

    # ── Phase 2: Critique (Llama 3.2) ──────────────────────────
    _emit(status_callback, "⚖️ Review with Llama 3.2…", 0.70)
    stop_ollama_model("gemma3:4b")
    
    critic_prompt = f"""You are a STRICT fact-checker. Review and correct these essay questions with EXTREME RIGOR.
DO NOT remove questions unless absolutely necessary - FIX them instead.

CONTEXT (GROUND TRUTH - ALL QUESTIONS MUST BE VERIFIABLE HERE):
{context}

ESSAY QUESTIONS TO VALIDATE ({len(initial_essays)} questions):
{json.dumps(initial_essays, indent=2)}

MANDATORY VALIDATION CHECKLIST (Check EVERY question):

1. CONTEXT GROUNDING:
   - Every term in the question MUST appear in the context
   - If a term is missing, REWRITE the question using context terms
   - ONLY remove questions if they cannot be rewritten to use context

2. EXPECTED CONCEPTS VALIDATION:
   - Each expected concept MUST be defined in the context
   - Remove concepts that are not in the context
   - Add relevant concepts that ARE in the context
   - Ensure concepts are specific and verifiable

3. BLOOM'S LEVEL ACCURACY:
   - Verify the bloom_level matches the question complexity
   - Remember: Recall facts
   - Understand: Explain concepts
   - Apply: Use knowledge in new situations
   - Analyze: Break down and examine
   - Evaluate: Judge and critique
   - Create: Produce original work

4. DIFFICULTY ALIGNMENT:
   - Easy: Straightforward, clear answer from context
   - Med: Requires synthesis of multiple concepts
   - Hard: Requires deep analysis and evaluation

5. ANSWERABILITY:
   - Can this question be answered using ONLY the context?
   - Are there enough details in the context to write a good essay?
   - If NO, rewrite to match available information

IMPORTANT: Try to keep all {len(initial_essays)} questions by FIXING issues rather than removing.
Only remove a question if it's completely unverifiable and cannot be rewritten.

OUTPUT REQUIREMENTS:
- Return ONLY valid JSON array
- Keep as many questions as possible (target: {len(initial_essays)} questions)
- Fix questions to use only context terms
- Validate and correct expected concepts
- Only remove truly unverifiable questions

Return the corrected essay questions as a JSON array with {len(initial_essays)} questions (or close to it):
"""
    try:
        raw_refined = critic_llm.invoke(critic_prompt).content
        refined_essays = _parse_json(raw_refined) or initial_essays
        
        print(f"📊 Critic results: {len(initial_essays)} essays → {len(refined_essays)} essays")
        
        # ── Validate and Filter Essays (same as MCQs) ────────────────
        final_essays = []
        context_lower = context.lower()
        context_terms = set(tokenize(context_lower)) - STOP_WORDS
        
        for i, essay in enumerate(refined_essays):
            if not isinstance(essay, dict) or "question" not in essay:
                print(f"⚠️ Essay {i+1}: Skipping - invalid format")
                continue
            
            question_text = essay.get("question", "").lower()
            question_terms = set(tokenize(question_text)) - STOP_WORDS
            
            # Calculate question coverage
            question_coverage = len(question_terms & context_terms) / len(question_terms) if question_terms else 0
            
            # STRICT FILTERING: Reject essays with low coverage
            if question_coverage < 0.5:
                print(f"⚠️ Essay {i+1}: REJECTED - Question coverage too low ({question_coverage:.1%})")
                print(f"   Question: {essay.get('question', '')[:80]}")
                print(f"   Missing terms: {question_terms - context_terms}")
                continue
            
            # Validate expected concepts
            expected_concepts = essay.get("expected_concepts", [])
            low_coverage_concepts = []
            
            for concept in expected_concepts:
                concept_terms = set(tokenize(concept.lower())) - STOP_WORDS
                concept_coverage = len(concept_terms & context_terms) / len(concept_terms) if concept_terms else 0
                
                if concept_coverage < 0.4:
                    low_coverage_concepts.append((concept, concept_coverage))
            
            # Warn about low-coverage concepts
            if low_coverage_concepts:
                print(f"⚠️ Essay {i+1}: Some expected concepts have low coverage:")
                for concept, coverage in low_coverage_concepts:
                    print(f"   - '{concept}': {coverage:.1%}")
            
            # Add validation metadata
            essay['_validation'] = {
                'question_coverage': question_coverage,
                'low_coverage_concepts': [c[0] for c in low_coverage_concepts]
            }
            
            # Warnings for moderate coverage
            if question_coverage < 0.7:
                print(f"⚠️ Essay {i+1}: Moderate question coverage ({question_coverage:.1%})")
            
            final_essays.append(essay)
        
        print(f"--- FINAL ESSAYS: {len(final_essays)} essays after validation (started with {len(refined_essays)}) ---")
        
        # Warn if too many filtered
        if len(final_essays) < n * 0.7:
            print(f"⚠️ WARNING: Only {len(final_essays)}/{n} essays passed validation!")
            print(f"   This indicates context quality issues or overly strict filtering")
        
        # ── Calculate Per-Essay Metrics (if enabled) ────────────────────
        if enable_metrics:
            _emit(status_callback, "📊 Calculating RAG metrics…", 0.90)
            metrics_store = MetricsStore()
            
            # Calculate metrics for EACH essay question (like MCQs)
            for i, essay in enumerate(final_essays):
                question_text = essay.get("question", "")
                
                # Evaluate metrics for this essay
                metrics = evaluate_rag_response(
                    generated=question_text,
                    retrieved_context=context,
                    references=None,
                    metadata={
                        'lesson_name': lesson_name,
                        'question_type': 'essay',
                        'bloom_level': essay.get('bloom_level', 'Unknown'),
                        'difficulty': essay.get('difficulty', 'Unknown'),
                        'question_index': i,
                        'question_coverage': essay.get('_validation', {}).get('question_coverage', 0)
                    }
                )
                
                # Store metrics
                metrics_store.add_metrics(metrics)
                
                # Attach metrics to essay
                essay['metrics'] = {
                    'groundedness': metrics.groundedness,
                    'hallucination_rate': metrics.hallucination_rate
                }
                
                print(f"✅ Essay {i+1} Metrics - Groundedness: {metrics.groundedness:.2%}, "
                      f"Hallucination: {metrics.hallucination_rate:.2%}, "
                      f"Q-Coverage: {essay['_validation']['question_coverage']:.2%}")
            
            return final_essays, True  # Return flag instead of metrics object
        
        return final_essays, None
    except Exception as e:
        print(f"⚠️ Essay generation failed: {e}")
        import traceback
        traceback.print_exc()
        if enable_metrics:
            return initial_essays, None
        return initial_essays


def evaluate_essay_response(
    llm, 
    question_obj: dict, 
    student_text: str,
    enable_metrics: bool = False  # NEW: Enable metrics calculation
) -> dict:
    # Essay evaluation uses the provided llm (Gemma3)
    from engine.vram_util import stop_ollama_model
    stop_ollama_model("llama3.2:latest") # Ensure Gemma3 has VRAM
    
    prompt = f"""GRADE the student's essay answer.
QUESTION: {question_obj['question']}
STUDENT ANSWER: {student_text}

Provide a score (0-10) and constructive feedback based on factual accuracy.
Return ONLY JSON: {{"score": 8, "feedback": "..."}}
"""
    try:
        raw = llm.invoke(prompt).content
        result = _parse_json(raw, is_list=False) or {"score": 0, "feedback": "Grading failed."}
        
        # ── Calculate Metrics (if enabled) ────────────────────────────
        if enable_metrics and student_text:
            # Get the context from the question metadata if available
            context = question_obj.get('source_context', '')
            
            if context:
                metrics = evaluate_rag_response(
                    generated=student_text,
                    retrieved_context=context,
                    references=None,
                    metadata={
                        'question': question_obj.get('question', ''),
                        'response_type': 'student_essay',
                        'bloom_level': question_obj.get('bloom_level', 'Unknown')
                    }
                )
                
                # Attach metrics to result
                result['metrics'] = {
                    'groundedness': metrics.groundedness,
                    'hallucination_rate': metrics.hallucination_rate
                }
                
                print(f"Student Response Metrics - Groundedness: {metrics.groundedness:.2f}, "
                      f"Hallucination: {metrics.hallucination_rate:.2f}")
                
                # Store metrics
                metrics_store = MetricsStore()
                metrics_store.add_metrics(metrics)
        
        return result
    except:
        return {"score": 0, "feedback": "Evaluation error."}
