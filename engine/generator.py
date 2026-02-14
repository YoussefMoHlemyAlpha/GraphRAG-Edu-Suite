import json
from langchain_core.prompts import PromptTemplate
from engine.graph_store import QuizGraphStore
import re
def generate_graph_quiz(llm, lesson_name, n):
    store = QuizGraphStore()
    
    # Improved query: Returns human-readable facts
    context_query = """
    MATCH (l:Lesson {name: $lesson})<-[:PART_OF]-(n)
    OPTIONAL MATCH (n)-[r]->(m)
    RETURN 
        CASE 
            WHEN r IS NULL THEN "Topic: " + n.id
            ELSE "Fact: " + n.id + " " + toLower(replace(type(r), '_', ' ')) + " " + m.id 
        END AS fact
    LIMIT 100
    """
    results = store.query(context_query, {"lesson": lesson_name})
    
    # If the graph is empty, we must stop here or it will hallucinate!
    if not results:
        print(f"⚠️ No nodes found for lesson: {lesson_name}")
        return [{"question": "Error: Knowledge Graph is empty for this lesson. Please rebuild the graph.", "options": ["N/A"], "correct_index": 0, "bloom_level": "N/A"}]

    context = "\n".join([r['fact'] for r in results if r['fact']])
    print(f"📊 Fact context size: {len(context)} characters")

    # Step 1: Strict Generation
    gen_prompt = f"""
    ### ROLE: Educational Content Creator
    ### DATA SOURCE (STRICTLY USE ONLY THIS):
    {context}

    ### MANDATORY INSTRUCTIONS:
    1. ZERO EXTERNAL KNOWLEDGE: Do NOT use any information from your training data. 
    2. NO GENERAL KNOWLEDGE: Do NOT ask about Paris, France, Einstein, Mountains, or Planets unless they are EXPLICITLY mentioned in the DATA SOURCE above.
    3. TOPIC FOCUS: If the DATA SOURCE is about "AI" or "Reasoning", every single question must be about "AI" or "Reasoning".
    4. MCQ STRUCTURE: 
       - Exactly 4 options (array of strings).
       - 1 Correct Answer (verifiable from DATA).
       - 3 Distractors (plausible but wrong).
    5. QUANTITY: Generate exactly {n} questions.
    6. DIVERSITY: Cover Bloom's levels: Remember, Understand, Apply, Analyze, Evaluate, Create.
    7. FORMAT: Return only the JSON array.

    ### JSON FORMAT:
    [
      {{
        "question": "Question about a fact from the data?",
        "options": ["Correct string", "Incorrect 1", "Incorrect 2", "Incorrect 3"],
        "correct_index": 0,
        "bloom_level": "Understand"
      }}
    ]
    """
    
    raw_quiz = llm.invoke(gen_prompt)

    # Step 2: Verification
    critique_prompt = f"""
    ### ROLE: Accuracy Auditor
    ### TASK: Scan the generated quiz and remove/fix any errors.

    --- ORIGINAL QUIZ ---
    {raw_quiz}

    --- REFERENCE DATA ---
    {context}

    --- AUDIT CHECKLIST ---
    1. NO HALLUCINATIONS: If a question is about the "Auditor", "Designer", or "Instructions", DELETE it and replace it with a content-based question.
    2. STRUCTURE: Ensure exactly {n} questions exist.
    3. FORMAT: Options MUST be plain strings. 
    4. ACCURACY: The answer must match the REFERENCE DATA.

    ### OUTPUT: Return ONLY the final JSON array.
    """
    verified_quiz = llm.invoke(critique_prompt)
    
    try:
        data = json.loads(verified_quiz)
        return data if data else [{"question": "Error: LLM was unable to generate questions from this context. Try a different lesson.", "options": ["N/A"], "correct_index": 0, "bloom_level": "N/A"}]
    except:
        # Fallback: find the JSON array if LLM adds chatter
        import re
        match = re.search(r'\[.*\]', verified_quiz, re.DOTALL)
        if match:
            try:
                data = json.loads(match.group())
                return data if data else [{"question": "Error: LLM generated an empty quiz.", "options": ["N/A"], "correct_index": 0, "bloom_level": "N/A"}]
            except:
                pass
        
        print(f"❌ JSON Parsing Failed. Raw Response: {verified_quiz}")
        return [{"question": "Error: Failed to parse quiz data. Please try again.", "options": ["N/A"], "correct_index": 0, "bloom_level": "N/A"}]





def generate_essay_questions(llm, lesson_name, n):
    store = QuizGraphStore()
    
    # Retrieve structural relationships (Context)
    res = store.query("""
        MATCH (l:Lesson {name: $lesson})<-[:PART_OF]-(n)-[r]->(m)
        RETURN n.id + ' ' + type(r) + ' ' + m.id AS fact LIMIT 50
    """, {"lesson": lesson_name})
    
    context = "\n".join([r['fact'] for r in res if r['fact'] is not None])
    
    if not context:
        return [{
            "question": "Error: Knowledge Graph is empty. Please upload PDFs and build the graph first.",
            "bloom_level": "N/A",
            "key_concepts": [],
            "rubric": []
        }]

    prompt = f"""
    CONTEXT FROM GRAPH:
    {context}

    TASK: Generate {n} Essay Questions for a 'Supervised ML' course.
    BLOOM LEVELS: Focus on 'Analyze', 'Evaluate', or 'Create'.
    
    REQUIREMENT: Each question must force the student to connect multiple facts from the context.
    
    RETURN ONLY A JSON ARRAY:
    [
      {{
        "question": "The essay prompt...",
        "bloom_level": "Analyze/Evaluate",
        "key_concepts": ["concept1", "concept2"],
        "rubric": ["Point 1 for grading", "Point 2 for grading"]
      }}
    ]
    """
    raw_output = llm.invoke(prompt)
    return parse_json_safely(raw_output)

def evaluate_essay_response(llm, question_obj, student_text):
    eval_prompt = f"""
    QUESTION: {question_obj['question']}
    RUBRIC: {question_obj['rubric']}
    EXPECTED CONCEPTS: {question_obj['key_concepts']}
    
    STUDENT ANSWER: 
    {student_text}
    
    TASK: Grade the essay based on the Rubric. Check if they correctly used the concepts.
    RETURN ONLY JSON:
    {{
      "score": out of 10,
      "feedback": "constructive advice",
      "missed_entities": ["concept1", "concept2"]
    }}
    """
    raw_eval = llm.invoke(eval_prompt)
    return parse_json_safely(raw_eval, is_list=False)

def parse_json_safely(input_text, is_list=True):
    """The 'Regex' Logic to prevent crashes from LLM chatter"""
    pattern = r'\[.*\]' if is_list else r'\{.*\}'
    match = re.search(pattern, input_text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group())
        except:
            return [] if is_list else {}
    return [] if is_list else {}
