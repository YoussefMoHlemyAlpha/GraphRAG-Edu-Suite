import json
from langchain_ollama import OllamaLLM

def test_bloom_and_quantity():
    llm = OllamaLLM(model="llama3", temperature=0)
    
    # Context with multiple facts to support higher Bloom's levels
    context = """
    Abduction is useful for diagnostic systems. It provides plausible explanations for observations.
    Deduction is used in logic programming. It reaches logically certain conclusions.
    Induction is the process of estimating the validity of observations of part of a class to all of the class.
    Hypothesis generation is a key part of abduction.
    Logic programming relies on mathematical logic.
    Diagnostic systems are used in medicine and engineering.
    """
    n = 6 # Requesting 6 to see if it covers all levels
    
    # Gen Prompt (Simplified version of the real one for testing)
    gen_prompt = f"""
    SYSTEM: You are an educational content generator.
    Your sole task is to create 4-option Multiple Choice Questions (MCQs) based on the provided data.

    <DATA_TO_USE>
    {context}
    </DATA_TO_USE>

    STRICT RULES:
    1. SUBJECT MATTER: Create questions ONLY from topics found inside <DATA_TO_USE>.
    2. MCQ STRUCTURE: Every question MUST have exactly 4 options.
    3. QUANTITY: Generate exactly {n} questions. This is mandatory.
    4. BLOOM'S DIVERSITY: You MUST cover all 6 levels. Use this guide:
       - Remember: Recall facts and basic concepts.
       - Understand: Explain ideas or concepts.
       - Apply: Use information in new situations.
       - Analyze: Draw connections among ideas.
       - Evaluate: Justify a stand or decision.
       - Create: Produce new or original work.
    5. OUTPUT: Return only the JSON array.
    """
    
    print(f"🧠 Testing generation for N={n} and all Bloom's levels...")
    try:
        response = llm.invoke(gen_prompt)
        print("--- RAW LLM RESPONSE ---")
        print(response)
        
        # Try to parse JSON
        import re
        match = re.search(r'\[.*\]', response, re.DOTALL)
        if match:
            data = json.loads(match.group(0))
            num_questions = len(data)
            levels_found = set(q.get('bloom_level') for q in data)
            
            print(f"📊 Questions generated: {num_questions} (Target: {n})")
            print(f"📊 Bloom's levels found: {levels_found}")
            
            if num_questions == n:
                print("✅ SUCCESS: Correct number of questions generated.")
            else:
                print(f"❌ FAILED: Generated {num_questions} questions, expected {n}.")
                
            if len(levels_found) >= 3: # Expecting at least some diversity in a single pass
                 print("✅ SUCCESS: Diverse Bloom's levels detected.")
            else:
                 print("❌ FAILED: Poor Bloom's diversity.")
        else:
            print("❌ FAILED: No JSON array found.")
            
    except Exception as e:
        print(f"❌ Error during test: {e}")

if __name__ == "__main__":
    test_bloom_and_quantity()
