import json
from langchain_ollama import OllamaLLM
from engine.generator import generate_graph_quiz
from engine.graph_store import QuizGraphStore
import os
from dotenv import load_dotenv

load_dotenv()

class MockLLM:
    def __init__(self, real_llm):
        self.real_llm = real_llm
    def invoke(self, prompt):
        return self.real_llm.invoke(prompt)

def check_end_to_end():
    llm = OllamaLLM(model="llama3", temperature=0)
    lesson_name = "KBAI01_Reasoning_deduction, induction, abduction_Saleh_PPT"
    n = 6
    
    print(f"🚀 Running End-to-End Quiz Generation Test for '{lesson_name}' with N={n}...")
    
    try:
        quiz = generate_graph_quiz(llm, lesson_name, n)
        
        num_questions = len(quiz)
        levels = set(q.get('bloom_level') for q in quiz)
        
        print(f"📊 Final Questions count: {num_questions}")
        print(f"📊 Bloom's Diversity: {levels}")
        
        if num_questions == n:
            print("✅ SUCCESS: Exact quantity matched.")
        else:
            print(f"❌ FAILED: Found {num_questions} questions, expected {n}.")
            
        if len(levels) >= 4: # Satisfactory diversity for an LLM
            print("✅ SUCCESS: High Bloom's diversity achieved.")
        else:
            print(f"⚠️ WARNING: Moderate Bloom's diversity ({len(levels)} levels).")
            
        # Check for 4 options
        options_ok = all(len(q.get('options', [])) == 4 for q in quiz)
        if options_ok:
            print("✅ SUCCESS: All questions have 4 options.")
        else:
            print("❌ FAILED: Some questions are missing options.")

    except Exception as e:
        print(f"❌ Error during end-to-end test: {e}")

if __name__ == "__main__":
    check_end_to_end()
