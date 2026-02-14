import os
from dotenv import load_dotenv
from langchain_ollama import OllamaLLM
from langchain_experimental.graph_transformers import LLMGraphTransformer
from langchain_core.documents import Document

load_dotenv()

def test_extraction():
    llm = OllamaLLM(model="llama3", temperature=0)
    transformer = LLMGraphTransformer(llm=llm)
    
    text = "The quick brown fox jumps over the lazy dog. The fox is a mammal."
    doc = [Document(page_content=text)]
    
    print("🧠 Testing extraction with llama3...")
    try:
        graph_docs = transformer.convert_to_graph_documents(doc)
        if graph_docs:
            nodes = graph_docs[0].nodes
            rels = graph_docs[0].relationships
            print(f"✅ Success! Extracted {len(nodes)} nodes and {len(rels)} relationships.")
            for node in nodes:
                print(f" - Node: {node.id} ({node.type})")
            for rel in rels:
                print(f" - Relation: {rel.source.id} --{rel.type}--> {rel.target.id}")
        else:
            print("❌ Failed: graph_docs is empty.")
    except Exception as e:
        print(f"❌ Error during extraction: {e}")

if __name__ == "__main__":
    test_extraction()
