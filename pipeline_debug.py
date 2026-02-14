import os
from dotenv import load_dotenv
from langchain_ollama import OllamaLLM
from langchain_experimental.graph_transformers import LLMGraphTransformer
from langchain_core.documents import Document

load_dotenv()

def debug_pipeline():
    llm = OllamaLLM(model="llama3", temperature=0)
    transformer = LLMGraphTransformer(llm=llm)
    
    text = "The Mitochondria is the powerhouse of the cell. It produces ATP."
    lesson_name = "Biology_Test"
    doc = [Document(page_content=text, metadata={"lesson": lesson_name})]
    
    print("🧠 Step 1: Extracting...")
    graph_docs = transformer.convert_to_graph_documents(doc)
    
    if not graph_docs:
        print("❌ Error: graph_docs is empty!")
        return

    g_doc = graph_docs[0]
    print(f"✅ Extracted {len(g_doc.nodes)} nodes and {len(g_doc.relationships)} relationships.")
    
    # Check if the individual nodes have a connection to the source document in the data structure
    # Actually, LLMGraphTransformer doesn't add the MENTIONED_IN relationship to graph_docs.
    # It's add_graph_documents that creates the link IF include_source=True.
    
    print(f"--- Nodes Extracted ---")
    for node in g_doc.nodes:
        print(f"Node: {node.id}")
        
    print(f"--- Relationships Extracted ---")
    for rel in g_doc.relationships:
        print(f"Rel: {rel.source.id} -[{rel.type}]-> {rel.target.id}")
        
    print(f"Source Metadata: {g_doc.source.metadata}")

if __name__ == "__main__":
    debug_pipeline()
