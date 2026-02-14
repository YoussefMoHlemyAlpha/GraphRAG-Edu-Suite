import pytesseract
from pdf2image import convert_from_path
import tempfile
from PyPDF2 import PdfReader
from langchain_experimental.graph_transformers import LLMGraphTransformer
from langchain_core.documents import Document
from engine.graph_store import QuizGraphStore
from langchain_text_splitters import RecursiveCharacterTextSplitter

def perform_ocr(pdf_file):
    """Fallback OCR extraction for scanned PDFs."""
    print(f"📷 Starting OCR on {pdf_file.name}...")
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        tmp.write(pdf_file.getvalue())
        tmp_path = tmp.name
    
    try:
        images = convert_from_path(tmp_path)
        extracted_text = ""
        for i, image in enumerate(images):
            text = pytesseract.image_to_string(image)
            extracted_text += f"\n--- Page {i+1} ---\n{text}"
        return extracted_text.strip()
    finally:
        import os
        if os.path.exists(tmp_path):
            os.remove(tmp_path)

def process_pdf_to_graph(pdf_file, llm):
    store = QuizGraphStore()
    
    # 1. Extract and Chunk Text
    reader = PdfReader(pdf_file)
    raw_text = "\n".join([p.extract_text() for p in reader.pages if p.extract_text()]).strip()
    
    # Threshold: If less than 100 characters extracted, try OCR
    if len(raw_text) < 100:
        print(f"🔍 Low text density ({len(raw_text)} chars). Attempting OCR fallback...")
        try:
            raw_text = perform_ocr(pdf_file)
        except Exception as e:
            print(f"❌ OCR Failed: {str(e)}")
            if not raw_text:
                raise ValueError(f"No readable text found in {pdf_file.name} and OCR failed.")

    if not raw_text:
        print(f"⚠️ Failed to extract text from {pdf_file.name}")
        raise ValueError(f"No readable text found in {pdf_file.name}. It might be an image-only PDF or empty.")

    lesson_name = pdf_file.name.replace(".pdf", "")
    print(f"📖 Processing: {lesson_name} ({len(raw_text)} characters)")

    # Splitting into chunks ensures the LLM doesn't miss details
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=2000, chunk_overlap=200)
    chunks = text_splitter.split_text(raw_text)
    docs = [Document(page_content=chunk, metadata={"lesson": lesson_name}) for chunk in chunks]
    print(f"🧩 Split text into {len(docs)} chunks for deep extraction.")

    # 2. Transform (Knowledge Graph Extraction)
    transformer = LLMGraphTransformer(
        llm=llm,
        allowed_nodes=["Concept", "Definition", "Process", "Characteristic", "Relationship", "Method", "Example"],
        allowed_relationships=["FOLLOWS", "DESCRIBES", "IMPLEMENTS", "PART_OF", "CAUSES", "SIMILAR_TO", "CONTRASTS_WITH", "MENTIONS","USED_IN"],
        node_properties=False 
    )
    
    print(f"🧠 Extracting from {len(docs)} chunks (this may take a few minutes)...")
    graph_docs = transformer.convert_to_graph_documents(docs)
    
    # Check if anything was actually extracted
    total_nodes = sum(len(g_doc.nodes) for g_doc in graph_docs)
    if total_nodes == 0:
        print(f"⚠️ No entities extracted from {lesson_name}")
        raise ValueError(f"The AI could not identify any key concepts or relationships in {lesson_name}. The content might be too short or unrelated to the schema.")

    # 3. Load into Neo4j
    store.graph.add_graph_documents(graph_docs, baseEntityLabel=True, include_source=True)
    
    # 4. Manual Linking (The Reliability Booster)
    # We loop through all processed chunks and link their nodes
    linked_nodes = 0
    for g_doc in graph_docs:
        if g_doc.nodes:
            for node in g_doc.nodes:
                link_single_query = """
                MERGE (l:Lesson {name: $lesson})
                MERGE (n:__Entity__ {id: $node_id})
                MERGE (n)-[:PART_OF]->(l)
                """
                store.query(link_single_query, {"lesson": lesson_name, "node_id": node.id})
                linked_nodes += 1
    
    print(f"🔗 Linked {linked_nodes} unique concepts to lesson: {lesson_name}")
    
    # Also ensure the Document node itself is linked
    link_doc_query = """
    MERGE (l:Lesson {name: $lesson})
    WITH l
    MATCH (d:Document) WHERE d.lesson = $lesson
    MERGE (d)-[:PART_OF]->(l)
    """
    store.query(link_doc_query, {"lesson": lesson_name})
    
    return lesson_name
