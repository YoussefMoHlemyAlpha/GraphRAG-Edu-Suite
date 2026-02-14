from langchain_core.messages import HumanMessage
import fitz  # PyMuPDF
import base64
import io
from PIL import Image
from PyPDF2 import PdfReader
from langchain_experimental.graph_transformers import LLMGraphTransformer
from langchain_core.documents import Document
from engine.graph_store import QuizGraphStore
from langchain_text_splitters import RecursiveCharacterTextSplitter

def extract_text_vlm(pdf_file, vision_llm):
    """Fallback VLM extraction for scanned PDFs."""
    print(f"👁️ Starting VLM extraction on {pdf_file.name}...")
    
    # Reset file pointer and read bytes
    pdf_file.seek(0)
    pdf_bytes = pdf_file.read()
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    
    total_pages = len(doc)
    print(f"📄 PDF has {total_pages} pages.")
    
    extracted_text = ""
    for i in range(total_pages):
        print(f"⏳ Processing page {i+1}/{total_pages}...")
        page = doc[i]
        pix = page.get_pixmap(matrix=fitz.Matrix(2, 2)) # Higher resolution
        img_data = pix.tobytes("png")
        
        # Base64 encode
        base64_image = base64.b64encode(img_data).decode('utf-8')
        
        # VLM Call using HumanMessage
        message = HumanMessage(
            content=[
                {"type": "text", "text": "Extract all readable text from this page exactly as it appears. If there is no text, return an empty string."},
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/png;base64,{base64_image}"},
                },
            ]
        )
        
        try:
            response = vision_llm.invoke([message])
            page_text = response.content.strip()
            if page_text:
                print(f"✅ Page {i+1} extracted ({len(page_text)} chars). Preview: {page_text[:50]}...")
            else:
                print(f"⚠️ Page {i+1} returned empty text.")
            extracted_text += f"\n--- Page {i+1} ---\n{page_text}"
        except Exception as e:
            print(f"❌ VLM error on page {i+1}: {str(e)}")
            raise e
            
    doc.close()
    return extracted_text.strip()

def extract_text_pypdf2(pdf_file):
    """Standard text extraction using PyPDF2."""
    reader = PdfReader(pdf_file)
    return "\n".join([p.extract_text() for p in reader.pages if p.extract_text()]).strip()

def process_pdf_to_graph(pdf_file, llm, vision_llm=None):
    store = QuizGraphStore()
    
    # 1. Extract Text
    raw_text = extract_text_pypdf2(pdf_file)
    
    # Check if we need VLM fallback
    if len(raw_text) < 100 and vision_llm:
        print(f"🔍 Low text density ({len(raw_text)} chars). Attempting VLM fallback...")
        try:
            raw_text = extract_text_vlm(pdf_file, vision_llm)
        except Exception as e:
            print(f"❌ VLM Extraction Failed: {str(e)}")
            if not raw_text:
                raise ValueError(f"No readable text found in {pdf_file.name} and VLM failed. Error: {str(e)}")

    if not raw_text:
        print(f"⚠️ Failed to extract text from {pdf_file.name}")
        raise ValueError(f"No readable text found in {pdf_file.name}. It might be an image-only PDF or empty.")

    lesson_name = pdf_file.name.replace(".pdf", "")
    print(f"📖 Processing: {lesson_name} ({len(raw_text)} characters)")

    # 1. Chunking
    # 8B models struggle with large contexts and complex JSON outputs
    is_high_accuracy = "70b" in getattr(llm, "model_name", "").lower()
    chunk_size = 2000 if is_high_accuracy else 1000
    chunk_overlap = 200 if is_high_accuracy else 100

    # Splitting into chunks ensures the LLM doesn't miss details
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    chunks = text_splitter.split_text(raw_text)
    docs = [Document(page_content=chunk, metadata={"lesson": lesson_name}) for chunk in chunks]
    print(f"🧩 Split text into {len(docs)} chunks for {'Deep' if is_high_accuracy else 'Safe'} extraction.")

    # 2. Transform (Knowledge Graph Extraction)
    if is_high_accuracy:
        nodes = ["Concept", "Definition", "Process", "Characteristic", "Relationship", "Method", "Example"]
        rels = ["FOLLOWS", "DESCRIBES", "IMPLEMENTS", "PART_OF", "CAUSES", "SIMILAR_TO", "CONTRASTS_WITH", "MENTIONS", "USED_IN"]
    else:
        print(f"🛠️ Using free-form schema for {getattr(llm, 'model_name', 'fallback model')}")
        # Mixtral works best when it can naturally define labels without strict tool-calling constraints.
        nodes = None
        rels = None

    transformer = LLMGraphTransformer(
        llm=llm,
        allowed_nodes=nodes,
        allowed_relationships=rels,
        node_properties=False 
    )
    
    print(f"🧠 Extracting from {len(docs)} chunks (using {'70B' if is_high_accuracy else '8B'} model)...")
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
