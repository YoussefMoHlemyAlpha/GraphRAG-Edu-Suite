# System Architecture: GraphRAG-Edu-Suite

This project is an advanced **Graph-based Retrieval-Augmented Generation (GraphRAG)** learning system. It transforms static documents into a dynamic knowledge graph to generate high-quality educational content.

## 💡 The Full Idea: Vision to Intelligence

The core concept is to take **any** study material (even handwritten or scanned notes) and turn it into a **Living Knowledge Base** that can test you, grade you, and help you master a subject.

```mermaid
flowchart LR
    subgraph INPUT ["📂 STEP 1: INPUT"]
        direction TB
        A1["Digital PDF"]
        A2["Scanned Notes/Images"]
    end

    subgraph BRAIN ["🧠 STEP 2: LOCAL AI (Ollama)"]
        direction TB
        B1["LLaVA 'Sees' the text"]
        B2["Llama 3 'Understands' concepts"]
        B3["Graph Transformer maps links"]
    end

    subgraph KNOWLEDGE ["🕸️ STEP 3: THE GRAPH (Neo4j)"]
        direction TB
        C1["Nodes: Facts & Definitions"]
        C2["Edges: Logical Relationships"]
    end

    subgraph OUTPUT ["🎓 STEP 4: LEARNING"]
        direction TB
        D1["Custom MCQ Quizzes"]
        D2["Deep Essay Prompts"]
        D3["AI Grading & Feedback"]
    end

    INPUT --> BRAIN
    BRAIN --> KNOWLEDGE
    KNOWLEDGE --> OUTPUT
    
    %% Styling
    style INPUT fill:#f5f5f5,stroke:#333
    style BRAIN fill:#e1f5fe,stroke:#01579b
    style KNOWLEDGE fill:#fff3e0,stroke:#e65100
    style OUTPUT fill:#e8f5e9,stroke:#1b5e20
```

---

## 🏗️ Technology Stack

| Component | Responsibility |
| :--- | :--- |
| **Streamlit** | Frontend UI and Session Management. |
| **LangChain** | Orchestration framework for LLM chains and Graph transformers. |
| **Ollama** | Local inference engine for Llama 3 (Text) and LLaVA (Vision). |
| **Neo4j (Aura)** | Graph Database for storing entities and relationships. |
| **PyMuPDF / VLM** | Text extraction from digital and scanned PDFs. |

## 🔄 End-to-End System Diagram

This diagram illustrates the full lifecycle of a document in the GraphRAG-Edu-Suite, from the initial upload to the final AI-powered evaluation.

```mermaid
flowchart TD
    %% 1. Ingestion Stage
    subgraph STAGE_1 ["1. Ingestion & Detection"]
        PDF([".pdf Document"]) --> IS_SCANNED{Is Scanned?}
        IS_SCANNED -- "No (Text)" --> TEXT_EXTRACT["PyPDF2/PyMuPDF Extraction"]
        IS_SCANNED -- "Yes (Images)" --> VISION_LLM[/"LLaVA (Vision Model)"/]
    end

    %% 2. Transformation Stage
    subgraph STAGE_2 ["2. Knowledge Engineering"]
        TEXT_EXTRACT --> CHUNKING["Recursive Character Chunks"]
        VISION_LLM --> CHUNKING
        CHUNKING --> LLM_TRANSFORM[/"Llama 3 (Graph Transformer)"/]
        LLM_TRANSFORM --> NODES["Entities (Concepts, Processes)"]
        LLM_TRANSFORM --> RELS["Relationships (PART_OF, CAUSES)"]
    end

    %% 3. Persistence Stage
    subgraph STAGE_3 ["3. Graph Storage (NEO4J)"]
        NODES --> GRAPH_DB[("Neo4j Aura DB")]
        RELS --> GRAPH_DB
        LESSON_LINK["Manual Lesson Linker"] --> GRAPH_DB
    end

    %% 4. Retrieval & Generation Stage
    subgraph STAGE_4 ["4. GraphRAG Execution"]
        USER_QUERY([User Choice: Lesson]) --> CYPHER_GEN["Cypher Query Engine"]
        GRAPH_DB <--> CYPHER_GEN
        CYPHER_GEN --> FACTS["Relational Context (Facts)"]
        FACTS --> QUIZ_GEN[/"Llama 3 (Quiz Gen)"/]
        FACTS --> ESSAY_GEN[/"Llama 3 (Essay Gen)"/]
    end

    %% 5. Evaluation & UI Stage
    subgraph STAGE_5 ["5. Feedback & UI"]
        QUIZ_GEN --> ST_UI[["Streamlit Dashboard"]]
        ESSAY_GEN --> ST_UI
        ST_UI -- "User Input" --> EVAL_LLM[/"Llama 3 (Grade Auditor)"/]
        EVAL_LLM --> REPORT["Performance Analytics"]
        REPORT --> ST_UI
    end

    %% Styles
    classDef llm fill:#f9f,stroke:#333,stroke-width:2px;
    classDef db fill:#00f,stroke:#fff,stroke-width:2px,color:#fff;
    class VISION_LLM,LLM_TRANSFORM,QUIZ_GEN,ESSAY_GEN,EVAL_LLM llm;
    class GRAPH_DB db;
```

## ⏳ The RAG Workflow (Sequence)

This sequence diagram shows exactly what happens, step-by-step, when you click **"Generate Quiz"**.

```mermaid
sequenceDiagram
    autonumber
    actor User
    participant App as Streamlit UI
    participant RAG as RAG Controller
    participant DB as Neo4j Graph
    participant LLM as Llama 3 (Ollama)

    User->>App: Selects Lesson & clicks "Generate Quiz"
    App->>RAG: Request Quiz (Topic, Difficulty)
    
    Note over RAG, DB: 1. RETRIEVAL PHASE
    RAG->>DB: MATCH (n)-[r]->(m) WHERE ... (Cypher Query)
    DB-->>RAG: Returns Related Facts (Nodes + Edges)
    
    Note over RAG, LLM: 2. AUGMENTATION PHASE
    RAG->>RAG: Formats Facts into Context String
    RAG->>LLM: Send Prompt + Context + Strict JSON Schema
    
    Note over LLM: 3. GENERATION PHASE
    LLM-->>RAG: Generates Quiz JSON (Questions + Options)
    
    Note over RAG, LLM: 4. SELF-CORRECTION (CRITIC)
    RAG->>LLM: "Critique this quiz against the facts"
    LLM-->>RAG: Returns Verified JSON
    
    RAG-->>App: Display Final Quiz
    App-->>User: Shows Interactive Quiz
```

### 1. The Knowledge Extraction Phase
- **Chunking**: Documents are split into overlapping chunks (1000-2000 chars) so the LLM doesn't lose context.
- **Entity Extraction**: `LLMGraphTransformer` uses Llama 3 to identify "Concepts", "Processes", and "Relationships".
- **Graph Ingestion**: These entities are stored in Neo4j, creating a web of connected knowledge rather than just a flat list of text.

### 2. The Retrieval Phase (GraphRAG)
Unlike standard RAG (which just finds similar text chunks), **GraphRAG** traverses relationships. 
- When generating a quiz, the system queries Neo4j for specific facts and their logical links (e.g., `Concept A --[IMPLEMENTS]--> Process B`).
- This prevents hallucinations by forcing the LLM to use only established graph paths.

### 3. The Generation Phase
- **MCQ Quiz**: The LLM creates questions with 1 correct answer and 3 distractors based solely on graph facts.
- **Essay Lab**: The system challenges students to connect disparate nodes in the graph (e.g., "Explain how Process B affects Concept C").
- **Self-Audit**: A two-step "Critic" loop verifies the generated JSON before showing it to the user.

## 🧠 Model Roles
- **Llama 3 (8B)**: The "Brain". Handles logic, JSON formatting, and complex relationship extraction.
- **LLaVA**: The "Eyes". Specifically used when a PDF is detected as an image to "read" the text visually.
