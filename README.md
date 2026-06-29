# LangGraph AI Customer Support Automation System

**Developer:** Aditya Aryan  
**Email:** adityaryan2023@vitstudent.ac.in  

A **production-quality, fully local** customer support AI built with LangGraph, LangChain, Ollama (`qwen2.5:7b`), ChromaDB (RAG), and SQLite (memory + checkpoints).

---

## 🏗 Architecture

```
Customer Query → Intent Classifier → Conditional Router
                                           ↓
                    ┌──────────────────────┴──────────────────────┐
                    ↓            ↓              ↓                  ↓
               Sales Agent  Tech Agent    Billing Agent     Account Agent
                    └──────────────────────┬──────────────────────┘
                                           ↓
                                  SQLite Memory Retrieval
                                           ↓
                                     RAG Retrieval
                                           ↓
                                    Generate Draft
                                           ↓
                               Approval Required? (HITL)
                              YES ↓              NO ↓
                         Human Approval      Supervisor Agent
                               ↓                    ↓
                         Supervisor Agent    Final Response
                               ↓
                         Final Response
```

---

## 📁 Project Structure

```
customer_support_ai/
│
├── app.py                      # Main entry point — runs the graph interactively
├── graph.py                    # LangGraph StateGraph definition and compilation
├── state.py                    # Shared TypedDict State (flows through all nodes)
├── requirements.txt            # Python dependencies
├── .env                        # Active environment configuration
├── .env.example                # Environment template with documentation
├── create_sample_docs.py       # Generates the four company PDF documents
├── test_state.py               # Unit test for state.py
│
├── agents/                     # All LLM-powered agent nodes
│   ├── __init__.py
│   ├── classifier.py           # Intent classification + approval detection
│   ├── sales_agent.py          # Handles: pricing, features, upgrades
│   ├── technical_agent.py      # Handles: bugs, troubleshooting, how-to
│   ├── billing_agent.py        # Handles: invoices, payments, refunds
│   ├── account_agent.py        # Handles: login, password, cancellation
│   └── supervisor_agent.py     # Quality gate: verifies final response
│
├── rag/                        # Retrieval-Augmented Generation pipeline
│   ├── __init__.py
│   ├── loader.py               # PDF loading + text chunking
│   ├── vectorstore.py          # ChromaDB setup + document ingestion
│   └── retriever.py            # Query embedding + top-K retrieval
│
├── memory/                     # SQLite conversation memory
│   ├── __init__.py
│   └── sqlite_memory.py        # Read/write conversation history
│
├── utils/                      # Shared utilities
│   ├── __init__.py
│   ├── router.py               # Conditional edge functions for LangGraph
│   ├── approval.py             # HITL interrupt + resume logic
│   └── prompts.py              # All prompt templates (one per agent)
│
├── documents/                  # Company source documents for RAG
│   ├── company_policy.pdf
│   ├── pricing_guide.pdf
│   ├── technical_manual.pdf
│   └── faq.pdf
│
├── database/                   # Persistent storage
│   ├── memory.db               # SQLite: conversation history + LangGraph checkpoints
│   └── chroma_db/              # ChromaDB vector store (auto-created on first run)
│
└── screenshots/                # Store demo screenshots for submission
```

---

## ⚙️ Prerequisites

| Requirement | Version | Notes |
|---|---|---|
| Python | 3.11 | Required |
| Ollama | Latest | Must be running locally |
| qwen2.5:7b | — | `ollama pull qwen2.5:7b` |
| nomic-embed-text | — | `ollama pull nomic-embed-text` |

---

## 🚀 Installation

### 1. Pull required Ollama models
```bash
ollama pull qwen2.5:7b
ollama pull nomic-embed-text
```

### 2. Clone / navigate to project
```bash
cd customer_support_ai
```

### 3. Install Python dependencies
```bash
pip install -r requirements.txt
```

### 4. Create sample company documents
```bash
python create_sample_docs.py
```

### 5. Initialize the RAG vector store
```bash
python -c "from rag.vectorstore import initialize_vectorstore; initialize_vectorstore()"
```

### 6. Run the application
```bash
python app.py
```

---

## 🧪 Testing

Each module has its own test script. Run them in order:

```bash
# Module 1: State
python test_state.py

# Module 2: SQLite Memory
python memory/test_memory.py

# Module 3: RAG
python rag/test_rag.py

# Module 4: Intent Classifier
python agents/test_classifier.py

# Full system demo (all 5 queries)
python app.py --demo
```

---

## 💬 Demo Queries

| # | Query | Expected Route | HITL? |
|---|---|---|---|
| 1 | "What are the pricing plans available?" | Sales Agent | No |
| 2 | "I forgot my password." | Account Agent | No |
| 3 | "My application crashes when I upload a file." | Technical Agent | No |
| 4 | "I need a refund." | Billing Agent → Human Approval | **Yes** |
| 5 | "What was my previous support issue?" | Memory Recall → Supervisor | No |

---

## 🤖 Models Used

| Purpose | Model | Provider |
|---|---|---|
| Intent Classification | qwen2.5:7b | Ollama (local) |
| All Support Agents | qwen2.5:7b | Ollama (local) |
| Supervisor Agent | qwen2.5:7b | Ollama (local) |
| Document Embeddings | nomic-embed-text | Ollama (local) |
| Vector Store | ChromaDB | Local filesystem |
| Memory Store | SQLite | Local filesystem |

---

## 🔒 Human-in-the-Loop (HITL)

The following requests automatically pause the graph and require human approval:

- Refund requests
- Subscription cancellation
- Account closure
- Compensation requests
- Escalation to management

When HITL is triggered:
1. The graph **pauses** and saves its state to SQLite (via LangGraph checkpoint)
2. You are shown the customer query + agent draft + approval prompt
3. Enter `approve`, `reject`, or `edit [your version]`
4. The graph **resumes** from the saved checkpoint

---

## 📋 Submission Checklist

- [ ] `state.py` — TypedDict state with all required fields
- [ ] `agents/classifier.py` — Intent classification with 5 intents
- [ ] `agents/sales_agent.py` — Sales domain agent
- [ ] `agents/technical_agent.py` — Technical support agent
- [ ] `agents/billing_agent.py` — Billing agent with HITL trigger
- [ ] `agents/account_agent.py` — Account agent with HITL trigger
- [ ] `agents/supervisor_agent.py` — Quality gate supervisor
- [ ] `rag/loader.py` — PDF loading and chunking
- [ ] `rag/vectorstore.py` — ChromaDB setup and ingestion
- [ ] `rag/retriever.py` — Top-K retrieval with Ollama embeddings
- [ ] `memory/sqlite_memory.py` — Conversation history read/write
- [ ] `utils/router.py` — Conditional edge functions
- [ ] `utils/approval.py` — HITL interrupt/resume logic
- [ ] `utils/prompts.py` — All prompt templates
- [ ] `graph.py` — Complete LangGraph StateGraph
- [ ] `app.py` — Interactive entry point
- [ ] All 5 demo queries working correctly
- [ ] `screenshots/` folder populated
