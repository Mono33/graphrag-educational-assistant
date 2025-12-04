# 🎓 GraphRAG Educational Assistant

AI-powered educational assistant that combines Knowledge Graphs (Neo4j) with Retrieval-Augmented Generation (RAG) to provide personalized teaching recommendations in Italian.

![Python](https://img.shields.io/badge/python-3.10+-blue.svg)
![Neo4j](https://img.shields.io/badge/neo4j-5.0+-green.svg)
![Streamlit](https://img.shields.io/badge/streamlit-1.28+-red.svg)
![OpenAI](https://img.shields.io/badge/openai-GPT--4o-orange.svg)
![Node2Vec](https://img.shields.io/badge/Node2Vec-Enabled-brightgreen.svg)

---

## 🚀 **Quick Start**

**New to this project? Start here!** → [**QUICKSTART.md**](QUICKSTART.md)

Get up and running in 5 minutes with our step-by-step guide.

---

## 🌟 Features

- **🌍 Multilingual Query Processing**: Handles Italian and English educational queries with intelligent translation
- **🔍 Hybrid Graph Retrieval**: Combines traditional graph traversal with Node2Vec semantic search for superior results
- **🎯 Educational Context Building**: Structures recommendations specifically for teaching methodologies
- **💬 Natural Language Generation**: Produces natural Italian responses tailored for educators
- **🖥️ Interactive Streamlit Interface**: Beautiful, user-friendly demo for live presentations
- **📊 Evidence-Based Responses**: All recommendations backed by knowledge graph evidence

---

## 🏗️ Architecture

```
User Query (Italian/English)
    ↓
Text2Cypher Converter (Multilingual)
    ↓
Hybrid Graph Retriever (Neo4j + Node2Vec)
    ├── Direct Graph Traversal
    ├── Semantic Search (Node2Vec)
    └── Neighbor Expansion
    ↓
Context Builder (Structured Educational Context)
    ↓
Response Generator (GPT-4o LLM)
    ↓
Natural Italian Response
```

### Core Components:

1. **Text2Cypher** (`text2cypher.py`, `multilingual_text2cypher.py`)
   - Converts natural language queries to Cypher
   - Supports Italian and English with intelligent translation
   - Self-repairing Cypher syntax

2. **Graph Retriever** (`graph_retriever.py`)
   - Hybrid retrieval: Graph + Vector search
   - Node2Vec embeddings for semantic similarity
   - Neighbor expansion for comprehensive results

3. **Context Builder** (`context_builder.py`)
   - Structures raw graph data into educational context
   - Methodology recommendations with confidence levels
   - Student profile generation

4. **Response Generator** (`llm_chain.py`)
   - Generates natural Italian responses
   - Evidence-based recommendations
   - Confidence assessment

5. **Streamlit Interface** (`streamlit_app.py`)
   - Interactive web application
   - Real-time pipeline visualization
   - Evidence and comparison views

6. **FastAPI Module** (`api/`) 🆕
   - REST API for external integrations
   - Provides structured context for prompt injection
   - See [API_INTEGRATION_GUIDE.md](API_INTEGRATION_GUIDE.md) for details

---

## 🚀 Quick Start

### Prerequisites

- Python 3.10+
- Neo4j Database (local or Aura)
- OpenAI API Key

### Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/Mono33/graphrag-educational-assistant.git
   cd graphrag-educational-assistant
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements_streamlit.txt
   ```

3. **Configure environment**
   ```bash
   # Copy the example environment file
   cp .env.example .env
   
   # Edit .env with your credentials
   # - NEO4J_URI: Your Neo4j connection URI
   # - NEO4J_USER: Your Neo4j username
   # - NEO4J_PASSWORD: Your Neo4j password
   # - OPENAI_API_KEY: Your OpenAI API key
   ```

4. **Set up Neo4j database**
   ```bash
   # Process your data (if starting fresh)
   python process_data_graph4.py
   
   # Ingest data into Neo4j
   python data_ingestion_neo4j.py --file concepts4_neo4j.json --password YOUR_PASSWORD --clear
   ```

5. **Train Node2Vec model** (optional, pre-trained models included)
   ```bash
   python train_node2vec.py
   ```

6. **Run the Streamlit app**
```bash
   streamlit run streamlit_app.py
   ```

7. **Access the app**
   - Open your browser to: `http://localhost:8501`
   - Start asking educational questions in Italian! 🇮🇹

---

## 🔧 Configuration

### Environment Variables

Create a `.env` file with:

```env
# Neo4j Configuration
NEO4J_URI=bolt://localhost:7687
NEO4J_USER=neo4j
NEO4J_PASSWORD=your_password

# OpenAI Configuration
OPENAI_API_KEY=sk-your-openai-key

# Node2Vec Model Path
NODE2VEC_MODEL_DIR=./models
```

### Neo4j Setup

Your Neo4j database should contain:
- **Nodes**: Educational concepts, methodologies, strategies
- **Relationships**: BELONGS_TO, ADDRESSES, SUPPORTS, etc.
- **Properties**: name, description, category, etc.

See `data_ingestion_neo4j.py` for the ingestion script.

---

## 📊 Node2Vec Models

The system uses Node2Vec for semantic search. Pre-trained models are in `models/`:
- `educational_node2vec_embeddings.npz`: Node embeddings (109 nodes)
- `educational_node2vec_config.json`: Model configuration
- `educational_node2vec.pkl`: Trained model

### Retrain Node2Vec:
```bash
python train_node2vec.py
```

**Model Parameters:**
- Dimensions: 128
- Walk Length: 80
- Num Walks: 200
- Workers: 4
- P: 1.0, Q: 1.0

---

## 🎯 Usage Examples

### Example Queries (in Italian):

1. **"Ci sono strategie per studenti con ADHD?"**
   - *Are there strategies for students with ADHD?*

2. **"Metodologie per studenti dello spettro autistico?"**
   - *Methodologies for students on the autism spectrum?*

3. **"Come aiutare studenti senza motivazione?"**
   - *How to help students without motivation?*

4. **"Il mio studente ha ADHD, cosa posso fare?"**
   - *My student has ADHD, what can I do?*

### Example Response:

```
Sì, ci sono diverse strategie efficaci per studenti con ADHD:

1. Flipped Classroom (ALTA confidenza)
   - Permette agli studenti di lavorare al proprio ritmo
   - Riduce la pressione del tempo in classe
   - Favorisce l'apprendimento attivo

2. Cooperative Learning (ALTA confidenza)
   - Sviluppa competenze sociali
   - Migliora l'attenzione attraverso l'interazione
   ...
```

---

## 🛠️ Development

### Project Structure

```
graphaixlearning/
├── streamlit_app.py                # Streamlit interface
├── graph_retriever.py              # Hybrid retrieval + Node2Vec
├── text2cypher.py                  # Base Text2Cypher
├── multilingual_text2cypher.py     # Multilingual support
├── context_builder.py              # Context structuring
├── llm_chain.py                    # Response generation
├── config.py                       # Configuration management
├── train_node2vec.py               # Node2Vec training
├── data_ingestion_neo4j.py         # Neo4j data import
├── api/                            # 🆕 FastAPI module for integrations
│   ├── main.py                     # FastAPI app entry point
│   ├── routes/context.py           # /api/v1/context endpoint
│   ├── schemas/models.py           # Pydantic models
│   └── graphrag_client.py          # Helper client for DEV team
├── models/                         # Node2Vec models
│   ├── educational_node2vec_embeddings.npz
│   ├── educational_node2vec_config.json
│   └── educational_node2vec.pkl
├── requirements.txt                # Dependencies
├── .env.example                    # Environment template
├── API_INTEGRATION_GUIDE.md        # 🆕 API documentation
└── README.md                       # This file
```

### Testing

Test scripts are available in the `NOTPUSHED/` folder for local development.

### Adding New Data

1. **Prepare data**: Format as JSON (see `concepts4_neo4j.json`)
2. **Process**: Run `process_data_graph4.py` if needed
3. **Ingest**: Run `data_ingestion_neo4j.py --file your_data.json --clear`
4. **Retrain Node2Vec**: Run `train_node2vec.py`
5. **Test**: Launch app and verify results

---

## 📚 Additional Resources

For additional documentation and guides, see the `NOTPUSHED/` folder (local development only).

---

## 🧪 Technologies Used

- **[Neo4j](https://neo4j.com/)**: Graph database for knowledge representation
- **[OpenAI GPT-4o](https://openai.com/)**: Language model for response generation
- **[LangChain](https://langchain.com/)**: LLM application framework
- **[Streamlit](https://streamlit.io/)**: Web interface framework
- **[Node2Vec](https://github.com/eliorc/node2vec)**: Graph embedding for semantic search
- **[NetworkX](https://networkx.org/)**: Graph analysis library
- **[Pandas](https://pandas.pydata.org/)**: Data manipulation
- **[NumPy](https://numpy.org/)**: Numerical computing

---

## 🎓 Educational Context

This system is designed specifically for **teachers and educators** who need:
- Evidence-based teaching strategies
- Recommendations for students with special needs (ADHD, autism, etc.)
- Personalized methodology suggestions
- Quick access to educational best practices

All responses are:
- ✅ In Italian (primary audience)
- ✅ Based on knowledge graph evidence
- ✅ Structured for classroom implementation
- ✅ Confidence-assessed for reliability

---

## 🚀 Deployment Options

### 1. Local Development (Streamlit)
```bash
streamlit run streamlit_app.py
```

### 2. API Integration (FastAPI) 🆕
```bash
uvicorn api.main:app --reload --port 8000
```
See [API_INTEGRATION_GUIDE.md](API_INTEGRATION_GUIDE.md) for full documentation.

### 3. Cloud Deployment
- Use Neo4j Aura for cloud database
- Deploy Streamlit app to Streamlit Cloud, Heroku, or AWS
- Contact the development team for deployment guides

---

## 📈 Performance

- **Average Query Time**: 2-4 seconds
- **Retrieval Accuracy**: 85%+ relevant results
- **Node2Vec Coverage**: 109 educational concepts
- **Supported Languages**: Italian (primary), English (queries)

---

## 🤝 Contributing

This is an educational project. If you want to contribute:
1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Submit a pull request

---

## 📞 Support & Contact

For questions or issues:
- Open an issue on GitHub
- Contact: [Your Email/FEM-modena]

---

## 📝 License

[Specify your license here - MIT, Apache 2.0, etc.]

---

## 👥 Authors

- **Louis Mono, Angelo Casali** - Initial development - [Mono33](https://github.com/Mono33)
- **FEM-modena Team** - Educational content and validation

---

## 🙏 Acknowledgments

- Built for educational purposes
- Knowledge graph curated by FEM-modena educational team
- Powered by Neo4j, OpenAI, and open-source technologies

---

## 📊 Future Enhancements

- [ ] Multi-language response generation (English, Spanish)
- [x] ~~Integration with existing educational agent~~ ✅ **DONE** (FastAPI module added)
- [ ] Expanded knowledge graph (500+ concepts)
- [ ] Student progress tracking
- [ ] Collaborative filtering for recommendations
- [ ] Mobile-responsive interface

---

**Made with ❤️ for educators by educators**
