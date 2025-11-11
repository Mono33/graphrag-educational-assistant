# 🚀 GraphRAG Educational Assistant - Quick Start Guide

**Get up and running in 5 minutes!**

---

## 📋 **Prerequisites**

- Python 3.9+
- Neo4j Aura account (free tier available)
- OpenAI API key

---

## 1️⃣ **Clone & Install**

```bash
# Clone repository
git clone https://github.com/Mono33/graphrag-educational-assistant
cd graphrag-educational-assistant/graphaixlearning

# Create virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

---

## 2️⃣ **Configure Environment**

Create a `.env` file in the `graphaixlearning` directory:

```bash
# Copy example (if available)
cp .env.example .env

# Or create manually with these keys:
```

```ini
# Neo4j Aura Configuration
NEO4J_URI=neo4j+s://YOUR_AURA_URI.databases.neo4j.io
NEO4J_USER=neo4j
NEO4J_PASSWORD=your_neo4j_password_here

# OpenAI Configuration
OPENAI_API_KEY=sk-your-openai-api-key-here
OPENAI_MODEL=gpt-4o  # or gpt-3.5-turbo-16k

# Optional: Node2Vec Configuration
USE_NODE2VEC=true
NODE2VEC_DIMENSIONS=128
```

**Where to get credentials**:
- **Neo4j Aura**: https://neo4j.com/cloud/aura/ (free tier available)
- **OpenAI API Key**: https://platform.openai.com/api-keys

---

## 3️⃣ **Ingest Data**

### **Option A: Neuro Domain** (Neuroscience) ⭐ Recommended

```bash
# Ingest Neuro data (478 nodes, 501 relationships)
python data_ingestion_neo4j.py \
    --file kg_neuro_neo4j.json \
    --domain neuro \
    --clear-domain neuro \
    --uri $NEO4J_URI \
    --user neo4j \
    --password $NEO4J_PASSWORD
```

**Expected output**:
```
✅ Cleared domain 'neuro' - deleted X nodes
✅ Successfully imported 478 nodes
✅ Created 501 relationships
⏱️  Import completed in X seconds
```

---

### **Option B: UDL Domain** (Universal Design for Learning)

```bash
# Ingest UDL data (if you have it)
python data_ingestion_neo4j.py \
    --file kg_udl_neo4j.json \
    --domain udl \
    --clear-domain udl \
    --uri $NEO4J_URI \
    --user neo4j \
    --password $NEO4J_PASSWORD
```

---

### **Option C: Multi-Domain** (Both UDL + Neuro)

```bash
# Ingest both domains
python data_ingestion_neo4j.py --file kg_udl_neo4j.json --domain udl
python data_ingestion_neo4j.py --file kg_neuro_neo4j.json --domain neuro
```

---

## 4️⃣ **Train Node2Vec** (Optional but Recommended)

Node2Vec enables semantic search for better query results (+60-80% recall improvement).

### **Train for Neuro Domain**:
```bash
python train_node2vec.py neuro
```

**Expected output**:
```
============================================================
Node2Vec Training - Domain: NEURO
============================================================
Extracting graph data from Neo4j (domain: neuro)...
Extracted graph: 478 nodes, 501 edges
Training Node2Vec model...
[Progress bars]
Node2Vec training completed!
Building embeddings index: 478 nodes
Model saved to models/neuro_node2vec

🔍 Node2Vec Similarity Test Results (Domain: neuro)
============================================================
[Test results showing similarity scores]

✅ Training completed successfully!
📁 Model saved to: models/neuro_node2vec
🎯 Ready to use in graph retriever with use_vectors=True
```

**Training time**: ~1-3 minutes (depending on your CPU)

---

### **Train for UDL Domain** (if you ingested UDL data):
```bash
python train_node2vec.py udl
```

---

### **Train for All Domains**:
```bash
python train_node2vec.py all
```

**Note**: If you skip this step, the system will still work but without semantic search capabilities. You can train Node2Vec later at any time.

---

## 5️⃣ **Run the App**

```bash
streamlit run streamlit_app.py
```

**Expected output**:
```
You can now view your Streamlit app in your browser.

Local URL: http://localhost:8501
Network URL: http://192.168.1.X:8501
```

🎉 **Open your browser and start asking questions!**

---

## 6️⃣ **Test the System**

### **Example Queries (Neuro Domain)**:

1. **"Che cos'è la motivazione intrinseca?"**
   - Expected: Definition of intrinsic motivation

2. **"Come la memoria di lavoro supporta l'apprendimento?"**
   - Expected: Relationships between working memory and learning

3. **"Qual è la differenza tra mentalità di crescita e mentalità fissa?"**
   - Expected: Comparison of growth vs fixed mindset

### **Example Queries (UDL Domain)**:

1. **"Ci sono strategie per studenti con ADHD?"**
   - Expected: Pedagogical methodologies for ADHD students

2. **"Quali metodologie funzionano per studenti con autismo?"**
   - Expected: Autism-specific teaching strategies

3. **"Come posso aiutare studenti ipovedenti?"**
   - Expected: Visual impairment support strategies

---

## 🔧 **Troubleshooting**

### **Issue: "Module not found" errors**
```bash
# Reinstall dependencies
pip install -r requirements.txt --upgrade
```

### **Issue: "Neo4j connection failed"**
```bash
# Verify credentials
python -c "from config import config; print(config.neo4j.uri)"

# Test connection
python verify_ingestion.py
```

### **Issue: "No Node2Vec model found"**
```bash
# Train model for your domain
python train_node2vec.py neuro  # or udl
```

### **Issue: "OpenAI API error"**
```bash
# Verify API key
python -c "from config import config; print(config.openai.api_key[:10])"

# Check your OpenAI account has credits:
# https://platform.openai.com/usage
```

---

## 📊 **Verify Installation**

### **Check Neo4j Data**:
```bash
python audit_neuro_graph.py
```

**Expected output**:
```
📊 Neuro Domain Audit Report
============================
Total Nodes: 478
Total Relationships: 501
Unique Labels: 195
Relationship Types: 111
...
```

### **Check Node2Vec Model**:
```bash
python -c "import os; print('Neuro model exists:', os.path.exists('models/neuro_node2vec_model.pkl'))"
```

**Expected output**: `Neuro model exists: True`

---

## 🎯 **Next Steps**

### **1. Explore the App**:
- Try different domains (UDL, Neuro, All)
- Check the pipeline stages
- Review evidence and context
- Export query metrics

### **2. Customize**:
- Add your own data (see `MULTI_DOMAIN_INGESTION_GUIDE.md`)
- Train domain-specific Node2Vec models
- Adjust retrieval parameters in `config.py`

### **3. Deploy** (Optional):
- Deploy to Streamlit Cloud (free)
- Deploy to Hugging Face Spaces
- Deploy to your own server

---

## 📚 **Additional Documentation**

- **Full README**: `README.md` (project overview, architecture)
- **Node2Vec Training**: `NODE2VEC_TRAINING_GUIDE.md` (detailed training guide)
- **Data Ingestion**: `MULTI_DOMAIN_INGESTION_GUIDE.md` (multi-domain setup)
- **Pipeline Analysis**: `PIPELINE_ANALYSIS_AND_RECOMMENDATIONS.md` (architecture deep-dive)

---

## 🆘 **Getting Help**

1. Check the logs in the Streamlit app sidebar
2. Review the troubleshooting section above
3. Consult the full documentation
4. Check GitHub issues: https://github.com/Mono33/graphrag-educational-assistant/issues

---

## ✅ **Quick Checklist**

- [ ] Python 3.9+ installed
- [ ] Virtual environment created
- [ ] Dependencies installed (`pip install -r requirements.txt`)
- [ ] `.env` file configured with Neo4j and OpenAI credentials
- [ ] Data ingested (at least one domain)
- [ ] Node2Vec trained (optional but recommended)
- [ ] App running (`streamlit run streamlit_app.py`)
- [ ] Test query executed successfully

---

**🎉 Congratulations! You're ready to explore GraphRAG Educational Assistant!**

For questions or issues, see the full documentation or create a GitHub issue.

