#!/usr/bin/env python3
"""
Streamlit Demo Interface for GraphRAG Educational Assistant
Same interface as Gradio version with all pipeline stages, evidence, and comparison
"""

import streamlit as st
import asyncio
import logging
import json
import time
import pandas as pd
from typing import Dict, Any, Optional
from dataclasses import asdict
from datetime import datetime
from io import BytesIO

# Import your GraphRAG components
from graph_retriever import EnhancedMultilingualText2Cypher
from llm_chain import EducationalResponseGenerator
from context_builder import EducationalContext
from config import config
from query_metrics import MetricsCalculator, QueryMetrics

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Page configuration
st.set_page_config(
    page_title="🎓 GraphRAG Educational Assistant - Multi-Domain",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .stButton>button {
        width: 100%;
        background-color: #4CAF50;
        color: white;
        font-weight: bold;
    }
    .stTextArea>div>div>textarea {
        font-size: 16px;
    }
    .success-box {
        padding: 1rem;
        border-radius: 0.5rem;
        background-color: #d4edda;
        border: 1px solid #c3e6cb;
        color: #155724;
    }
    .info-box {
        padding: 1rem;
        border-radius: 0.5rem;
        background-color: #d1ecf1;
        border: 1px solid #bee5eb;
        color: #0c5460;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def initialize_system(domain: str = "all", use_vectors: bool = True):
    """Initialize the GraphRAG system components (cached per domain)
    
    Args:
        domain: Domain to initialize for ('udl', 'neuro', 'all')
        use_vectors: Enable Node2Vec semantic search (default: True)
    """
    logger.info(f"Initializing GraphRAG system (domain={domain}, use_vectors={use_vectors})...")
    processor = EnhancedMultilingualText2Cypher(
        use_vectors=use_vectors,
        domain=domain,
        config={'max_nodes': 15, 'max_edges': 30}
    )
    logger.info(f"System initialized successfully! (Node2Vec: {'✅ Enabled' if use_vectors else '❌ Disabled'})")
    return processor

@st.cache_resource
def get_metrics_calculator(mode: str = "hybrid_auto", domain: str = "all"):
    """Initialize metrics calculator (cached per domain)
    
    Args:
        mode: Evaluation mode ('simple', 'hybrid', 'research', 'hybrid_auto')
        domain: Domain filter ('udl', 'neuro', 'all')
    
    Returns:
        MetricsCalculator instance
    """
    # Initialize OpenAI client for hybrid_auto, hybrid, and research modes
    openai_client = None
    if mode in ['hybrid_auto', 'hybrid', 'research']:
        try:
            from openai import OpenAI
            openai_client = OpenAI(api_key=config.openai.api_key)
            logger.info(f"OpenAI client initialized for metrics (mode={mode})")
        except Exception as e:
            logger.warning(f"Could not initialize OpenAI client: {e}. Falling back to 'simple' mode.")
            mode = 'simple'
    
    # Initialize with domain support for automatic Italian→English translation
    calculator = MetricsCalculator(
        mode=mode, 
        domain=domain, 
        openai_client=openai_client,
        relevance_threshold=50.0,  # Context Relevance fallback threshold
        faithfulness_low_threshold=40.0,  # Low faithfulness threshold
        faithfulness_high_threshold=90.0  # High faithfulness threshold
    )
    logger.info(f"MetricsCalculator initialized (mode={mode}, domain={domain})")
    return calculator

def get_generator(domain: str):
    """Get domain-specific response generator"""
    return EducationalResponseGenerator(
        openai_api_key=config.openai.api_key,
        language="italian",
        domain=domain,
        model=config.openai.model  # Uses model from .env (e.g., gpt-4o)
    )

async def process_query_async(query: str, domain: str, processor, generator):
    """Process query through the full pipeline
    
    Args:
        query: Natural language query
        domain: Domain filter ('udl', 'neuro', 'all')
        processor: GraphRAG processor
        generator: Response generator
    """
    try:
        # Get retrieval result (now with domain support)
        result = await processor.process_query_with_retrieval(query, domain=domain)
        
        # Generate response
        educational_context_obj = result.get('educational_context_obj')
        if educational_context_obj:
            response = await generator.generate_response(educational_context_obj, query)
            result['llm_response'] = response
        else:
            result['llm_response'] = {
                'response': "Non sono riuscito a generare una risposta. Riprova con una domanda diversa.",
                'confidence': 'LOW',
                'evidence_used': []
            }
        
        return result
    
    except Exception as e:
        logger.error(f"Error processing query: {e}")
        return {
            'error': str(e),
            'llm_response': {
                'response': f"Errore: {str(e)}",
                'confidence': 'VERY_LOW',
                'evidence_used': []
            }
        }

def format_pipeline_stages(result: Dict) -> str:
    """Format pipeline execution stages for display"""
    stages = []
    
    # Show domain info
    domain = result.get('domain', 'N/A')
    stages.append(f"🏷️ **Domain:** {domain.upper()}\n")
    
    # Stage 1: Text2Cypher
    cypher_result = result.get('cypher_result', {})
    is_valid = cypher_result.get('metadata', {}).get('is_valid', False)
    stages.append(f"### ✅ Stage 1: Text2Cypher (Domain-Aware)")
    stages.append(f"**Status:** {'Valid ✓' if is_valid else 'Invalid ✗'}")
    stages.append(f"**Domain:** {domain}")
    stages.append(f"**Generated Cypher:**")
    stages.append(f"```cypher\n{cypher_result.get('cypher_query', 'N/A')}\n```")
    
    # Stage 2: Graph Retrieval
    retrieval_result = result.get('retrieval_result')
    if retrieval_result:
        metadata = retrieval_result.metadata
        stages.append(f"\n### ✅ Stage 2: Graph Retrieval")
        stages.append(f"- **Graph Nodes:** {metadata.get('graph_count', 0)}")
        stages.append(f"- **Semantic Nodes:** {metadata.get('semantic_count', 0)}")
        stages.append(f"- **Total Nodes:** {metadata.get('total_nodes', 0)}")
        stages.append(f"- **Relationships:** {metadata.get('total_triples', 0)}")
        
        # Timings
        timings = metadata.get('timings', {})
        stages.append(f"\n**⏱️ Execution Times:**")
        stages.append(f"- Graph Traversal: {timings.get('graph_traversal', 0):.3f}s")
        stages.append(f"- Semantic Search: {timings.get('semantic_search', 0):.3f}s")
        stages.append(f"- Fusion: {timings.get('fusion', 0):.3f}s")
        stages.append(f"- **Total: {timings.get('total', 0):.3f}s**")
    
    # Stage 3: Context Building
    educational_context_dict = result.get('educational_context', {})
    if educational_context_dict:
        stages.append(f"\n### ✅ Stage 3: Context Builder")
        stages.append(f"**Confidence:** {educational_context_dict.get('confidence_assessment', 'N/A')}")
        
        primary = educational_context_dict.get('primary_methodologies', [])
        stages.append(f"\n**Primary Methodologies:** {len(primary)}")
        for i, method in enumerate(primary, 1):
            stages.append(f"  {i}. **{method.get('name', 'N/A')}** (confidence: {method.get('confidence', 'N/A')})")
        
        supporting = educational_context_dict.get('supporting_methodologies', [])
        if supporting:
            stages.append(f"\n**Supporting Methodologies:** {len(supporting)}")
            for i, method in enumerate(supporting[:3], 1):
                stages.append(f"  {i}. {method.get('name', 'N/A')}")
    
    # Stage 4: LLM Generation
    llm_response = result.get('llm_response', {})
    stages.append(f"\n### ✅ Stage 4: Response Generation")
    stages.append(f"**Status:** Complete ✓")
    stages.append(f"**Language:** Italian 🇮🇹")
    stages.append(f"**Confidence:** {llm_response.get('confidence', 'N/A')}")
    
    return "\n".join(stages)

def format_evidence(result: Dict):
    """Format evidence data for display"""
    retrieval_result = result.get('retrieval_result')
    
    if not retrieval_result:
        return "No evidence available", pd.DataFrame()
    
    # Create nodes dataframe
    nodes_data = []
    for node in retrieval_result.nodes[:15]:
        nodes_data.append({
            'Name': node.get('name', 'N/A'),
            'Category': node.get('category', 'N/A'),
            'Labels': ', '.join(node.get('labels', [])),
            'Source': node.get('source', 'graph')
        })
    
    nodes_df = pd.DataFrame(nodes_data) if nodes_data else pd.DataFrame()
    
    # Create relationships text
    relationships_text = []
    relationships_text.append("### 🔗 Key Relationships\n")
    for i, (source, rel_type, target) in enumerate(retrieval_result.triples[:10], 1):
        relationships_text.append(f"{i}. **{source}** → `{rel_type}` → **{target}**")
    
    # Facets
    facets_text = []
    facets_text.append("\n### 📊 Knowledge Graph Statistics\n")
    label_counts = retrieval_result.facets.get('label_counts', {})
    if label_counts:
        facets_text.append("**Node Types:**")
        for label, count in sorted(label_counts.items(), key=lambda x: x[1], reverse=True)[:5]:
            facets_text.append(f"- {label}: {count}")
    
    rel_counts = retrieval_result.facets.get('rel_counts', {})
    if rel_counts:
        facets_text.append("\n**Relationship Types:**")
        for rel, count in sorted(rel_counts.items(), key=lambda x: x[1], reverse=True)[:5]:
            facets_text.append(f"- {rel}: {count}")
    
    evidence_text = "\n".join(relationships_text) + "\n" + "\n".join(facets_text)
    
    return evidence_text, nodes_df

def generate_download_content_txt(query: str, response: str, metrics: Dict, domain: str, confidence: str) -> str:
    """Generate TXT format download content"""
    timestamp = datetime.now().strftime("%d %B %Y, %H:%M:%S")
    
    # Map domain codes to full names
    domain_names = {
        "udl": "UDL (Universal Design for Learning)",
        "neuro": "Neuro (Neuroscience)",
        "all": "All Domains (Cross-Domain)"
    }
    domain_full = domain_names.get(domain, domain.upper())
    
    # Confidence emoji
    confidence_emoji = {
        'VERY_HIGH': '🟢',
        'HIGH': '🟢',
        'MEDIUM': '🟡',
        'LOW': '🟠',
        'VERY_LOW': '🔴'
    }
    emoji = confidence_emoji.get(confidence, '⚪')
    
    content = f"""=================================================
GRAPHRAG EDUCATIONAL ASSISTANT - RISPOSTA
=================================================

DATA: {timestamp}
DOMINIO: {domain_full}

-------------------------------------------------
DOMANDA:
-------------------------------------------------
{query}

-------------------------------------------------
RISPOSTA:
-------------------------------------------------
{response}

-------------------------------------------------
METRICHE DI QUALITÀ:
-------------------------------------------------
• Confidence: {emoji} {confidence}
• Context Relevance: {metrics.get('context_relevance', 0):.1f}%
• Answer Faithfulness: {metrics.get('faithfulness', 0):.1f}%
• Query Complexity: {metrics.get('query_complexity', 'N/A')}
• Graph Coverage: {metrics.get('graph_coverage', 0):.1f} hops
• Total Nodes Retrieved: {metrics.get('total_nodes', 0)}
• Total Relationships: {metrics.get('total_relationships', 0)}
• Response Time: {metrics.get('response_time_sec', 0):.2f}s
• Evaluation Mode: {metrics.get('evaluation_mode', 'N/A')}

-------------------------------------------------
Generated by GraphRAG Educational Assistant
https://mono33-graphrag-educational-assistant-fem.streamlit.app
=================================================
"""
    return content

def generate_download_content_markdown(query: str, response: str, metrics: Dict, domain: str, confidence: str) -> str:
    """Generate Markdown format download content"""
    timestamp = datetime.now().strftime("%d %B %Y, %H:%M:%S")
    
    # Map domain codes to full names
    domain_names = {
        "udl": "UDL (Universal Design for Learning)",
        "neuro": "Neuro (Neuroscience)",
        "all": "All Domains (Cross-Domain)"
    }
    domain_full = domain_names.get(domain, domain.upper())
    
    # Confidence emoji
    confidence_emoji = {
        'VERY_HIGH': '🟢',
        'HIGH': '🟢',
        'MEDIUM': '🟡',
        'LOW': '🟠',
        'VERY_LOW': '🔴'
    }
    emoji = confidence_emoji.get(confidence, '⚪')
    
    content = f"""# 🎓 GraphRAG Educational Assistant - Risposta

## 📋 Informazioni

- **Data:** {timestamp}
- **Dominio:** {domain_full}

---

## ❓ Domanda

{query}

---

## 💡 Risposta

{response}

---

## 📊 Metriche di Qualità

| Metrica | Valore |
|---------|--------|
| **Confidence** | {emoji} {confidence} |
| **Context Relevance** | {metrics.get('context_relevance', 0):.1f}% |
| **Answer Faithfulness** | {metrics.get('faithfulness', 0):.1f}% |
| **Query Complexity** | {metrics.get('query_complexity', 'N/A')} |
| **Graph Coverage** | {metrics.get('graph_coverage', 0):.1f} hops |
| **Total Nodes Retrieved** | {metrics.get('total_nodes', 0)} |
| **Total Relationships** | {metrics.get('total_relationships', 0)} |
| **Response Time** | {metrics.get('response_time_sec', 0):.2f}s |
| **Evaluation Mode** | {metrics.get('evaluation_mode', 'N/A')} |

---

*Generated by [GraphRAG Educational Assistant](https://mono33-graphrag-educational-assistant-fem.streamlit.app)*
"""
    return content

def generate_download_content_pdf(query: str, response: str, metrics: Dict, domain: str, confidence: str) -> bytes:
    """Generate PDF format download content using reportlab"""
    try:
        from reportlab.lib.pagesizes import letter, A4
        from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
        from reportlab.lib.units import inch
        from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
        from reportlab.lib import colors
        from reportlab.lib.enums import TA_CENTER, TA_LEFT
        
        buffer = BytesIO()
        doc = SimpleDocTemplate(buffer, pagesize=A4, topMargin=0.5*inch, bottomMargin=0.5*inch)
        
        # Container for elements
        elements = []
        
        # Styles
        styles = getSampleStyleSheet()
        title_style = ParagraphStyle(
            'CustomTitle',
            parent=styles['Heading1'],
            fontSize=18,
            textColor=colors.HexColor('#4CAF50'),
            spaceAfter=12,
            alignment=TA_CENTER
        )
        
        heading_style = ParagraphStyle(
            'CustomHeading',
            parent=styles['Heading2'],
            fontSize=14,
            textColor=colors.HexColor('#2196F3'),
            spaceAfter=10,
            spaceBefore=10
        )
        
        normal_style = styles['BodyText']
        
        timestamp = datetime.now().strftime("%d %B %Y, %H:%M:%S")
        
        # Map domain codes to full names
        domain_names = {
            "udl": "UDL (Universal Design for Learning)",
            "neuro": "Neuro (Neuroscience)",
            "all": "All Domains (Cross-Domain)"
        }
        domain_full = domain_names.get(domain, domain.upper())
        
        # Title
        elements.append(Paragraph("GraphRAG Educational Assistant", title_style))
        elements.append(Paragraph("Risposta Generata", title_style))
        elements.append(Spacer(1, 0.2*inch))
        
        # Metadata
        elements.append(Paragraph(f"<b>Data:</b> {timestamp}", normal_style))
        elements.append(Paragraph(f"<b>Dominio:</b> {domain_full}", normal_style))
        elements.append(Spacer(1, 0.3*inch))
        
        # Question
        elements.append(Paragraph("Domanda", heading_style))
        elements.append(Paragraph(query, normal_style))
        elements.append(Spacer(1, 0.2*inch))
        
        # Response
        elements.append(Paragraph("Risposta", heading_style))
        # Split response into paragraphs for better formatting
        response_paragraphs = response.split('\n\n')
        for para in response_paragraphs:
            if para.strip():
                elements.append(Paragraph(para.replace('\n', '<br/>'), normal_style))
                elements.append(Spacer(1, 0.1*inch))
        
        elements.append(Spacer(1, 0.2*inch))
        
        # Metrics table
        elements.append(Paragraph("Metriche di Qualità", heading_style))
        
        metrics_data = [
            ['Metrica', 'Valore'],
            ['Confidence', f"{confidence}"],
            ['Context Relevance', f"{metrics.get('context_relevance', 0):.1f}%"],
            ['Answer Faithfulness', f"{metrics.get('faithfulness', 0):.1f}%"],
            ['Query Complexity', metrics.get('query_complexity', 'N/A')],
            ['Graph Coverage', f"{metrics.get('graph_coverage', 0):.1f} hops"],
            ['Total Nodes Retrieved', str(metrics.get('total_nodes', 0))],
            ['Total Relationships', str(metrics.get('total_relationships', 0))],
            ['Response Time', f"{metrics.get('response_time_sec', 0):.2f}s"],
            ['Evaluation Mode', metrics.get('evaluation_mode', 'N/A')]
        ]
        
        metrics_table = Table(metrics_data, colWidths=[3*inch, 2*inch])
        metrics_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#4CAF50')),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, 0), 12),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
            ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
            ('GRID', (0, 0), (-1, -1), 1, colors.black)
        ]))
        
        elements.append(metrics_table)
        elements.append(Spacer(1, 0.3*inch))
        
        # Footer
        footer_style = ParagraphStyle(
            'Footer',
            parent=styles['Normal'],
            fontSize=9,
            textColor=colors.grey,
            alignment=TA_CENTER
        )
        elements.append(Paragraph("Generated by GraphRAG Educational Assistant", footer_style))
        elements.append(Paragraph("https://mono33-graphrag-educational-assistant-fem.streamlit.app", footer_style))
        
        # Build PDF
        doc.build(elements)
        
        buffer.seek(0)
        return buffer.getvalue()
        
    except ImportError:
        logger.warning("reportlab not installed, PDF generation unavailable")
        return None
    except Exception as e:
        logger.error(f"Error generating PDF: {e}")
        return None

def generate_filename(query: str, domain: str, format: str) -> str:
    """Generate filename for download"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Create short query slug (first 3-4 words, max 30 chars)
    query_words = query.split()[:4]
    query_slug = "_".join(query_words)[:30].lower()
    # Remove special characters
    query_slug = "".join(c if c.isalnum() or c == "_" else "" for c in query_slug)
    
    # Map domain to short code
    domain_code = domain if domain in ['udl', 'neuro', 'all'] else 'unknown'
    
    filename = f"risposta_{domain_code}_{query_slug}_{timestamp}.{format}"
    return filename

def main():
    """Main Streamlit app"""
    
    # Initialize query_metrics at the very start (before sidebar renders)
    if 'query_metrics' not in st.session_state:
        st.session_state['query_metrics'] = []
    
    # Title and description
    st.title("🎓 GraphRAG Educational Assistant - Multi-Domain")
    st.markdown("""
    **Sistema di supporto pedagogico per insegnanti italiani basato su Knowledge Graph**
    
    Fai domande in italiano su strategie didattiche, neuroscienze dell'apprendimento, e bisogni educativi speciali.
    """)
    
    # Sidebar
    with st.sidebar:
        st.header("📚 Informazioni")
        st.markdown("""
        ### 🔍 Come funziona
        
        1. **Text2Cypher**: Traduce la tua domanda in query Cypher
        2. **Graph Retrieval**: Recupera dati dal knowledge graph (con Node2Vec)
        3. **Context Builder**: Costruisce contesto educativo strutturato
        4. **Response Generator**: Genera risposta in italiano
        
        ### 💡 Esempi di domande
        
        **UDL (Universal Design for Learning):**
        - "Ci sono strategie per studenti ipovedenti?"
        - "Il mio studente ha l'ADHD, cosa posso fare?"
        - "Metodologie per disturbi dello spettro autistico?"
        
        **Neuro (Neuroscience):**
        - "Come la memoria di lavoro supporta l'apprendimento?"
        - "Quali emozioni facilitano la creatività?"
        - "Come migliorare l'attenzione selettiva?"
        
        ### ℹ️ Tecnologie
        
        - **Neo4j** + **Node2Vec** (Knowledge Graph)
        - **OpenAI GPT** (LLM)
        - **LangChain** (Orchestration)
        - **Streamlit** (Interface)
        """)
        
        st.divider()
        
        # Settings
        st.header("⚙️ Impostazioni")
        show_pipeline = st.checkbox("Mostra Pipeline Stages", value=True)
        show_evidence = st.checkbox("Mostra Evidence", value=True)
        show_context = st.checkbox("Mostra Context", value=False)
        
        st.divider()
        
        # Query Metrics Dashboard
        st.header("📊 Metriche Query")
        if 'last_metrics' in st.session_state:
            # Get metrics for CURRENT QUERY ONLY (not averages)
            last_metrics = st.session_state['last_metrics']
            
            # Top row: Confidence & Performance
            st.markdown("### 📈 Confidence & Performance")
            col1, col2 = st.columns(2)
            with col1:
                # Get confidence from last query
                last_confidence = st.session_state.get('last_confidence', 'MEDIUM')
                
                # Map confidence to emoji
                confidence_emoji = {
                    'VERY_HIGH': '🟢',
                    'HIGH': '🟢',
                    'MEDIUM': '🟡',
                    'LOW': '🟠',
                    'VERY_LOW': '🔴'
                }
                emoji = confidence_emoji.get(last_confidence, '⚪')
                
                st.metric(
                    "Confidence", 
                    f"{emoji} {last_confidence}",
                    help="LLM's confidence in the response quality (VERY_HIGH=95%, HIGH=80%, MEDIUM=60%, LOW=40%, VERY_LOW=20%)"
                )
            with col2:
                st.metric(
                    "Response Time", 
                    f"{last_metrics['response_time_sec']:.2f}s",
                    help="Time taken to process THIS query"
                )
            
            # Bottom row: Quality & Intelligence
            st.markdown("### 🎯 Quality Metrics")
            col3, col4 = st.columns(2)
            with col3:
                st.metric(
                    "Context Relevance", 
                    f"{last_metrics['context_relevance']:.1f}%",
                    help="How relevant is retrieved context to THIS query"
                )
                st.metric(
                    "Query Complexity", 
                    last_metrics['query_complexity'],
                    help="Complexity of THIS query (SIMPLE/MEDIUM/COMPLEX)"
                )
            with col4:
                st.metric(
                    "Answer Faithfulness", 
                    f"{last_metrics['faithfulness']:.1f}%",
                    help="Is the LLM response grounded in the context for THIS query?"
                )
                st.metric(
                    "Graph Coverage", 
                    f"{last_metrics['graph_coverage']:.1f} hops",
                    help="Exploration depth in the knowledge graph for THIS query"
                )
            
            # Download metrics (all queries history)
            if 'query_metrics' in st.session_state and len(st.session_state['query_metrics']) > 0:
                st.divider()
                metrics_df = pd.DataFrame(st.session_state['query_metrics'])
                csv = metrics_df.to_csv(index=False).encode('utf-8')
                st.download_button(
                    label="📥 Download All Queries CSV",
                    data=csv,
                    file_name=f"query_metrics_{time.strftime('%Y%m%d_%H%M%S')}.csv",
                    mime="text/csv",
                    use_container_width=True,
                    help="Download metrics for all queries in this session"
                )
        else:
            st.info("Nessuna metrica disponibile. Esegui una query per iniziare il tracking.")
    
    # Domain selector (moved before initialization for domain-specific loading)
    st.header("💬 Fai una Domanda")
    
    domain_options = {
        "UDL (Universal Design for Learning)": "udl",
        "Neuro (Neuroscience)": "neuro",
        "All Domains (Cross-Domain)": "all"
    }
    
    selected_domain_label = st.selectbox(
        "📚 Seleziona il dominio di conoscenza:",
        options=list(domain_options.keys()),
        index=0,  # Default to UDL (first option)
        help="Scegli il dominio specifico per ottenere risposte più accurate"
    )
    
    selected_domain = domain_options[selected_domain_label]
    
    # Initialize system with selected domain (processor is cached per domain)
    with st.spinner(f"🔄 Inizializzazione sistema per dominio: {selected_domain}..."):
        processor = initialize_system(domain=selected_domain, use_vectors=True)
    
    # Show info about selected domain
    domain_info = {
        "udl": "🎯 Focus su strategie didattiche per studenti con bisogni educativi speciali (BES, DSA, disabilità)",
        "neuro": "🧠 Focus su neuroscienze dell'apprendimento (attenzione, memoria, emozioni, funzioni esecutive)",
        "all": "🌐 Ricerca in tutti i domini (può combinare UDL e Neuro)"
    }
    st.info(f"{domain_info[selected_domain]} | **Node2Vec: ✅ Abilitato** (ricerca semantica attiva)")

    
    query = st.text_area(
        "Inserisci la tua domanda in italiano:",
        placeholder="Es: Ci sono strategie per studenti con ADHD?" if selected_domain == "udl" else "Es: Come la memoria di lavoro supporta l'apprendimento?",
        height=100,
        key="query_input"
    )
    
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        submit_button = st.button("🚀 Genera Risposta", type="primary", use_container_width=True)
    
    # Process query
    if submit_button and query.strip():
        start_time = time.time()
        with st.spinner(f"🔄 Elaborazione in corso (Domain: {selected_domain_label})..."):
            # Create domain-specific generator
            generator = get_generator(selected_domain)
            # Run async function with domain parameter
            result = asyncio.run(process_query_async(query, selected_domain, processor, generator))
            
            # Calculate processing time
            elapsed_time = time.time() - start_time
            
            # Track query metrics
            if 'query_metrics' not in st.session_state:
                st.session_state['query_metrics'] = []
            
            # Extract data from result
            cypher_result = result.get('cypher_result', {})
            retrieval_result = result.get('retrieval_result')
            llm_response = result.get('llm_response', {})
            
            # Calculate quality metrics using MetricsCalculator
            # Calculator is domain-aware and automatically translates Italian queries
            # Mode: 'hybrid_auto' = smart fallback (Version A + Version B when needed)
            metrics_calculator = get_metrics_calculator(mode="hybrid_auto", domain=selected_domain)
            
            query_metrics = metrics_calculator.calculate_all(
                query=query,  # Pass original query, calculator handles translation
                retrieved_nodes=retrieval_result.nodes if retrieval_result else [],
                llm_response=llm_response.get('response', ''),
                cypher_query=cypher_result.get('cypher_query', ''),
                total_relationships=retrieval_result.metadata.get('total_triples', 0) if retrieval_result else 0,
                domain=selected_domain  # Pass domain for correct Italian→English translation
            )
            
            # Store metrics in a flat dictionary for DataFrame
            metrics = {
                'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
                'query': query[:100],  # Truncate for display
                'domain': selected_domain,
                'response_time_sec': round(elapsed_time, 2),
                'context_relevance': query_metrics.context_relevance,
                'faithfulness': query_metrics.faithfulness,
                'query_complexity': query_metrics.query_complexity,
                'graph_coverage': query_metrics.graph_coverage,
                'total_nodes': query_metrics.total_nodes,
                'total_relationships': query_metrics.total_relationships,
                'evaluation_mode': query_metrics.evaluation_mode
            }
            st.session_state['query_metrics'].append(metrics)
            
            # Store in session state
            st.session_state['last_result'] = result
            st.session_state['last_query'] = query
            st.session_state['last_domain'] = selected_domain_label
            st.session_state['last_metrics'] = metrics
            st.session_state['last_confidence'] = llm_response.get('confidence', 'MEDIUM')
            
            # Force a rerun to update sidebar metrics immediately
            st.rerun()
    
    # Display results
    if 'last_result' in st.session_state:
        result = st.session_state['last_result']
        
        st.divider()
        
        # Response
        st.header("📝 Risposta")
        
        # Show domain badge
        if 'last_domain' in st.session_state:
            st.caption(f"🏷️ Domain: **{st.session_state['last_domain']}**")
        
        llm_response = result.get('llm_response', {})
        response_text = llm_response.get('response', 'No response generated')
        
        # Display response in a nice box
        st.markdown(f"""
        <div class="success-box">
        {response_text}
        </div>
        """, unsafe_allow_html=True)
        
        # Confidence indicator
        confidence = llm_response.get('confidence', 'MEDIUM')
        confidence_colors = {
            'VERY_HIGH': '🟢',
            'HIGH': '🟢',
            'MEDIUM': '🟡',
            'LOW': '🟠',
            'VERY_LOW': '🔴'
        }
        st.caption(f"Confidence: {confidence_colors.get(confidence, '⚪')} {confidence}")
        
        # Download buttons in expander (collapsed by default for clean UI)
        with st.expander("📥 Download Risposta", expanded=False):
            st.markdown("Scarica la risposta in diversi formati:")
            
            # Get data for download
            query_text = st.session_state.get('last_query', '')
            metrics = st.session_state.get('last_metrics', {})
            domain = st.session_state.get('last_result', {}).get('domain', 'unknown')
            
            # Create three columns for download buttons
            col1, col2, col3 = st.columns(3)
            
            with col1:
                # TXT Download
                txt_content = generate_download_content_txt(
                    query=query_text,
                    response=response_text,
                    metrics=metrics,
                    domain=domain,
                    confidence=confidence
                )
                txt_filename = generate_filename(query_text, domain, 'txt')
                st.download_button(
                    label="📄 Download as TXT",
                    data=txt_content,
                    file_name=txt_filename,
                    mime="text/plain",
                    use_container_width=True,
                    help="Download as plain text file"
                )
            
            with col2:
                # Markdown Download
                md_content = generate_download_content_markdown(
                    query=query_text,
                    response=response_text,
                    metrics=metrics,
                    domain=domain,
                    confidence=confidence
                )
                md_filename = generate_filename(query_text, domain, 'md')
                st.download_button(
                    label="📝 Download as Markdown",
                    data=md_content,
                    file_name=md_filename,
                    mime="text/markdown",
                    use_container_width=True,
                    help="Download as Markdown file (formatted)"
                )
            
            with col3:
                # PDF Download
                pdf_content = generate_download_content_pdf(
                    query=query_text,
                    response=response_text,
                    metrics=metrics,
                    domain=domain,
                    confidence=confidence
                )
                
                if pdf_content:
                    pdf_filename = generate_filename(query_text, domain, 'pdf')
                    st.download_button(
                        label="📕 Download as PDF",
                        data=pdf_content,
                        file_name=pdf_filename,
                        mime="application/pdf",
                        use_container_width=True,
                        help="Download as PDF file (professional format)"
                    )
                else:
                    st.button(
                        "📕 PDF (Not Available)",
                        disabled=True,
                        use_container_width=True,
                        help="Install reportlab library to enable PDF export"
                    )
            
            # Add helpful info
            st.caption("💡 I file includono: domanda, risposta completa, metriche di qualità, e timestamp.")
        
        st.markdown("---")
        
        # Tabs for detailed information
        tab1, tab2, tab3, tab4 = st.tabs([
            "🔬 Pipeline Stages",
            "📊 Evidence & Data",
            "🧩 Educational Context",
            "📈 Comparison"
        ])
        
        with tab1:
            if show_pipeline:
                st.markdown("### 🔬 Pipeline Execution Stages")
                pipeline_info = format_pipeline_stages(result)
                st.markdown(pipeline_info)
            else:
                st.info("Enable 'Mostra Pipeline Stages' in sidebar to see details")
        
        with tab2:
            if show_evidence:
                st.markdown("### 📊 Evidence from Knowledge Graph")
                evidence_text, nodes_df = format_evidence(result)
                
                st.markdown(evidence_text)
                
                if not nodes_df.empty:
                    st.markdown("\n### 📋 Retrieved Nodes")
                    st.dataframe(nodes_df, use_container_width=True)
            else:
                st.info("Enable 'Mostra Evidence' in sidebar to see details")
        
        with tab3:
            if show_context:
                st.markdown("### 🧩 Educational Context (Structured)")
                educational_context = result.get('educational_context', {})
                
                if educational_context:
                    # Student Profile
                    st.markdown("#### 👤 Student Profile")
                    profile = educational_context.get('student_profile', {})
                    col1, col2 = st.columns(2)
                    with col1:
                        st.write("**Primary Needs:**", ', '.join(profile.get('primary_needs', [])))
                    with col2:
                        st.write("**Educational Context:**", profile.get('educational_context', 'general'))
                    
                    # Methodologies
                    st.markdown("#### 📚 Recommended Methodologies")
                    primary = educational_context.get('primary_methodologies', [])
                    for i, method in enumerate(primary, 1):
                        with st.expander(f"{i}. {method.get('name', 'N/A')} - {method.get('confidence', 'N/A')}"):
                            st.write("**Category:**", method.get('category', 'N/A'))
                            st.write("**Implementation:**", method.get('implementation_guidance', 'N/A'))
                            st.write("**Applications:**")
                            for app in method.get('classroom_applications', []):
                                st.write(f"- {app}")
                    
                    # Evidence
                    st.markdown("#### 🔬 Evidence Summary")
                    st.write(educational_context.get('evidence_summary', 'N/A'))
                    
                    # Implementation Priority
                    st.markdown("#### 🎯 Implementation Priority")
                    for i, priority in enumerate(educational_context.get('implementation_priority', []), 1):
                        st.write(f"{i}. {priority}")
                
                else:
                    st.warning("No educational context available")
            else:
                st.info("Enable 'Mostra Context' in sidebar to see details")
        
        with tab4:
            st.markdown("### 📈 Comparison: Hybrid vs Graph-Only")
            
            retrieval_result = result.get('retrieval_result')
            if retrieval_result:
                metadata = retrieval_result.metadata
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.metric(
                        "Graph Nodes (Direct)",
                        metadata.get('graph_count', 0),
                        help="Nodes from direct graph traversal"
                    )
                    st.metric(
                        "Total Nodes",
                        metadata.get('total_nodes', 0),
                        help="Total nodes after hybrid retrieval"
                    )
                
                with col2:
                    st.metric(
                        "Semantic Nodes (Node2Vec)",
                        metadata.get('semantic_count', 0),
                        help="Additional nodes from semantic search"
                    )
                    st.metric(
                        "Relationships",
                        metadata.get('total_triples', 0),
                        help="Total relationships retrieved"
                    )
                
                st.markdown("""
                **💡 Hybrid Retrieval Benefits:**
                - 🎯 **Precision**: Direct graph relationships (exact matches)
                - 🔍 **Breadth**: Node2Vec semantic similarity (related concepts)
                - 🚀 **Coverage**: Neighbor expansion (contextual information)
                """)
            else:
                st.warning("No comparison data available")

if __name__ == "__main__":
    main()

