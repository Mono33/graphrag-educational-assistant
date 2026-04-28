#!/usr/bin/env python3
"""
Streamlit Demo Interface for GraphRAG Educational Assistant
Same interface as Gradio version with all pipeline stages, evidence, and comparison.

Run from the repo root:
    streamlit run apps/streamlit/main.py
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
from aix.retrieval.graph_retriever import EnhancedMultilingualText2Cypher
from aix.generation.llm_chain import EducationalResponseGenerator
from aix.retrieval.context_builder import EducationalContext
from aix.core.config import config
from aix.retrieval.query_metrics import MetricsCalculator, QueryMetrics

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
            openai_client = config.openai.get_client()
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


# ═══════════════════════════════════════════════════════════════════════════════
# AGENT MODE DOWNLOAD FUNCTIONS
# ═══════════════════════════════════════════════════════════════════════════════

def generate_agent_download_txt(query: str, response: str, result, domain: str, query_intent: str) -> str:
    """Generate TXT format download content for Agent mode"""
    timestamp = datetime.now().strftime("%d %B %Y, %H:%M:%S")
    
    domain_names = {
        "udl": "UDL (Universal Design for Learning)",
        "neuro": "Neuro (Neuroscience)"
    }
    domain_full = domain_names.get(domain, domain.upper())
    
    intent_names = {
        "lesson_creation": "Piano di Lezione",
        "activity_design": "Attività Didattica",
        "definition": "Definizione",
        "comparison": "Confronto",
        "explanation": "Spiegazione",
        "recommendation": "Raccomandazioni",
        "list": "Elenco"
    }
    intent_label = intent_names.get(query_intent, query_intent)
    
    # Build scores text
    scores_text = ""
    if result.scores:
        scores_text = "\n".join([f"  • {k.title()}: {v}/5" for k, v in result.scores.items()])
    else:
        scores_text = "  Non disponibili"
    
    content = f"""=================================================
AGENTIC GRAPHRAG - {intent_label.upper()}
=================================================

DATA: {timestamp}
DOMINIO: {domain_full}
TIPO: {intent_label}

-------------------------------------------------
RICHIESTA:
-------------------------------------------------
{query}

-------------------------------------------------
RISPOSTA:
-------------------------------------------------
{response}

-------------------------------------------------
METRICHE DI QUALITÀ:
-------------------------------------------------
• Stato: {"✅ Approvato" if result.approved else "❌ Non Approvato"}
• Revisioni effettuate: {result.revision_count}
• Nodi Knowledge Graph utilizzati: {result.nodes_used}
• Raccomandazioni integrate: {result.recommendations_used}

PUNTEGGI CRITIC AGENT:
{scores_text}

-------------------------------------------------
Generated by Agentic GraphRAG Educational Assistant
=================================================
"""
    return content


def generate_agent_download_markdown(query: str, response: str, result, domain: str, query_intent: str) -> str:
    """Generate Markdown format download content for Agent mode"""
    timestamp = datetime.now().strftime("%d %B %Y, %H:%M:%S")
    
    domain_names = {
        "udl": "UDL (Universal Design for Learning)",
        "neuro": "Neuro (Neuroscience)"
    }
    domain_full = domain_names.get(domain, domain.upper())
    
    intent_names = {
        "lesson_creation": "Piano di Lezione",
        "activity_design": "Attività Didattica",
        "definition": "Definizione",
        "comparison": "Confronto",
        "explanation": "Spiegazione",
        "recommendation": "Raccomandazioni",
        "list": "Elenco"
    }
    intent_label = intent_names.get(query_intent, query_intent)
    
    # Build scores table
    scores_table = ""
    if result.scores:
        scores_table = "| Criterio | Punteggio |\n|----------|----------|\n"
        for k, v in result.scores.items():
            scores_table += f"| **{k.title()}** | {v}/5 |\n"
    else:
        scores_table = "*Punteggi non disponibili*"
    
    content = f"""# 🤖 Agentic GraphRAG - {intent_label}

## 📋 Informazioni

- **Data:** {timestamp}
- **Dominio:** {domain_full}
- **Tipo di contenuto:** {intent_label}
- **Stato:** {"✅ Approvato" if result.approved else "❌ Non Approvato"}

---

## ❓ Richiesta Originale

{query}

---

## 📝 Risposta Generata

{response}

---

## 📊 Metriche di Qualità

| Metrica | Valore |
|---------|--------|
| **Stato** | {"✅ Approvato" if result.approved else "❌ Non Approvato"} |
| **Revisioni** | {result.revision_count} |
| **Nodi KG utilizzati** | {result.nodes_used} |
| **Raccomandazioni** | {result.recommendations_used} |

### Punteggi Critic Agent

{scores_table}

---

*Generato da [Agentic GraphRAG Educational Assistant](https://teamaifem33.streamlit.app)*
"""
    return content


def generate_agent_download_pdf(query: str, response: str, result, domain: str, query_intent: str) -> bytes:
    """Generate PDF format download content for Agent mode using reportlab"""
    try:
        from reportlab.lib.pagesizes import A4
        from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
        from reportlab.lib.units import inch
        from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
        from reportlab.lib import colors
        from reportlab.lib.enums import TA_CENTER, TA_LEFT
        
        buffer = BytesIO()
        doc = SimpleDocTemplate(buffer, pagesize=A4, topMargin=0.5*inch, bottomMargin=0.5*inch)
        
        elements = []
        styles = getSampleStyleSheet()
        
        title_style = ParagraphStyle(
            'AgentTitle',
            parent=styles['Heading1'],
            fontSize=18,
            textColor=colors.HexColor('#9C27B0'),  # Purple for Agent mode
            spaceAfter=12,
            alignment=TA_CENTER
        )
        
        heading_style = ParagraphStyle(
            'AgentHeading',
            parent=styles['Heading2'],
            fontSize=14,
            textColor=colors.HexColor('#673AB7'),
            spaceAfter=10,
            spaceBefore=10
        )
        
        normal_style = styles['BodyText']
        
        timestamp = datetime.now().strftime("%d %B %Y, %H:%M:%S")
        
        domain_names = {
            "udl": "UDL (Universal Design for Learning)",
            "neuro": "Neuro (Neuroscience)"
        }
        domain_full = domain_names.get(domain, domain.upper())
        
        intent_names = {
            "lesson_creation": "Piano di Lezione",
            "activity_design": "Attività Didattica",
            "definition": "Definizione",
            "comparison": "Confronto",
            "explanation": "Spiegazione",
            "recommendation": "Raccomandazioni",
            "list": "Elenco"
        }
        intent_label = intent_names.get(query_intent, query_intent)
        
        # Title
        elements.append(Paragraph("🤖 Agentic GraphRAG", title_style))
        elements.append(Paragraph(intent_label, title_style))
        elements.append(Spacer(1, 0.2*inch))
        
        # Metadata
        elements.append(Paragraph(f"<b>Data:</b> {timestamp}", normal_style))
        elements.append(Paragraph(f"<b>Dominio:</b> {domain_full}", normal_style))
        status = "✅ Approvato" if result.approved else "❌ Non Approvato"
        elements.append(Paragraph(f"<b>Stato:</b> {status}", normal_style))
        elements.append(Spacer(1, 0.2*inch))
        
        # Query
        elements.append(Paragraph("Richiesta", heading_style))
        # Handle special characters
        safe_query = query.replace('&', '&amp;').replace('<', '&lt;').replace('>', '&gt;')
        elements.append(Paragraph(safe_query, normal_style))
        elements.append(Spacer(1, 0.2*inch))
        
        # Response
        elements.append(Paragraph("Risposta", heading_style))
        
        # Split response into paragraphs and add them
        safe_response = response.replace('&', '&amp;').replace('<', '&lt;').replace('>', '&gt;')
        for para in safe_response.split('\n\n'):
            if para.strip():
                # Handle markdown headers
                if para.startswith('#'):
                    para = para.lstrip('#').strip()
                    elements.append(Paragraph(f"<b>{para}</b>", normal_style))
                else:
                    elements.append(Paragraph(para.replace('\n', '<br/>'), normal_style))
                elements.append(Spacer(1, 0.1*inch))
        
        elements.append(Spacer(1, 0.2*inch))
        
        # Metrics
        elements.append(Paragraph("Metriche di Qualità", heading_style))
        
        metrics_data = [
            ["Metrica", "Valore"],
            ["Stato", "Approvato" if result.approved else "Non Approvato"],
            ["Revisioni", str(result.revision_count)],
            ["Nodi KG utilizzati", str(result.nodes_used)],
            ["Raccomandazioni", str(result.recommendations_used)]
        ]
        
        # Add scores if available
        if result.scores:
            for k, v in result.scores.items():
                metrics_data.append([k.title(), f"{v}/5"])
        
        table = Table(metrics_data, colWidths=[3*inch, 2*inch])
        table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#9C27B0')),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, 0), 12),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
            ('BACKGROUND', (0, 1), (-1, -1), colors.HexColor('#F3E5F5')),
            ('TEXTCOLOR', (0, 1), (-1, -1), colors.black),
            ('FONTNAME', (0, 1), (-1, -1), 'Helvetica'),
            ('FONTSIZE', (0, 1), (-1, -1), 10),
            ('GRID', (0, 0), (-1, -1), 1, colors.HexColor('#CE93D8')),
            ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
        ]))
        elements.append(table)
        
        elements.append(Spacer(1, 0.3*inch))
        
        # Footer
        footer_style = ParagraphStyle(
            'Footer',
            parent=styles['Normal'],
            fontSize=8,
            textColor=colors.gray,
            alignment=TA_CENTER
        )
        elements.append(Paragraph("Generated by Agentic GraphRAG Educational Assistant", footer_style))
        
        doc.build(elements)
        buffer.seek(0)
        return buffer.getvalue()
        
    except ImportError:
        logger.warning("reportlab not installed, PDF generation unavailable")
        return None
    except Exception as e:
        logger.error(f"Error generating Agent PDF: {e}")
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

def render_agent_mode():
    """
    Render the Agent Mode UI - Lesson Plan Generator
    
    This function provides a complete UI for generating lesson plans
    using the multi-agent pipeline: Planner → Retriever → Writer → Critic

    Status: this Streamlit-based Agent Mode is **retired** as the canonical
    end-to-end test surface for the agent pipeline. It is superseded by the
    Path C webui (FastAPI + htmx 2 + WebAwesome + Tailwind + sse-starlette)
    served at ``/webui/`` from the same uvicorn process — see
    ``docs/product/ClickUp_Agentic_GraphRAG_Update.md`` §6.6.

    The Streamlit page is kept here only for backwards compatibility with
    existing teacher-side flows; **no new agent features land here**
    (tool approval, multi-turn chat, file upload, conversation memory all
    live on /webui/). The GraphRAG admin mode in this same Streamlit app
    is unaffected by this retirement and remains the read-only KG inspector.
    """
    # Retirement banner: shown only on Agent Mode (the GraphRAG admin mode
    # in this same app is intentionally untouched — it remains the
    # read-only knowledge-graph inspector for the team).
    st.warning(
        "🚧 **Questa interfaccia di Agent Mode è in fase di pensionamento.**\n\n"
        "L'interfaccia ufficiale per generare piani di lezione con la pipeline "
        "multi-agente (Planner → Retriever → Writer → Critic) è ora la nuova "
        "**chat workspace su `/webui/`**, che offre:\n"
        "- Profilo educativo persistente (BES, classe, mobilità, dispositivi) e modifica inline\n"
        "- Streaming live dei singoli step dell'agente come chat card\n"
        "- Pannello risorse multimediali curate (video, articoli, OER)\n"
        "- Replay del piano finale su reload pagina\n\n"
        "👉 **Apri la nuova interfaccia:** "
        "[http://127.0.0.1:8765/webui/](http://127.0.0.1:8765/webui/) "
        "*(richiede l'API FastAPI attiva: "
        "`python -m uvicorn aix.api.main:app --host 127.0.0.1 --port 8765`)*\n\n"
        "Le nuove feature agente (tool approval, file upload, chat multi-turno, "
        "conversation memory) saranno disponibili **solo** su `/webui/`.\n\n"
        "ℹ️ La modalità **GraphRAG** in questa app *non è interessata* da questa "
        "migrazione e rimane lo strumento ufficiale di ispezione admin read-only "
        "del Knowledge Graph."
    )
    st.divider()

    st.header("🎓 Agent Mode: Lesson Plan Generator")
    
    st.markdown("""
    **Genera piani di lezione completi** basati sul Knowledge Graph.
    
    Il sistema utilizza una pipeline multi-agente:
    1. **Planner**: Analizza la tua richiesta e crea un piano di ricerca
    2. **Retriever**: Recupera conoscenze dal Knowledge Graph
    3. **Writer**: Genera il piano di lezione strutturato
    4. **Critic**: Valuta la qualità e richiede revisioni se necessario
    """)
    
    st.divider()
    
    # Domain and Language selectors for Agent mode
    col1, col2 = st.columns(2)
    with col1:
        agent_domain = st.selectbox(
            "📚 Dominio:",
            ["neuro", "udl"],
            index=0,
            format_func=lambda x: "Neuro (Neuroscience)" if x == "neuro" else "UDL (Universal Design)",
            key="agent_domain"
        )
    with col2:
        agent_language = st.selectbox(
            "🌍 Lingua output:",
            ["it", "en"],
            index=0,
            format_func=lambda x: "Italiano 🇮🇹" if x == "it" else "English 🇬🇧",
            key="agent_language"
        )
    
    # Query input
    st.markdown("### 📝 Descrivi la lezione che vuoi creare")
    
    agent_query = st.text_area(
        "La tua richiesta:",
        placeholder="Es: Crea una lezione di 45 minuti sulla metacognizione per studenti delle scuole superiori",
        height=120,
        key="agent_query_input",
        help="Descrivi in dettaglio cosa vuoi: argomento, livello scolastico, durata, obiettivi specifici..."
    )
    
    # Example queries
    with st.expander("💡 Esempi di richieste", expanded=False):
        st.markdown("""
        **Lezioni complete:**
        - "Crea una lezione di 45 minuti sulla metacognizione per studenti delle scuole superiori"
        - "Progetta un'attività didattica sul growth mindset per bambini della scuola primaria"
        
        **Strategie specifiche:**
        - "Come insegnare le strategie di memoria di lavoro agli studenti con difficoltà di apprendimento?"
        - "Progetta un'unità didattica sull'attenzione e le funzioni esecutive per adolescenti con ADHD"
        
        **Quiz e valutazioni:**
        - "Crea un quiz formativo sulla neuroplasticità per studenti universitari"
        - "Progetta una verifica sulla regolazione emotiva con domande a risposta aperta"
        """)
    
    # Generate button
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        generate_button = st.button(
            "🚀 Genera Piano di Lezione",
            type="primary",
            use_container_width=True,
            key="agent_generate_btn"
        )
    
    # ═══════════════════════════════════════════════════════════════════════════
    # PHASE 3: Handle upsell query (user clicked "Crea una lezione" button)
    # ═══════════════════════════════════════════════════════════════════════════
    upsell_query = st.session_state.get('agent_upsell_query')
    if upsell_query:
        # Show info about the conversion
        upsell_type = st.session_state.get('agent_upsell_type', 'lesson')
        type_emoji = "📚" if upsell_type == 'lesson' else "🎯"
        type_label = "lezione completa" if upsell_type == 'lesson' else "attività pratica"
        
        st.info(f"{type_emoji} **Conversione in corso!** Sto generando una {type_label} basata sulla tua domanda precedente...")
        
        # Clear the upsell query from session state
        del st.session_state['agent_upsell_query']
        if 'agent_upsell_type' in st.session_state:
            del st.session_state['agent_upsell_type']
        
        # Process the upsell query
        _process_agent_query(upsell_query, agent_domain, agent_language)
    # ═══════════════════════════════════════════════════════════════════════════
    
    # Process query (manual input)
    if generate_button and agent_query.strip():
        _process_agent_query(agent_query, agent_domain, agent_language)
    
    # Display results if available
    if 'agent_result' in st.session_state and st.session_state['agent_result']:
        _display_agent_results()


def _process_agent_query(query: str, domain: str, language: str):
    """Process query through the Agent pipeline"""
    import asyncio
    
    # Show progress
    progress_container = st.container()
    
    with progress_container:
        st.markdown("### ⏳ Generazione in corso...")
        
        # Progress steps
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        try:
            # Import the orchestrator
            from aix.agent.orchestrator import AgentOrchestrator
            
            # Step 1: Initialize
            status_text.markdown("🔧 **Inizializzazione agenti...**")
            progress_bar.progress(10)
            
            # Domain and language are passed to constructor, not to create_lesson_plan
            orchestrator = AgentOrchestrator(
                domain=domain,
                language=language,
                max_revisions=2
            )
            
            # Step 2: Planning
            status_text.markdown("📋 **Planner Agent**: Analisi della richiesta...")
            progress_bar.progress(25)
            
            # Step 3: Execute pipeline (async)
            status_text.markdown("🔍 **Retriever Agent**: Ricerca nel Knowledge Graph...")
            progress_bar.progress(40)
            
            # Run the async orchestrator
            result = asyncio.run(
                orchestrator.create_lesson_plan(query=query)
            )
            
            # Step 4: Writing
            status_text.markdown("✍️ **Writer Agent**: Generazione piano di lezione...")
            progress_bar.progress(70)
            
            # Step 5: Critique
            status_text.markdown("🔬 **Critic Agent**: Valutazione qualità...")
            progress_bar.progress(90)
            
            # Complete
            progress_bar.progress(100)
            
            if result.success:
                status_text.markdown("✅ **Completato!** Piano di lezione generato con successo.")
            else:
                status_text.markdown(f"⚠️ **Completato con avvisi**: {result.error or 'Vedi dettagli sotto'}")
            
            # Store result in session state
            st.session_state['agent_result'] = result
            st.session_state['agent_query'] = query
            st.session_state['agent_result_domain'] = domain  # Store domain for download (different key to avoid widget conflict)
            
            # Force rerun to display results
            time.sleep(0.5)
            st.rerun()
            
        except Exception as e:
            progress_bar.progress(100)
            status_text.markdown(f"❌ **Errore**: {str(e)}")
            logger.error(f"Agent pipeline error: {e}")
            import traceback
            logger.error(traceback.format_exc())
            st.error(f"Si è verificato un errore: {str(e)}")


def _is_lesson_intent(intent: str) -> bool:
    """Check if the intent is a lesson-type (not eligible for upsell)"""
    lesson_intents = {"lesson_creation", "activity_design", "assessment", "unit_plan"}
    return intent in lesson_intents


def _get_content_type_label(intent: str) -> str:
    """Get Italian label for the content type based on intent"""
    labels = {
        "definition": "Definizione",
        "comparison": "Confronto",
        "explanation": "Spiegazione",
        "recommendation": "Raccomandazioni",
        "list": "Elenco",
        "lesson_creation": "Piano di Lezione",
        "activity_design": "Attività"
    }
    return labels.get(intent, "Contenuto")


def _display_agent_results():
    """Display Agent mode results"""
    result = st.session_state.get('agent_result')
    query = st.session_state.get('agent_query', '')
    
    if not result:
        return
    
    st.divider()
    
    # Get intent for display and upsell logic
    query_intent = getattr(result, 'query_intent', 'lesson_creation') or 'lesson_creation'
    key_concepts = getattr(result, 'key_concepts', []) or []
    content_type_label = _get_content_type_label(query_intent)
    
    # NEW Phase A: Get scope status for hybrid mode indicator
    scope_status = getattr(result, 'scope_status', 'in_scope') or 'in_scope'
    is_hybrid = getattr(result, 'is_hybrid', False)
    
    # Results header with status - adapt title based on intent
    col1, col2, col3, col4 = st.columns([2, 1, 1, 1])
    with col1:
        if _is_lesson_intent(query_intent):
            st.markdown("## 📄 Piano di Lezione Generato")
        else:
            st.markdown(f"## 📄 {content_type_label} Generata")
    with col2:
        if result.approved:
            st.success("✅ Approvato")
        else:
            st.warning("⚠️ Non approvato")
    with col3:
        st.metric("Revisioni", result.revision_count)
    with col4:
        # NEW Phase A: Scope status indicator
        if scope_status == "in_scope":
            st.success("✅ In-Scope")
        elif scope_status == "partial_scope":
            st.warning("⚠️ Ibrido")
        elif scope_status == "out_of_scope":
            st.info("🌐 Esterno")
    
    # NEW Phase A: Hybrid mode disclaimer
    if is_hybrid:
        st.info(
            "🌐 **Modalità Ibrida**: Il contenuto disciplinare proviene da fonti esterne "
            "(Wikipedia, pubblicazioni accademiche). Le strategie pedagogiche sono basate "
            "sul Knowledge Graph FEM."
        )
    
    # Tabs for different views
    tab1, tab2, tab3 = st.tabs(["📝 Piano di Lezione", "📊 Metriche", "🔍 Dettagli"])
    
    with tab1:
        if result.lesson_plan:
            # Display the lesson plan in markdown
            st.markdown(result.lesson_plan, unsafe_allow_html=True)
            
            # ═══════════════════════════════════════════════════════════════════════
            # PHASE 3: UPSELL BUTTONS - Show for non-lesson intents
            # ═══════════════════════════════════════════════════════════════════════
            if not _is_lesson_intent(query_intent) and key_concepts:
                st.divider()
                st.markdown("### 💡 Vuoi approfondire?")
                st.caption("Trasforma questa risposta in materiale didattico strutturato:")
                
                upsell_col1, upsell_col2 = st.columns(2)
                
                # Generate concept string for the new query
                concepts_str = ", ".join(key_concepts[:3])  # Limit to first 3 concepts
                
                with upsell_col1:
                    if st.button(
                        "📚 Crea una lezione completa",
                        use_container_width=True,
                        key="upsell_lesson_btn",
                        help=f"Genera un piano di lezione completo su: {concepts_str}"
                    ):
                        # Store the new query and trigger re-processing
                        new_query = f"Crea una lezione su {concepts_str}"
                        st.session_state['agent_upsell_query'] = new_query
                        st.session_state['agent_upsell_type'] = 'lesson'
                        # Clear previous result to show loading
                        del st.session_state['agent_result']
                        st.rerun()
                
                with upsell_col2:
                    if st.button(
                        "🎯 Crea un'attività pratica",
                        use_container_width=True,
                        key="upsell_activity_btn",
                        help=f"Genera un'attività didattica su: {concepts_str}"
                    ):
                        new_query = f"Progetta un'attività pratica su {concepts_str}"
                        st.session_state['agent_upsell_query'] = new_query
                        st.session_state['agent_upsell_type'] = 'activity'
                        del st.session_state['agent_result']
                        st.rerun()
            # ═══════════════════════════════════════════════════════════════════════
            
            # ═══════════════════════════════════════════════════════════════════════
            # PHASE 3 (MEDIA): MEDIA ENHANCEMENT BUTTONS
            # ═══════════════════════════════════════════════════════════════════════
            curated_media = getattr(result, 'curated_media', None)
            has_media = curated_media and any(curated_media.values())
            
            if has_media:
                st.divider()
                st.markdown("### 🎨 Arricchisci con Media")
                st.caption("Esplora contenuti multimediali per questa risposta:")
                
                media_col1, media_col2, media_col3, media_col4 = st.columns(4)
                
                with media_col1:
                    videos = curated_media.get('videos', [])
                    video_count = len(videos)
                    if st.button(
                        f"🎥 Video ({video_count})",
                        use_container_width=True,
                        key="media_video_btn",
                        help="Mostra video educativi suggeriti",
                        disabled=video_count == 0
                    ):
                        st.session_state['show_media_videos'] = True
                        st.session_state['show_media_resources'] = False
                        st.session_state['show_media_citations'] = False
                        st.session_state['show_media_textbooks'] = False
                
                with media_col2:
                    resources = curated_media.get('resources', [])
                    resource_count = len(resources)
                    if st.button(
                        f"🔗 Risorse ({resource_count})",
                        use_container_width=True,
                        key="media_resources_btn",
                        help="Mostra risorse educative (Wikipedia, etc.)",
                        disabled=resource_count == 0
                    ):
                        st.session_state['show_media_videos'] = False
                        st.session_state['show_media_resources'] = True
                        st.session_state['show_media_citations'] = False
                        st.session_state['show_media_textbooks'] = False
                
                with media_col3:
                    citations = curated_media.get('citations', [])
                    citation_count = len(citations)
                    if st.button(
                        f"📖 Papers ({citation_count})",
                        use_container_width=True,
                        key="media_citations_btn",
                        help="Mostra riferimenti scientifici",
                        disabled=citation_count == 0
                    ):
                        st.session_state['show_media_videos'] = False
                        st.session_state['show_media_resources'] = False
                        st.session_state['show_media_citations'] = True
                        st.session_state['show_media_textbooks'] = False
                
                with media_col4:
                    textbooks = curated_media.get('open_textbooks', [])
                    textbook_count = len(textbooks)
                    if st.button(
                        f"📚 Libri OER ({textbook_count})",
                        use_container_width=True,
                        key="media_textbooks_btn",
                        help="Mostra libri di testo aperti (OpenStax, DOAB, etc.)",
                        disabled=textbook_count == 0
                    ):
                        st.session_state['show_media_videos'] = False
                        st.session_state['show_media_resources'] = False
                        st.session_state['show_media_citations'] = False
                        st.session_state['show_media_textbooks'] = True
                
                # Display selected media content
                if st.session_state.get('show_media_videos') and videos:
                    st.markdown("#### 🎥 Video Educativi Trovati")
                    for i, v in enumerate(videos, 1):
                        title = v.get('title', 'Video')
                        url = v.get('url')
                        duration = v.get('duration_hint', '')
                        search_query = v.get('search_query', '')
                        
                        if url:
                            st.markdown(f"**{i}. [{title}]({url})** {f'({duration})' if duration else ''}")
                            st.caption(f"   🔗 {url}")
                        else:
                            st.markdown(f"**{i}. {title}** {f'({duration})' if duration else ''}")
                            st.caption(f"   🔍 Cerca su YouTube: \"{search_query}\"")
                
                if st.session_state.get('show_media_resources') and resources:
                    st.markdown("#### 🔗 Risorse Educative")
                    for i, r in enumerate(resources, 1):
                        title = r.get('title', 'Resource')
                        url = r.get('url') or r.get('suggested_url')
                        res_type = r.get('type', 'educational').title()
                        
                        if url:
                            st.markdown(f"**{i}. [{title}]({url})** ({res_type})")
                        else:
                            st.markdown(f"**{i}. {title}** ({res_type})")
                
                if st.session_state.get('show_media_citations') and citations:
                    st.markdown("#### 📖 Riferimenti Scientifici")
                    for i, c in enumerate(citations, 1):
                        authors = c.get('authors', [])
                        authors_str = ', '.join(authors[:2])
                        if len(authors) > 2:
                            authors_str += ' et al.'
                        year = c.get('year', '')
                        title = c.get('title', '')
                        journal = c.get('journal', '')
                        doi = c.get('doi')
                        
                        citation_text = f"**{i}. {authors_str}**"
                        if year:
                            citation_text += f" ({year})"
                        citation_text += f". *{title}*"
                        if journal:
                            citation_text += f". {journal}"
                        
                        st.markdown(citation_text)
                        if doi:
                            st.caption(f"   DOI: [{doi}](https://doi.org/{doi})")
                
                if st.session_state.get('show_media_textbooks') and textbooks:
                    st.markdown("#### 📚 Libri di Testo Aperti (OER)")
                    st.caption("Risorse educative aperte - copyright-safe (CC BY)")
                    for i, t in enumerate(textbooks, 1):
                        title = t.get('title', 'Textbook')
                        source = t.get('source', 'Unknown')
                        chapter = t.get('chapter', '')
                        url = t.get('url')
                        license_type = t.get('license', 'CC BY 4.0')
                        relevance = t.get('relevance', '')
                        
                        if url:
                            st.markdown(f"**{i}. [{title}]({url})**")
                        else:
                            st.markdown(f"**{i}. {title}**")
                        
                        info_parts = [f"📕 Fonte: {source}"]
                        if chapter:
                            info_parts.append(f"📑 {chapter}")
                        info_parts.append(f"📜 {license_type}")
                        st.caption("   " + " | ".join(info_parts))
                        
                        if relevance:
                            st.caption(f"   💡 {relevance}")
            # ═══════════════════════════════════════════════════════════════════════
            
            # ═══════════════════════════════════════════════════════════════════════
            # PHASE 0b: EXPERT-VETTED RESOURCES (Copyright-Safe)
            # Resources curated by domain experts from kg_neuro_resources.json
            # ═══════════════════════════════════════════════════════════════════════
            if key_concepts:
                st.divider()
                with st.expander("📚 Risorse Consigliate dagli Esperti", expanded=False):
                    st.caption("Risorse educative verificate da esperti di dominio (tutte copyright-safe):")
                    
                    try:
                        from aix.agent.media.resource_lookup import ResourceLookup
                        
                        # Get domain from session state
                        expert_domain = st.session_state.get('agent_result_domain', 'neuro')
                        
                        # Initialize lookup
                        lookup = ResourceLookup(domain=expert_domain)
                        
                        if lookup.loaded:
                            # Find resources for key concepts
                            resources = lookup.find_resources_for_concepts(
                                key_concepts[:5], 
                                max_per_concept=2
                            )
                            
                            if resources.has_resources():
                                # Show concept context
                                st.info(f"🎯 Concetti: {', '.join(key_concepts[:3])}")
                                
                                # Group resources by type for cleaner display
                                textbooks = resources.filter_by_type("textbook")
                                simulations = resources.filter_by_type("interactive_simulation")
                                websites = resources.filter_by_type("website")
                                courses = resources.filter_by_type("course")
                                papers = resources.filter_by_type("academic_paper")
                                videos = resources.filter_by_type("video_channel")
                                
                                # Display textbooks
                                if textbooks:
                                    st.markdown("#### 📖 Libri di Testo Aperti")
                                    for r in textbooks:
                                        st.markdown(f"**[{r.title}]({r.url})**")
                                        st.caption(f"   🏛️ {r.source_org} | 📜 {r.license} ✅")
                                        if r.description:
                                            st.caption(f"   {r.description[:100]}...")
                                
                                # Display interactive simulations
                                if simulations:
                                    st.markdown("#### 🎮 Simulazioni Interattive")
                                    for r in simulations:
                                        st.markdown(f"**[{r.title}]({r.url})**")
                                        st.caption(f"   🏛️ {r.source_org} | 📜 {r.license} ✅")
                                
                                # Display websites
                                if websites:
                                    st.markdown("#### 🌐 Siti Web Educativi")
                                    for r in websites[:4]:  # Limit to 4
                                        st.markdown(f"**[{r.title}]({r.url})**")
                                        st.caption(f"   🏛️ {r.source_org} | 📜 {r.license} ✅")
                                
                                # Display courses
                                if courses:
                                    st.markdown("#### 🎓 Corsi Online")
                                    for r in courses:
                                        st.markdown(f"**[{r.title}]({r.url})**")
                                        st.caption(f"   🏛️ {r.source_org} | 📜 {r.license} ✅")
                                
                                # Display academic papers
                                if papers:
                                    st.markdown("#### 📄 Articoli Accademici (Open Access)")
                                    for r in papers:
                                        st.markdown(f"**[{r.title}]({r.url})**")
                                        st.caption(f"   🏛️ {r.source_org} | 📜 {r.license} ✅")
                                
                                # Display video channels
                                if videos:
                                    st.markdown("#### 🎥 Canali Video")
                                    for r in videos:
                                        st.markdown(f"**[{r.title}]({r.url})**")
                                        st.caption(f"   🏛️ {r.source_org} | ⚠️ Solo link (copyright)")
                                
                                # Footer with stats
                                st.divider()
                                stats = lookup.get_stats()
                                st.caption(
                                    f"📊 Totale risorse disponibili: {stats['total_resources']} | "
                                    f"Tutte verificate copyright-safe ✅"
                                )
                            else:
                                st.info("Nessuna risorsa esperta trovata per questi concetti.")
                        else:
                            st.warning("Database risorse esperti non disponibile.")
                    
                    except ImportError as e:
                        st.warning(f"Modulo risorse non disponibile: {e}")
                    except Exception as e:
                        st.error(f"Errore nel caricamento risorse: {e}")
            # ═══════════════════════════════════════════════════════════════════════
            
            # ═══════════════════════════════════════════════════════════════════════
            # PHASE 4: LIVE EXTERNAL API SEARCH
            # ═══════════════════════════════════════════════════════════════════════
            if key_concepts:
                st.divider()
                with st.expander("🔍 Ricerca Live (API Esterne)", expanded=False):
                    st.caption("Cerca contenuti in tempo reale da fonti esterne:")
                    
                    # Search topic selector
                    search_topic = st.selectbox(
                        "Seleziona argomento da cercare:",
                        options=key_concepts[:5],
                        key="live_search_topic"
                    )
                    
                    live_col1, live_col2, live_col3 = st.columns(3)
                    
                    with live_col1:
                        if st.button(
                            "🎥 Cerca su YouTube",
                            use_container_width=True,
                            key="live_youtube_btn",
                            help="Cerca video educativi in tempo reale"
                        ):
                            st.session_state['live_search_type'] = 'youtube'
                            st.session_state['live_search_query'] = search_topic
                    
                    with live_col2:
                        if st.button(
                            "📖 Wikipedia",
                            use_container_width=True,
                            key="live_wikipedia_btn",
                            help="Cerca articolo Wikipedia"
                        ):
                            st.session_state['live_search_type'] = 'wikipedia'
                            st.session_state['live_search_query'] = search_topic
                    
                    with live_col3:
                        if st.button(
                            "📚 Papers Recenti",
                            use_container_width=True,
                            key="live_papers_btn",
                            help="Cerca articoli scientifici recenti (2020+)"
                        ):
                            st.session_state['live_search_type'] = 'papers'
                            st.session_state['live_search_query'] = search_topic
                    
                    # Display live search results
                    if st.session_state.get('live_search_type') and st.session_state.get('live_search_query'):
                        search_type = st.session_state['live_search_type']
                        search_query = st.session_state['live_search_query']
                        
                        with st.spinner(f"Cercando '{search_query}'..."):
                            try:
                                from aix.agent.media.external_apis import ExternalMediaAPI
                                api = ExternalMediaAPI()
                                
                                if search_type == 'youtube':
                                    st.markdown(f"#### 🎥 Video YouTube per '{search_query}'")
                                    videos = asyncio.run(api.search_youtube(search_query, max_results=5))
                                    
                                    if videos:
                                        for i, v in enumerate(videos, 1):
                                            st.markdown(f"**{i}. [{v.title}]({v.url})**")
                                            if v.channel:
                                                st.caption(f"   📺 {v.channel}")
                                    else:
                                        st.info("Nessun video trovato.")
                                
                                elif search_type == 'wikipedia':
                                    # Get output language from session state
                                    wiki_lang = st.session_state.get('agent_language', 'it')
                                    lang_name = "Italiano" if wiki_lang == "it" else "English"
                                    st.markdown(f"#### 📖 Wikipedia ({lang_name}): '{search_query}'")
                                    wiki = asyncio.run(api.get_wikipedia_summary(search_query, language=wiki_lang))
                                    
                                    if wiki:
                                        st.markdown(f"**[{wiki.title}]({wiki.url})**")
                                        st.markdown(wiki.summary)
                                        if wiki.thumbnail_url:
                                            st.image(wiki.thumbnail_url, width=200)
                                    else:
                                        st.info(f"Articolo non trovato su Wikipedia {lang_name}.")
                                
                                elif search_type == 'papers':
                                    st.markdown(f"#### 📚 Papers Recenti per '{search_query}'")
                                    papers = asyncio.run(api.search_semantic_scholar(
                                        search_query, max_results=5, year_from=2020
                                    ))
                                    
                                    if papers:
                                        for i, p in enumerate(papers, 1):
                                            authors_str = ", ".join(p.authors[:2])
                                            if len(p.authors) > 2:
                                                authors_str += " et al."
                                            
                                            st.markdown(f"**{i}. [{p.title}]({p.url})**")
                                            st.caption(f"   👤 {authors_str} ({p.year}) | 📊 {p.citation_count} citations")
                                            if p.is_open_access:
                                                st.caption(f"   ✅ Open Access | [PDF]({p.pdf_url})")
                                            if p.doi:
                                                st.caption(f"   DOI: {p.doi}")
                                    else:
                                        st.info("Nessun paper trovato.")
                                
                                asyncio.run(api.close())
                                
                            except Exception as e:
                                st.error(f"Errore nella ricerca: {e}")
                        
                        # Clear search state after displaying
                        if st.button("🔄 Nuova ricerca", key="clear_live_search"):
                            st.session_state['live_search_type'] = None
                            st.session_state['live_search_query'] = None
                            st.rerun()
            # ═══════════════════════════════════════════════════════════════════════
            
            # ═══════════════════════════════════════════════════════════════════════
            # PHASE 5: AI DIAGRAM GENERATION (Multi-Generator)
            # Mermaid.js (FREE), DALL-E 3 ($0.04), Canva (Coming Soon)
            # ═══════════════════════════════════════════════════════════════════════
            if key_concepts:
                st.divider()
                with st.expander("🎨 Genera Diagramma AI", expanded=False):
                    st.caption("Genera diagrammi educativi con diversi generatori")
                    
                    # Generator selector with descriptions
                    generator_options = {
                        "🆓 Mermaid.js (Gratuito, testo preciso)": "mermaid",
                        "🎨 DALL-E 3 (Visivo, $0.04/immagine)": "dalle",
                        "📐 Canva (Pro templates - Coming Soon)": "canva"
                    }
                    
                    selected_generator_label = st.radio(
                        "📋 Scegli il generatore:",
                        options=list(generator_options.keys()),
                        index=0,  # Default to Mermaid (free)
                        key="diagram_generator_selector",
                        horizontal=True
                    )
                    
                    selected_generator = generator_options[selected_generator_label]
                    
                    # Show info based on generator
                    if selected_generator == "mermaid":
                        st.info("✅ **Mermaid.js**: Gratuito, testo accurato, SVG scalabile, codice modificabile")
                    elif selected_generator == "dalle":
                        st.warning("⚠️ **DALL-E 3**: Costo ~$0.04 per immagine, visivamente attraente ma testo può essere impreciso")
                    elif selected_generator == "canva":
                        st.info("🚧 **Canva**: Integrazione in arrivo. Usa Mermaid.js o DALL-E nel frattempo.")
                    
                    # Concept selector
                    gen_concept = st.selectbox(
                        "📊 Seleziona concetto:",
                        options=key_concepts[:5],
                        key="diagram_concept"
                    )
                    
                    # Diagram type selector (same for all generators, mapped internally)
                    diagram_types = {
                        "Mappa Concettuale": "mindmap",
                        "Diagramma di Flusso": "flowchart",
                        "Gerarchia": "hierarchy",
                        "Timeline": "timeline",
                        "Confronto": "comparison",
                        "Processo": "process"
                    }
                    
                    selected_type_label = st.selectbox(
                        "📐 Tipo di diagramma:",
                        options=list(diagram_types.keys()),
                        key="diagram_type_selector"
                    )
                    selected_type = diagram_types[selected_type_label]
                    
                    # Optional description (for DALL-E)
                    if selected_generator == "dalle":
                        custom_description = st.text_area(
                            "Descrizione aggiuntiva (opzionale):",
                            value="",
                            key="diagram_description",
                            help="Aggiungi dettagli specifici per il diagramma DALL-E"
                        )
                    else:
                        custom_description = ""
                    
                    # Generate button (disabled for Canva)
                    button_disabled = selected_generator == "canva"
                    button_label = "🎨 Genera Diagramma" if not button_disabled else "🚧 Coming Soon"
                    
                    if st.button(
                        button_label,
                        use_container_width=True,
                        key="diagram_generate_btn",
                        type="primary",
                        disabled=button_disabled
                    ):
                        # Determine spinner message
                        if selected_generator == "mermaid":
                            spinner_msg = f"Generando diagramma Mermaid per '{gen_concept}'..."
                        else:
                            spinner_msg = f"Generando diagramma DALL-E per '{gen_concept}'... (10-20 secondi)"
                        
                        with st.spinner(spinner_msg):
                            try:
                                if selected_generator == "mermaid":
                                    # Use Mermaid generator (FREE)
                                    from aix.agent.media.mermaid_generator import MermaidGenerator
                                    
                                    generator = MermaidGenerator()
                                    mermaid_result = asyncio.run(generator.generate(
                                        concept=gen_concept,
                                        diagram_type=selected_type,
                                        related_concepts=key_concepts[:3],
                                        validate=False
                                    ))
                                    asyncio.run(generator.close())
                                    
                                    if mermaid_result.success:
                                        st.session_state['generated_diagram'] = {
                                            'type': 'mermaid',
                                            'concept': gen_concept,
                                            'diagram_type': selected_type_label,
                                            'svg_url': mermaid_result.svg_url,
                                            'png_url': mermaid_result.png_url,
                                            'mermaid_code': mermaid_result.mermaid_code,
                                            'cost': 0.0
                                        }
                                        st.success("✅ Diagramma Mermaid generato (GRATUITO)!")
                                    else:
                                        st.error(f"❌ Errore: {mermaid_result.error_message}")
                                
                                elif selected_generator == "dalle":
                                    # Use DALL-E generator
                                    from aix.agent.media.image_generator import ImageGenerator, DiagramType
                                    
                                    # Map to DALL-E diagram types
                                    dalle_type_mapping = {
                                        "mindmap": "concept_map",
                                        "flowchart": "flowchart",
                                        "hierarchy": "hierarchy",
                                        "timeline": "infographic",
                                        "comparison": "comparison",
                                        "process": "process"
                                    }
                                    dalle_type_str = dalle_type_mapping.get(selected_type, "concept_map")
                                    
                                    generator = ImageGenerator()
                                    diagram_type = DiagramType(dalle_type_str)
                                    
                                    description = custom_description or f"Educational concept about {gen_concept}"
                                    
                                    result_img = asyncio.run(generator.generate_educational_diagram(
                                        concept=gen_concept,
                                        description=description,
                                        diagram_type=diagram_type
                                    ))
                                    
                                    if result_img:
                                        st.session_state['generated_diagram'] = {
                                            'type': 'dalle',
                                            'concept': gen_concept,
                                            'diagram_type': selected_type_label,
                                            'image_url': result_img.url,
                                            'cost': result_img.cost_estimate,
                                            'generated_at': result_img.generated_at
                                        }
                                        st.success(f"✅ Diagramma DALL-E generato (${result_img.cost_estimate:.2f})!")
                                    else:
                                        st.error("❌ Generazione DALL-E fallita. Riprova.")
                                        
                            except Exception as e:
                                st.error(f"Errore: {e}")
                    
                    # Display generated diagram
                    if st.session_state.get('generated_diagram'):
                        gen_diag = st.session_state['generated_diagram']
                        st.markdown(f"#### 🖼️ Diagramma: {gen_diag['concept']}")
                        
                        if gen_diag['type'] == 'mermaid':
                            # Display Mermaid SVG
                            st.image(gen_diag['svg_url'], caption=f"{gen_diag['diagram_type']} (Mermaid.js)")
                            st.caption(f"💰 Costo: GRATUITO | 📊 Formato: SVG")
                            
                            # Show Mermaid code (copyable) - using checkbox instead of nested expander
                            show_code = st.checkbox("📝 Mostra codice Mermaid", key="show_mermaid_code")
                            if show_code:
                                st.code(gen_diag['mermaid_code'], language="text")
                                st.caption("💡 Copia questo codice per modificarlo su [mermaid.live](https://mermaid.live)")
                            
                            # Download links
                            col_svg, col_png = st.columns(2)
                            with col_svg:
                                st.markdown(f"[📥 Scarica SVG]({gen_diag['svg_url']})")
                            with col_png:
                                st.markdown(f"[📥 Scarica PNG]({gen_diag['png_url']})")
                        
                        elif gen_diag['type'] == 'dalle':
                            # Display DALL-E image
                            st.image(gen_diag['image_url'], caption=f"{gen_diag['diagram_type']} (DALL-E 3)")
                            st.caption(f"💰 Costo: ${gen_diag['cost']:.2f} | 📅 {gen_diag.get('generated_at', '')[:10]}")
                            
                            # Download link
                            st.markdown(f"[📥 Scarica Immagine]({gen_diag['image_url']})")
                        
                        # Clear button
                        if st.button("🗑️ Cancella diagramma", key="clear_diagram"):
                            st.session_state['generated_diagram'] = None
                            st.rerun()
            # ═══════════════════════════════════════════════════════════════════════
            
            # ═══════════════════════════════════════════════════════════════════════
            # DOWNLOAD SECTION - 3 formats like GraphRAG mode
            # ═══════════════════════════════════════════════════════════════════════
            st.divider()
            
            with st.expander("📥 Download Risposta", expanded=False):
                st.markdown("**Scarica la risposta in diversi formati:**")
                
                # Get query and domain from session state
                original_query = st.session_state.get('agent_query', 'Query')
                agent_domain = st.session_state.get('agent_result_domain', 'neuro')
                timestamp = time.strftime("%Y%m%d_%H%M%S")
                file_prefix = "lesson_plan" if _is_lesson_intent(query_intent) else query_intent
                
                dl_col1, dl_col2, dl_col3 = st.columns(3)
                
                with dl_col1:
                    # TXT Download
                    txt_content = generate_agent_download_txt(
                        query=original_query,
                        response=result.lesson_plan,
                        result=result,
                        domain=agent_domain,
                        query_intent=query_intent
                    )
                    st.download_button(
                        label="📄 Download as TXT",
                        data=txt_content,
                        file_name=f"{file_prefix}_{timestamp}.txt",
                        mime="text/plain",
                        use_container_width=True,
                        help="Download as plain text file"
                    )
                
                with dl_col2:
                    # Markdown Download
                    md_content = generate_agent_download_markdown(
                        query=original_query,
                        response=result.lesson_plan,
                        result=result,
                        domain=agent_domain,
                        query_intent=query_intent
                    )
                    st.download_button(
                        label="📝 Download as Markdown",
                        data=md_content,
                        file_name=f"{file_prefix}_{timestamp}.md",
                        mime="text/markdown",
                        use_container_width=True,
                        help="Download as Markdown file (formatted)"
                    )
                
                with dl_col3:
                    # PDF Download
                    pdf_content = generate_agent_download_pdf(
                        query=original_query,
                        response=result.lesson_plan,
                        result=result,
                        domain=agent_domain,
                        query_intent=query_intent
                    )
                    
                    if pdf_content:
                        st.download_button(
                            label="📕 Download as PDF",
                            data=pdf_content,
                            file_name=f"{file_prefix}_{timestamp}.pdf",
                            mime="application/pdf",
                            use_container_width=True,
                            help="Download as PDF file (professional format)"
                        )
                    else:
                        st.button(
                            "📕 PDF (Non Disponibile)",
                            disabled=True,
                            use_container_width=True,
                            help="Installa la libreria reportlab per abilitare l'export PDF"
                        )
                
                st.caption("💡 I file includono: richiesta originale, risposta completa, metriche di qualità e timestamp.")
            
            # Clear result button
            if st.button("🗑️ Nuova Richiesta", use_container_width=True):
                del st.session_state['agent_result']
                st.rerun()
        else:
            st.warning("Nessun piano di lezione generato.")
            if result.error:
                st.error(f"Errore: {result.error}")
    
    with tab2:
        st.markdown("### 📊 Metriche di Qualità")
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("📦 Nodi Utilizzati", result.nodes_used)
        with col2:
            st.metric("💡 Raccomandazioni", result.recommendations_used)
        with col3:
            approved_emoji = "✅" if result.approved else "❌"
            st.metric("Stato", f"{approved_emoji} {'Approvato' if result.approved else 'Non Approvato'}")
        
        # Scores breakdown
        if result.scores:
            st.markdown("#### Punteggi Critic Agent")
            scores_df = pd.DataFrame([
                {"Criterio": k.title(), "Punteggio": f"{v}/5"} 
                for k, v in result.scores.items()
            ])
            st.dataframe(scores_df, use_container_width=True, hide_index=True)
        
        # Critique summary
        if result.critique_summary:
            st.markdown("#### 💬 Feedback del Critic Agent")
            st.info(result.critique_summary)
    
    with tab3:
        st.markdown("### 🔍 Dettagli Pipeline")
        
        st.markdown(f"**Query originale:** {query}")
        st.markdown(f"**Successo:** {'✅ Sì' if result.success else '❌ No'}")
        st.markdown(f"**Revisioni effettuate:** {result.revision_count}")
        
        if result.error:
            st.error(f"**Errore:** {result.error}")
        
        # Show full result as JSON (collapsible)
        with st.expander("🔧 Dati grezzi (JSON)", expanded=False):
            st.json(result.to_dict())


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
        # === MODE SELECTOR (NEW) ===
        st.header("🎯 Modalità")
        app_mode = st.radio(
            "Seleziona modalità:",
            ["GraphRAG", "Agent (Lesson Planner)"],
            index=0,  # Default: GraphRAG
            help="**GraphRAG**: Ricerca nel Knowledge Graph con contesto e raccomandazioni.\n\n**Agent**: Generazione automatica di piani di lezione completi.",
            key="app_mode_selector"
        )
        
        st.divider()
        
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
        
        # Embedding Mode indicator
        st.divider()
        st.markdown("### 🔬 Ricerca Semantica")
        
        # Get current embedding mode from aix.core.config
        try:
            from aix.core.config import config as app_config
            embedding_mode = getattr(app_config.embedding, 'mode', 'node2vec')
            node2vec_weight = getattr(app_config.embedding, 'node2vec_weight', 0.4)
            semantic_weight = 1 - node2vec_weight
        except:
            embedding_mode = "node2vec"
            node2vec_weight = 1.0
            semantic_weight = 0.0
        
        if embedding_mode == "hybrid_semantic":
            st.success("🔀 **Modalità Ibrida**")
            st.markdown(f"""
            - **Node2Vec**: struttura del grafo ({node2vec_weight*100:.0f}%)
            - **OpenAI Embeddings**: significato testuale ({semantic_weight*100:.0f}%)
            """)
        else:
            st.info("📊 **Modalità Node2Vec**")
            st.markdown("""
            - Ricerca basata sulla struttura del grafo
            - Trova concetti simili per connessioni
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
    
    # ═══════════════════════════════════════════════════════════════════════════════
    # MAIN AREA - MODE-DEPENDENT CONTENT
    # ═══════════════════════════════════════════════════════════════════════════════
    
    if app_mode == "Agent (Lesson Planner)":
        # ╔════════════════════════════════════════════════════════════════════════╗
        # ║  AGENT MODE - LESSON PLAN GENERATOR                                     ║
        # ╚════════════════════════════════════════════════════════════════════════╝
        render_agent_mode()
        return  # Exit main() early - Agent mode has its own complete UI
    
    # ╔════════════════════════════════════════════════════════════════════════════╗
    # ║  GRAPHRAG MODE - ALL EXISTING CODE BELOW (100% UNCHANGED)                   ║
    # ╚════════════════════════════════════════════════════════════════════════════╝
    
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
            
            # Reuse the translation already done by the main pipeline to avoid
            # a redundant LLM API call inside MetricsCalculator
            _enhanced_query = cypher_result.get('enhanced_query')  # already translated English query
            query_metrics = metrics_calculator.calculate_all(
                query=query,
                retrieved_nodes=retrieval_result.nodes if retrieval_result else [],
                llm_response=llm_response.get('response', ''),
                cypher_query=cypher_result.get('cypher_query', ''),
                total_relationships=retrieval_result.metadata.get('total_triples', 0) if retrieval_result else 0,
                domain=selected_domain,
                translated_query=_enhanced_query,
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
            
            # Show current embedding mode
            try:
                from aix.core.config import config as app_config
                embedding_mode = getattr(app_config.embedding, 'mode', 'node2vec')
                node2vec_weight = getattr(app_config.embedding, 'node2vec_weight', 0.4)
                semantic_weight = 1 - node2vec_weight
            except:
                embedding_mode = "node2vec"
                node2vec_weight = 1.0
                semantic_weight = 0.0
            
            # Mode indicator badge
            if embedding_mode == "hybrid_semantic":
                st.success(f"🔀 **Hybrid Mode** (α={node2vec_weight:.0%} Node2Vec, β={semantic_weight:.0%} OpenAI)")
            else:
                st.info("📊 **Node2Vec Mode** (struttura grafo)")
            
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
                        "Semantic Nodes",
                        metadata.get('semantic_count', 0),
                        help="Additional nodes from semantic/vector search (scored by both Node2Vec and OpenAI in hybrid mode)"
                    )
                    st.metric(
                        "Relationships",
                        metadata.get('total_triples', 0),
                        help="Total relationships retrieved"
                    )
                
                if embedding_mode == "hybrid_semantic":
                    st.markdown(f"""
                    **💡 Hybrid Retrieval ({node2vec_weight:.0%}/{semantic_weight:.0%}):**
                    - 🎯 **Graph Structure** ({node2vec_weight:.0%}): Node2Vec cattura relazioni nel grafo
                    - 🔍 **Text Semantics** ({semantic_weight:.0%}): OpenAI cattura significato testuale
                    - 🚀 **Combined Score**: `α × Node2Vec + β × OpenAI`
                    """)
                else:
                    st.markdown("""
                    **💡 Node2Vec Retrieval:**
                    - 🎯 **Precision**: Direct graph relationships (exact matches)
                    - 🔍 **Breadth**: Node2Vec semantic similarity (related concepts)
                    - 🚀 **Coverage**: Neighbor expansion (contextual information)
                    """)
            else:
                st.warning("No comparison data available")

if __name__ == "__main__":
    main()

