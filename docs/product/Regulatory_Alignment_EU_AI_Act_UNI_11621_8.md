# Agentic GraphRAG — Regulatory Alignment Analysis

## EU AI Act (Regulation 2024/1689) and UNI 11621-8:2026

**Date:** May 14, 2026
**From:** AI Team (LM)
**To:** FEM Direction, Legal Office, Compliance
**Subject:** Assessment of the Agentic GraphRAG system's alignment with the EU Artificial Intelligence Act and the Italian standard UNI 11621-8:2026 on AI professional role profiles
**Classification:** Internal — Regulatory Analysis

---

## 1. Purpose of This Document

This document provides a structured analysis of how the Agentic GraphRAG system — the multi-agent AI pipeline integrated into the AixLearning platform for lesson-plan generation — aligns with two key regulatory instruments:

1. **The EU Artificial Intelligence Act** (Regulation 2024/1689), the world's first comprehensive legal framework for AI, with phased enforcement milestones from February 2025 through August 2027.

2. **UNI 11621-8:2026**, Italy's first-in-Europe national standard defining professional role profiles for the AI sector, published on April 30, 2026, and aligned with the European e-Competence Framework (UNI EN 16234-1).

The analysis is intended for the FEM legal office, direction, and compliance stakeholders. It identifies where the system is already aligned, where gaps exist, and what actions are recommended — with associated timelines and priorities.

---

## 2. System Under Analysis

**System name:** Agentic GraphRAG for AixLearning
**System type:** Multi-agent AI pipeline for educational content generation (lesson plans)
**Provider:** AI Team, FEM-modena (`graphrag-aixlearning` repository)
**Deployer:** FEM-modena (`FEM-modena/aixlearning` — the AixLearning production platform)

**What the system does:**
- Accepts a teacher's natural-language request along with their educational profile (class composition, disabilities, classroom environment, time constraints)
- Runs a 4-agent pipeline: Planner (intent analysis) → Retriever (Knowledge Graph + hybrid sources) → Writer (lesson plan generation) → Critic (quality review)
- Returns a structured lesson plan in Markdown format, grounded in a curated Neo4j Knowledge Graph (720+ pedagogical concepts across UDL and Neuroscience domains)
- Operates as a backend service consumed by the AixLearning Django frontend via API

**What the system does NOT do:**
- It does not evaluate students or their learning outcomes
- It does not determine access to education or assign persons to educational institutions
- It does not assess the educational level an individual should receive
- It does not monitor student behavior during tests
- It does not perform emotion recognition
- It does not make autonomous decisions — the teacher always reviews, edits, and decides whether to use the generated content

---

## 3. EU AI Act — Risk Classification

### 3.1 Is the system "high-risk" under Annex III?

**Conclusion: NO.** The system does not fall within any of the four high-risk categories defined in Annex III, Point 3 (Education and vocational training):

| Annex III Point 3 category | Description | Applicability to our system |
|---|---|---|
| **(a)** Admissions and access | AI determining access, admission, or assignment to educational institutions at all levels | **Not applicable.** The system does not decide who enters a school or program. It generates lesson plans for teachers who have already been admitted to their professional role. |
| **(b)** Learning outcome evaluation | AI evaluating learning outcomes, including AI used to steer the learning process | **Not applicable.** The system does not evaluate any student's learning. It produces educational content (lesson plans) that teachers then deliver. The teacher — not the AI — evaluates learning outcomes. |
| **(c)** Educational level assessment | AI assessing the appropriate level of education a person will receive or access | **Not applicable.** The system does not assess or recommend what level of education anyone should receive. |
| **(d)** Behavior monitoring and proctoring | AI monitoring and detecting prohibited behavior during tests | **Not applicable.** The system has no student monitoring, proctoring, or behavioral detection capabilities of any kind. |

The EU AI Act's own interpretive guidance uses a "language-learning chatbot with no evaluation function" as an example of a system that falls **outside** the scope of Annex III Point 3. Our system is even further from the regulatory boundary: it is a teacher-facing content authoring tool with no evaluative, classificatory, or monitoring function regarding students.

### 3.2 Is the system a "prohibited practice" under Article 5?

**Conclusion: NO.** The system does not engage in any of the eight prohibited practices defined in Article 5, including:

- No subliminal manipulation of persons
- No exploitation of vulnerabilities of specific groups
- No social scoring
- No real-time remote biometric identification
- **No emotion recognition in educational settings** (Article 5(1)(f)) — the system does not process any biometric data, facial expressions, or emotional states

### 3.3 What risk category does the system fall into?

The system is classified as a **limited-risk or minimal-risk AI system**. It is primarily a content-generation tool that assists a human professional (the teacher) in preparing educational materials. The teacher retains full control over whether and how to use the generated content.

However, certain EU AI Act obligations apply regardless of risk classification, as described in the following sections.

---

## 4. EU AI Act Obligations That Apply

### 4.1 Article 4 — AI Literacy (in force since February 2, 2025)

**Requirement:** Providers and deployers of AI systems must take measures to ensure, to their best extent, a sufficient level of AI literacy of their staff and other persons dealing with the operation and use of AI systems on their behalf.

**Our alignment:** The Agentic GraphRAG system includes built-in explainability features that directly support teacher AI literacy:
- The **coverage banner** (`#9.UX-5`) tells teachers in plain Italian whether the AI's response is grounded in the Knowledge Graph or based on general AI knowledge — e.g., *"Ricerca completata sulla base Neuro"* vs. *"Questo argomento non è presente nella base UDL: la lezione si baserà su conoscenze didattiche generali del nostro assistente e fonti esterne verificate ed integrate."*
- The **Critic verdict** communicates that the output was quality-checked
- The **media sidebar** surfaces the specific sources (academic papers, videos, open educational resources) used to inform the lesson plan

**Recommendation:** Supplement system-level explainability with a brief "How the AI works" guide for teachers, accessible from the AixLearning UI. This can be a static page explaining the 4-agent pipeline in non-technical language.

### 4.2 Article 50 — Transparency Obligations for AI-Generated Content

**Requirement:** Providers of AI systems that generate synthetic text, audio, image, or video content must ensure that the outputs are marked in a machine-readable format and are detectable as artificially generated or manipulated. The marking must be effective, interoperable, robust, and reliable, as far as technically feasible.

**Enforcement date: August 2, 2026.**

**Our alignment — current state:** The system does not currently apply machine-readable marking to the generated lesson plans. The teacher sees the content in their chat UI and can export it as PDF or DOCX, but the exported file carries no metadata indicating it was AI-generated.

**Our alignment — what already helps:**
- Every generated lesson plan is traceable through a `trace_id` (Langfuse) and a `GenerationTrace` database record in AixLearning
- The system's response is always delivered through a clearly labeled AI chat interface

**Gap: This is the most urgent compliance item.**

**Recommended actions:**
1. Add a **machine-readable metadata field** to the `lesson_plan_md` output — options include:
   - A `<!-- ai-generated: true, system: agentic-graphrag, trace_id: ... -->` HTML comment at the top of the Markdown
   - C2PA (Coalition for Content Provenance and Authenticity) metadata in exported PDF/DOCX files
   - A `X-AI-Generated: true` HTTP header on API responses
2. Add a **visible disclosure** in the AixLearning chat UI — e.g., a persistent footer or badge on every AI-generated message stating *"Contenuto generato dall'intelligenza artificiale"*
3. Ensure the disclosure persists when content is exported (PDF/DOCX should include an AI-generation notice on the first page or in document metadata)

**Timeline:** Must be implemented before August 2, 2026 (~3 months from today).

### 4.3 Article 50(1) — Interaction Transparency

**Requirement:** Providers of AI systems intended to interact directly with natural persons must ensure that the system is designed and developed in such a way that the natural person is informed they are interacting with an AI system, unless this is obvious from the circumstances and context of use.

**Our alignment:** The AixLearning chat UI presents the interaction as a conversation with an AI assistant. The tool names ("UDL", "NEURO") and the chat interface design make it contextually clear that the teacher is interacting with AI, not a human colleague. However, there is no explicit, persistent disclosure label.

**Recommendation:** Add a visible "Assistente AI" or "Generato da IA" label to the chat interface header. This is a minor Django template change on the AixLearning side.

### 4.4 Articles 8-15 — High-Risk Requirements (voluntarily aligned)

While the system is **not legally required** to comply with the high-risk requirements of Articles 8-15, it already demonstrates substantial alignment with many of them. This voluntary alignment is a competitive and reputational advantage for FEM — it demonstrates a commitment to responsible AI that goes beyond minimum legal requirements.

| EU AI Act Article | High-Risk Requirement | System alignment | Status |
|---|---|---|---|
| **Art. 9 — Risk Management** | Continuous identification and mitigation of risks | The 3-tier coverage classification (`healthy` / `limited` / `out_of_scope`) explicitly identifies when the system's knowledge is insufficient for a topic — a form of risk-aware output. The Critic agent identifies quality deficiencies and triggers automatic revision. | **Aligned** |
| **Art. 10 — Data Governance** | Training data must be representative, relevant, and documented | The Neo4j Knowledge Graph is curated from academic sources, domain-specific (UDL + Neuroscience), with 720+ auditable concepts and 745+ relationships. Data provenance is fully traceable. | **Aligned** |
| **Art. 11 — Technical Documentation** | System must be documented before placement on the market | Extensive documentation exists: architecture analysis, roadmap, integration strategy, frontend ADR, migration guides, and this regulatory alignment document. | **Aligned** |
| **Art. 13 — Transparency** | Users must be able to understand AI output and its limitations | The coverage banner tells teachers when content is KG-grounded vs. general AI knowledge. The Critic verdict communicates quality-check results. Domain-specific labels ("base UDL — pedagogia inclusiva" / "base Neuro") clarify the knowledge source. | **Strongly aligned** |
| **Art. 14 — Human Oversight** | Humans must be able to understand, monitor, and intervene | The teacher is always in the loop: they initiate the request, review the output, can edit the lesson plan, regenerate it, or discard it entirely. The AI never autonomously delivers content to students. Multi-turn conversation design means the teacher actively steers the agent. | **Strongly aligned** |
| **Art. 15 — Accuracy, Robustness, Cybersecurity** | System must be accurate, resilient to errors, and secure | The Critic agent provides an accuracy check. The Dramatiq retry mechanism (3 retries + automatic credit refund) provides robustness. Langfuse traces enable post-deployment monitoring. Internal Docker network architecture prevents public exposure of the agent API. | **Aligned** |
| **Art. 19 — Automatic Logging** | System must generate logs | Langfuse traces are live with per-agent telemetry, duration tracking, coverage metrics, and revision counts. `GenerationTrace` records persist in PostgreSQL with full trace IDs. | **Aligned** |

---

## 5. UNI 11621-8:2026 — Professional Competency Alignment

### 5.1 What the standard requires

UNI 11621-8:2026 (published April 30, 2026) defines 12 professional role profiles for the AI sector. It is not a system-level regulation — it defines the competencies, knowledge, skills, and KPIs expected of the professionals who design, build, and operate AI systems. The standard translates obligations from the EU AI Act and Italian Law 132/2025 into structured role definitions, aligned with the European e-Competence Framework.

The standard identifies three **cross-cutting themes** required across all 12 profiles:

1. **AI Act and regulatory compliance** — Every role must include compliance with EU Regulation 2024/1689, GDPR, and ISO/IEC 42001 among its main tasks
2. **Computational and environmental sustainability** — Every role includes energy efficiency indicators (kWh for inference, CO2eq reduction) and GreenOps practices
3. **Explainability and transparency (XAI)** — Every technical role requires explainable AI skills and the production of auditable documentation (Model Cards, AI registries, non-technical explanations)

### 5.2 Team role mapping

| UNI 11621-8 Role | Team member | Demonstrated competencies |
|---|---|---|
| **AI NLP Engineer** (Profile 11) | LM | Multi-agent pipeline design (LangGraph), Knowledge Graph integration (Neo4j), prompt chain management (Planner/Writer/Critic prompts), explainability by construction (`#9.UX-5` coverage banner), SSE streaming architecture, educational profile-aware content generation |
| **AI Data Engineer** (Profile 7) | AG (Angelo) | Knowledge Graph data pipeline (Neo4j ingestion), media pool curation and generation (OpenAlex, yt-dlp, LM Studio), dataset quality assurance (label normalization, deduplication), domain-specific data validation |
| **AI Product Manager** (Profile 3) | LM | AI product lifecycle management (CORE 0-6 roadmap), stakeholder communication (ClickUp documentation, handoff documents), risk assessment (coverage tier classification), integration strategy (AixLearning analysis), compliance awareness (this document) |
| **AI Security Specialist** (Profile 9) | LM (partial) | Input/output guardrails (CORE 2 #8), JWT authentication architecture (HS256 → RS256 migration plan), CORS configuration, internal Docker network isolation |

### 5.3 Alignment with cross-cutting themes

**Theme 1 — Regulatory compliance:**
The AI team demonstrates regulatory awareness through this document, the `#9.UX-5` explainability features (aligned with Art. 13 transparency), the human-in-the-loop architecture (aligned with Art. 14 human oversight), and the Langfuse observability stack (aligned with Art. 19 logging). The team has also documented the full integration strategy, technical architecture, and production-readiness roadmap.

**Theme 2 — Computational and environmental sustainability:**
This is a **gap**. The system does not currently track environmental metrics. UNI 11621-8 requires indicators such as kWh per inference and CO2eq per operation across all technical roles.

**Recommendation:** Add `total_tokens`, `estimated_kwh`, and `co2eq_grams` fields to the `AgentRunMeta` API response and to Langfuse traces. OpenRouter provides token counts per call; conversion to kWh can be estimated using published benchmarks for cloud GPU inference (approximately 0.001-0.003 kWh per 1,000 tokens for large language models).

**Theme 3 — Explainability and transparency (XAI):**
This is the system's **strongest alignment area**. UNI 11621-8 competency S009 ("Transparency, Documentation & AI Registry") requires the ability to "generate non-technical explanations of AI systems' decision-outputs for users and regulators" and to "define differentiated disclosure levels for different audiences." The Agentic GraphRAG system implements this through:

- The **coverage banner** — plain-Italian explanation of what the AI "knows" about a topic (3-tier: fully covered, partially covered, not in Knowledge Graph)
- The **domain-aware labels** — "base UDL (pedagogia inclusiva)" / "base Neuro" — contextualizing the knowledge source
- The **Critic verdict** — communicating that quality was checked and how many revisions were performed
- The **media sidebar** — surfacing the specific academic and educational sources used
- The **per-agent SSE events** — exposing the reasoning chain (Planner intent → Retriever coverage → Writer output → Critic assessment) in structured, machine-readable format

UNI 11621-8 KPI11 measures "human-verified oversight (HITL): the percentage of critical decisions with human oversight verified by third parties out of all critical decisions." In the Agentic GraphRAG system, **100% of outputs are human-overseen** — no lesson plan reaches a student without the teacher's explicit review and approval.

---

## 6. Gap Analysis and Recommended Actions

### 6.1 Priority classification

- **P1 — Legal obligation with deadline:** Must be implemented by a specific regulatory enforcement date
- **P2 — Strongly recommended:** Not strictly required for the current risk classification but demonstrates responsible AI commitment and prepares for potential reclassification
- **P3 — Good practice:** Enhances compliance posture and competitive positioning

### 6.2 Action items

| # | Action | Regulation | Priority | Effort | Deadline |
|---|---|---|---|---|---|
| 1 | **AI-generated content marking** — add machine-readable metadata to `lesson_plan_md` output and exported documents (PDF/DOCX) | Art. 50 EU AI Act | **P1** | ~1 day | **August 2, 2026** |
| 2 | **Visible AI disclosure** — add persistent "Contenuto generato dall'IA" label in AixLearning chat UI and on exported documents | Art. 50 + Art. 50(1) EU AI Act | **P1** | ~2 hours (Django template change) | **August 2, 2026** |
| 3 | **Environmental sustainability metrics** — track and expose `total_tokens`, `estimated_kwh`, `co2eq_grams` per agent run | UNI 11621-8 (cross-cutting Theme 2) | **P2** | ~1 day | Q3 2026 |
| 4 | **Formal risk assessment document** — create `AI_Risk_Assessment.md` documenting identified risks, mitigations, and residual risks for the lesson-planning use case | Art. 9 EU AI Act (good practice for non-high-risk) | **P2** | ~half a day | Q3 2026 |
| 5 | **Model Card** — create a formal Model Card for the Agentic GraphRAG (inputs/outputs, intended use, limitations, performance metrics, bias assessment, knowledge domain boundaries) | UNI 11621-8 (CAIO competency / Art. 11 good practice) | **P2** | ~half a day | Q3 2026 |
| 6 | **Teacher AI literacy guide** — create a "How the AI works" static page for teachers, accessible from the AixLearning UI, explaining the 4-agent pipeline in non-technical language | Art. 4 EU AI Act | **P3** | ~2 hours | Q4 2026 |
| 7 | **AI Registry entry** — register the system in FEM's internal AI registry (if one exists or is planned) per UNI 11621-8 CAIO responsibility S009 | UNI 11621-8 | **P3** | ~2 hours | Q4 2026 |

### 6.3 Total effort estimate

- **P1 actions (legal deadline August 2, 2026):** ~1.5 days
- **P2 actions (Q3 2026):** ~2 days
- **P3 actions (Q4 2026):** ~half a day
- **Total:** ~4 working days across Q2-Q4 2026

---

## 7. Conclusion

The Agentic GraphRAG system occupies a favorable regulatory position:

1. **It is not a high-risk AI system** under the EU AI Act. The system generates educational content for teachers — it does not evaluate students, determine access to education, or monitor behavior. This places it in the limited-risk or minimal-risk category.

2. **Despite not being high-risk, it already satisfies many high-risk requirements** — transparency (Art. 13), human oversight (Art. 14), accuracy through the Critic agent (Art. 15), automatic logging via Langfuse (Art. 19), and risk-aware output through coverage classification (Art. 9). This voluntary alignment is a competitive advantage and demonstrates responsible AI practices.

3. **The most urgent compliance item is Article 50** — AI-generated content marking, with a hard enforcement deadline of **August 2, 2026**. This requires machine-readable metadata on generated lesson plans and a visible AI disclosure in the user interface. Estimated effort: ~1.5 days.

4. **The team's professional competencies align well with UNI 11621-8:2026** — particularly in explainability (S009), human oversight (KPI11), and multi-agent system design. The main gap is environmental sustainability metrics (Theme 2), which can be addressed with ~1 day of engineering effort.

5. **The architecture is future-proof** — if the EU AI Act's scope were ever expanded to cover educational content-generation tools as high-risk (which is not currently anticipated), the system's existing features would already satisfy the majority of the additional requirements.

---

## Appendix A — Regulatory References

| Reference | Description | Link |
|---|---|---|
| EU AI Act (Regulation 2024/1689) | Full text of the EU Artificial Intelligence Act | [eur-lex.europa.eu](https://eur-lex.europa.eu/legal-content/EN/TXT/HTML/?uri=OJ%3AL_202401689) |
| Annex III, Point 3 | High-risk classification for education and vocational training | [ai-act-service-desk.ec.europa.eu](https://ai-act-service-desk.ec.europa.eu/en/ai-act/annex-3) |
| Article 50 Guidelines (Draft) | Draft guidelines on transparency obligations, published May 8, 2026 | [digital-strategy.ec.europa.eu](https://digital-strategy.ec.europa.eu/en/library/draft-guidelines-implementation-transparency-obligations-certain-ai-systems-under-article-50-ai-act) |
| UNI 11621-8:2026 | Italian standard on AI professional role profiles | [store.uni.com](https://store.uni.com/en/uni-11621-8-2026) |
| UNI 11621-8 Announcement | Official announcement from the Italian Digital Innovation Department | [innovazione.gov.it](https://innovazione.gov.it/notizie/articoli/intelligenza-artificiale-pubblicata-la-norma-uni-11621-8/) |
| C2PA | Coalition for Content Provenance and Authenticity — standard for content provenance metadata | [c2pa.org](https://c2pa.org/) |
| ISO/IEC 42001 | International standard for AI management systems | [iso.org](https://www.iso.org/standard/81230.html) |

## Appendix B — Glossary

| Term | Definition |
|---|---|
| **Annex III** | The EU AI Act annex listing high-risk AI system categories by domain |
| **Deployer** | Entity using an AI system under its authority (in our case: FEM-modena via AixLearning) |
| **Provider** | Entity developing or placing an AI system on the market (in our case: AI Team via graphrag-aixlearning) |
| **HITL** | Human-In-The-Loop — design pattern ensuring human oversight of AI decisions |
| **XAI** | Explainable Artificial Intelligence — techniques making AI reasoning transparent to users |
| **C2PA** | Coalition for Content Provenance and Authenticity — technical standard for embedding provenance metadata in digital content |
| **Model Card** | Standardized documentation describing an AI model's intended use, capabilities, limitations, and performance characteristics |
| **UDL** | Universal Design for Learning — pedagogical framework for inclusive education |
| **KG** | Knowledge Graph — structured representation of domain knowledge as nodes and relationships |
| **LangGraph** | State-machine framework for multi-agent AI system orchestration |
| **Langfuse** | Open-source observability platform for LLM-based applications |

---

**Document owner:** LM (AI Team)
**Intended reviewers:** FEM Legal Office, FEM Direction, FEM Compliance
**Version:** 1.0
**Next review:** After August 2, 2026 enforcement milestone
