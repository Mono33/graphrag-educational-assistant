# AI Literacy for FEM — Why, What, and How

**Author:** AI Team
**Date:** January 2026

---

## What is AI Literacy

AI literacy is **"the ability to understand, evaluate, and confidently use AI technologies, recognizing both their capabilities and limitations in personal, professional, and societal contexts."**

— AI Literacy Whitepaper v0.3, AI Global Literacy Initiative, 2025

It is NOT about making everyone a developer. It is about enabling people to:

- **Understand** what AI systems do and don't do
- **Evaluate** AI outputs critically — is this answer good? why? why not?
- **Interact** effectively — give actionable feedback, not "it doesn't work"
- **Decide** informed — should we change the model? expand the Knowledge Graph? change the prompt?

75–90% of knowledge workers already use AI tools at work (Microsoft/LinkedIn Work Trend Index 2024; Udacity 2025). AI literacy is not for the people who build AI — it is for everyone who uses, evaluates, or decides about AI.

---

## Why It Matters — The Hard Data

### 1. Most AI investments fail due to organizational readiness, not technology

- **Only 5% of global firms** derive meaningful returns from AI investments (BCG, 2025)
- **56% of CEOs** say AI hasn't produced revenue or cost benefits for their businesses to date (PwC Global CEO Survey, 2026)
- **95% of enterprise GenAI pilots fail** to deliver measurable value or reach production, despite ~$40 billion in corporate AI spending (MIT Research, 2025–2026)

The #1 root cause across all studies: leadership gaps, unclear success metrics, and poor integration with existing workflows — not technical flaws.

### 2. The EU AI Act makes AI literacy a legal obligation

**Article 4 of the EU AI Act** entered into force on **2 February 2025**. It requires:

> *"Providers and deployers of AI systems shall take measures to ensure, to their best extent, a sufficient level of AI literacy of their staff and other persons dealing with the operation and use of AI systems on their behalf."*

— EU AI Act, Article 4 (artificialintelligenceact.eu)

Key facts:
- Applies to **all organizations** that deploy or provide AI systems, regardless of risk classification
- No mandatory certification — but organizations must demonstrate proportional measures
- Already in effect — compliance is required now, not in the future
- Enforced by national market surveillance authorities

**FEM deploys AI through AIxLearning. FEM is subject to Article 4.**

### 3. Educational organizations face a credibility imperative

FEM is an educational institution that advocates for innovation in learning. Deploying AI in an educational product without ensuring internal AI literacy creates a coherence gap: the organization teaches others about learning, but doesn't invest in its own learning about AI.

---

## Why It Matters Specifically for AIxLearning

### Problem 1: Feedback quality

When a domain expert says "the response is wrong," that is not actionable. The AI system has multiple stages:

```
Query → Translation → Cypher Generation → Knowledge Graph Retrieval → Context Building → LLM Response
```

Each stage can fail independently. Without basic AI literacy, stakeholders cannot distinguish:
- A **retrieval problem** (wrong nodes retrieved from the Knowledge Graph)
- A **generation problem** (LLM hallucinating despite good context)
- A **prompt problem** (system prompt missing domain constraints)
- A **data problem** (Knowledge Graph is incomplete)

Result: generic feedback ("it doesn't work") instead of diagnostic feedback ("the retrieval returned irrelevant nodes for this query type"). The AI team must reverse-engineer every issue.

### Problem 2: Product decisions require AI understanding

Product decisions about AIxLearning — which model to use, whether to expand the Knowledge Graph, how to evaluate improvements, when to migrate providers — require understanding what these components do. When the question "why is AI literacy necessary?" arises within the product team of an AI product, that is itself evidence of the gap. Without AI literacy, decisions are made on intuition rather than informed assessment.

### Problem 3: Stakeholder misalignment

When different teams (AI, product, development, domain experts) have different understandings of what a "model change" or "GraphRAG integration" means, meetings become alignment sessions instead of decision sessions. A concrete example: the discussion around migrating from GPT-4o to GPT-5 revealed that the AI team, DEV team, and product team each had different understandings of what a model change involves, what testing is required, and what metrics matter. A shared vocabulary eliminates this overhead.

### Problem 4: EU AI Act compliance gap

FEM is deploying an AI system (AIxLearning). Under Article 4 of the EU AI Act, FEM must ensure that staff interacting with this system — domain experts, product team, teachers using the platform — have proportional AI literacy. This isn't a nice-to-have — it's a regulatory obligation in effect since February 2025.

---

## What FEM AI Literacy Should Cover

Based on the U.S. Department of Labor AI Literacy Framework (February 2026) and the Digital Education Council Framework (2025), adapted for FEM's context:

| Area | What staff should understand | Who needs it most |
|------|------------------------------|-------------------|
| **AI Fundamentals** | What LLMs are, how they generate text, what "hallucination" means, what prompts do | Everyone |
| **Knowledge Graph / GraphRAG** | What the KG contains, how retrieval works, why context matters, what Cypher queries do | Domain experts, Product |
| **Evaluation & Feedback** | How to evaluate AI outputs (accuracy, relevance, completeness), how to give diagnostic feedback | Domain experts |
| **Models & Providers** | What different models offer (GPT-4o, Claude, Gemini), cost/quality trade-offs, what a migration involves | Product, Direction |
| **Limitations & Risks** | What AI cannot do, bias risks, overreliance, when to escalate to human judgment | Everyone |
| **EU AI Act basics** | Article 4 obligations, what "deployer" means, proportional literacy requirements | Direction, Legal |

---

## Proposed Format

- **Lightweight document** — not a course, not a certification
- **Iterative** — evolves as the product evolves
- **Practical** — tied to AIxLearning's actual architecture, not abstract AI theory
- **Minimal effort** — produced by AI team as part of normal documentation
- **Accessible** — glossary + one-pagers per topic, shared on internal wiki or Confluence

---

## References

1. **EU AI Act, Article 4** — artificialintelligenceact.eu/article/4 (in force since 2 February 2025)
2. **European Commission AI Literacy FAQ** — digital-strategy.ec.europa.eu/en/faqs/ai-literacy-questions-answers
3. **AI Literacy Whitepaper v0.3** — AI Global Literacy Initiative, 2025 (aigl.blog)
4. **U.S. Department of Labor AI Literacy Framework** — February 2026 (blog.barracuda.com/2026/03/18/department-labor-ai-literacy-framework)
5. **Digital Education Council AI Literacy Framework** — March 2025 (digitaleducationcouncil.com)
6. **BCG: Only 5% of Companies See Value from AI** — Business Insider, October 2025
7. **PwC Global CEO Survey: 56% say AI hasn't delivered** — Business Insider, January 2026
8. **MIT Research: 95% of GenAI Pilots Fail** — WebProNews, 2026 (citing MIT study, 2025)
9. **Microsoft/LinkedIn Work Trend Index** — 75% of knowledge workers use AI, May 2024
10. **Udacity AI at Work Survey** — 90% of workers have used AI tools, September 2025
11. **Digital Promise: AI Literacy Insights** — March 2026 (digitalpromise.org)
