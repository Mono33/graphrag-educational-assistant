# Agentic GraphRAG — Functional Documentation

**Project:** Agentic GraphRAG — Multi-agent Lesson Planning System
**Owner:** FEM AI Team
**Document type:** Functional reference for product, pedagogy, business, and compliance stakeholders
**Version:** 0.1 (draft — table of contents only)
**Last updated:** May 2026

---

## Table of Contents

1. Introduction
   1.1 Purpose of the document
   1.2 Audience (Product, Pedagogy, Business, Legal/Compliance, Direction)
   1.3 Scope and out-of-scope
   1.4 Document conventions
   1.5 Related documents (Technical Documentation, Dev Handoff, Regulatory Alignment, Internal Deployment Plan)

2. Executive Summary
   2.1 What the platform does, in one page
   2.2 Why it matters (business + pedagogical rationale)
   2.3 Current state and pilot scope
   2.4 Strategic positioning vs. generic GenAI tools

3. Vision and Objectives
   3.1 Product vision
   3.2 Strategic objectives (FEM internal pilot, AixLearning integration, EU AI Act readiness)
   3.3 Success criteria and KPIs
   3.4 Non-goals

4. Target Users and Personas
   4.1 Primary persona — Teacher (school context)
   4.2 Secondary persona — Pedagogical coordinator
   4.3 Secondary persona — School / institute administrator
   4.4 Internal personas — AI Team, DEV team, Direction
   4.5 User needs matrix

5. Pedagogical and Scientific Foundations
   5.1 UDL (Universal Design for Learning) framework
   5.2 Neuroscience and Neurodidactics framework
   5.3 Why a Knowledge Graph (vs. plain LLM, vs. plain RAG)
   5.4 Why a multi-agent (Planner / Retriever / Writer / Critic) pipeline
   5.5 Editorial principles (grounded, transparent, customizable, didactically explicit)

6. Functional Capabilities (User-Facing)
   6.1 Lesson plan generation
   6.2 Multi-turn refinement / conversational editing
   6.3 Educational profile (class, classroom, subject, duration, BES/DSA)
   6.4 Evidence panel (sources, confidence, coverage)
   6.5 Streaming output and progressive feedback
   6.6 Export formats (Markdown, future PDF/DOCX)
   6.7 AI-content marking and disclaimers (EU AI Act compliance UX)

7. End-to-End User Journeys
   7.1 First-time teacher onboarding
   7.2 Standard lesson plan creation
   7.3 Iterative refinement with the Critic loop
   7.4 Reuse of saved profiles and prior conversations
   7.5 Failure / fallback journeys (no coverage, low confidence)

8. Functional Architecture (non-technical view)
   8.1 The four agents at a glance
   8.2 Knowledge Graph as the trusted backbone
   8.3 External sources as the breadth layer
   8.4 The teacher-facing WebUI
   8.5 Two deployment modes (standalone WebUI vs. native AixLearning integration)

9. Integration with AixLearning
   9.1 Functional view of Mode A (standalone internal pilot)
   9.2 Functional view of Mode B (native integration)
   9.3 Coexistence and migration path
   9.4 Roles and responsibilities (FEM AI Team vs. AixLearning DEV team)

10. Quality, Trust, and Editorial Control
    10.1 Grounded generation principle
    10.2 The Critic agent as an internal quality gate
    10.3 Coverage classification and confidence signaling
    10.4 Editorial review workflow (current and planned)
    10.5 Handling of low-confidence or out-of-scope requests

11. Compliance and Responsible AI
    11.1 EU AI Act positioning (limited-risk classification, expected obligations)
    11.2 UNI/PdR 11621-8 alignment
    11.3 GDPR considerations (data minimization, no PII in prompts)
    11.4 AI-generated content marking (UX and metadata)
    11.5 Logging, traceability, and auditability
    11.6 Human-in-the-loop guarantees

12. Operational Model (functional view)
    12.1 Internal pilot operations (FEM AI Team responsibilities)
    12.2 Support model (issue intake, response times, escalation)
    12.3 Monitoring and incident communication
    12.4 Release and change-management cadence

13. Roadmap (functional view)
    13.1 Wave 1 — Foundation (delivered)
    13.2 Wave 2 — Internal pilot deployment
    13.3 Wave 3 — Native AixLearning integration
    13.4 Beyond pilot — content expansion, multi-tenant, additional domains
    13.5 Explicitly deferred features

14. Risks, Limitations, and Mitigations
    14.1 Pedagogical risks (over-reliance, generic content)
    14.2 Technical risks (LLM provider, KG coverage, cost variance)
    14.3 Compliance risks (regulatory evolution)
    14.4 Adoption risks (teacher onboarding, change management)
    14.5 Mitigation strategies

15. Glossary (non-technical)

16. Appendices
    A. Mapping between functional capabilities and code modules (cross-link to Technical Documentation)
    B. Sample lesson plan inputs/outputs
    C. Functional acceptance checklist for the internal pilot
    D. Document changelog

---

> **Drafting status:** this document currently contains only the proposed table of contents. The narrative content of each section will be written in subsequent iterations once the structure is validated with Product and Direction.
