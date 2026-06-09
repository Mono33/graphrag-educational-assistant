# WebUI Frontend Comparison for Agentic GraphRAG UX

**Date:** May 2026  
**Owner:** FEM AI Team  
**Scope:** Comparison between the standalone Agentic GraphRAG WebUI and the native AixLearning frontend, with focus on UX implications for a multi-agent architecture.

---

## 1. Purpose

This note explains why the standalone Agentic GraphRAG WebUI and the native AixLearning frontend are complementary, not competing, frontend surfaces.

The key question is not only which technology stack is more modern. The key question is which user experience can best expose the value of an agentic architecture: Planner, Retriever, Writer, Critic, Knowledge Graph coverage, source evidence, streaming progress, and quality review.

---

## 2. Our Standalone WebUI Stack

The standalone WebUI is served directly by the Agentic GraphRAG FastAPI application.

| Layer | Technology |
|---|---|
| Backend | FastAPI |
| Templates | Jinja2 server-rendered templates |
| Interactivity | htmx 2 |
| Streaming | Server-Sent Events (SSE), `htmx-ext-sse`, FastAPI streaming |
| Component library | WebAwesome |
| Styling | Tailwind CSS + `aix-brand.css` |
| Small client-side behavior | Alpine.js |
| Authentication | FastAPI-Users, cookie/JWT |
| Database | PostgreSQL for WebUI state and LangGraph checkpoints |
| Deployment | Docker Compose, Caddy, PostgreSQL |

This is a server-rendered hypermedia application. The backend renders HTML fragments, htmx swaps them into the page, and SSE progressively streams agent events to the browser.

The WebUI is designed as the AI Team's direct pilot surface for `https://agente.aiforlearning.digital`.

---

## 3. Native AixLearning Frontend Stack

The native AixLearning platform follows a similar server-rendered philosophy, but inside the existing Django product.

| Layer | Technology |
|---|---|
| Backend | Django |
| Templates | Django templates |
| Interactivity | htmx 2 |
| Real-time updates | Mercure / SSE |
| Component library | WebAwesome |
| Asset pipeline | Bun / TypeScript / CSS bundling |
| Authentication | Existing AixLearning user/session model |
| Business data | Existing Django models for users, classes, parties, classrooms, credits, lesson flows |
| Deployment | AixLearning Docker Compose / platform infrastructure |

The native frontend is the natural long-term home for the product experience because teachers are already inside AixLearning, with existing accounts, classroom data, credit logic, and workflows.

---

## 4. UX Requirements of an Agentic Architecture

An Agentic GraphRAG system should not behave like a generic single-response chatbot. The frontend should show the teacher what the system is doing and why the generated lesson can be trusted.

The frontend should ideally expose:

- **Planner state:** what the system understood from the teacher request.
- **Retriever state:** what concepts, sources, and Knowledge Graph areas were used.
- **Coverage signal:** whether the answer is strongly KG-grounded or relies more on general AI knowledge.
- **Writer progress:** progressive generation of the lesson plan, preferably with token streaming.
- **Critic verdict:** whether the output was reviewed, approved, or revised.
- **Media/source panel:** articles, videos, OER, and relevant external references.
- **Failure states:** low coverage, no relevant KG concepts, timeout, or generation error.

If the UI only shows the final lesson plan as one chat bubble, the system still works technically, but most of the value of the multi-agent architecture is hidden from the teacher.

---

## 5. UX Comparison

| Dimension | Standalone Agentic GraphRAG WebUI | Native AixLearning Frontend |
|---|---|---|
| Primary purpose | Internal pilot and direct validation of the full agent pipeline | Long-term production experience inside the existing platform |
| Ownership | FEM AI Team | AixLearning DEV/Product team |
| Best use | Experimenting with agentic UX, trust signals, streaming, and domain-expert testing | Integrating the mature agent service into real teacher workflows |
| Agent explainability | Strong fit: can show Planner, Retriever, Writer, Critic cards directly | Depends on DEV adding new Django template sections |
| Streaming UX | Already aligned with SSE and htmx fragments | Possible through existing htmx/Mercure patterns |
| User/account integration | Separate WebUI accounts | Native AixLearning accounts and sessions |
| Classroom/profile integration | Uses its own profile form and persistence | Best fit long-term because it can reuse existing Django models |
| Speed to deploy | Fastest for the internal pilot | Slower because it depends on DEV roadmap, review, and release cadence |
| Long-term teacher adoption | Useful for pilot and expert validation | Strongest, because teachers remain in the platform they already use |
| Migration path | Jinja templates can inspire/guide Django template implementation | Destination frontend once the agentic UX pattern is validated |

---

## 6. Recommendation

The recommended approach is a two-step path:

1. **Use the standalone WebUI for the internal pilot.**  
   This gives the AI Team full control over the agentic UX, makes it easy to test with FEM domain experts, and allows rapid iteration on Planner/Retriever/Writer/Critic presentation without waiting for AixLearning release cycles.

2. **Move the best UX patterns into AixLearning once validated.**  
   The native Django frontend should eventually expose at least the minimum trust signals: Retriever coverage, Critic verdict, streaming progress, and source/media panel.

This avoids two risks:

- launching too early inside AixLearning with a single final-answer bubble that hides the value of the agentic system;
- building a completely separate SPA stack that would be expensive to maintain and harder to integrate later.

The current mirror-stack decision is therefore strategically sound: our WebUI uses FastAPI/Jinja2/htmx/WebAwesome/Tailwind, while AixLearning uses Django/templates/htmx/WebAwesome/Mercure. The paradigms are close enough that the standalone WebUI can act as a UX laboratory and later be translated into native Django templates without a full rewrite.

---

## 7. Practical DEV Implication

For a first native AixLearning integration, DEV can call `POST /api/v1/agent/run` or `POST /api/v1/agent/stream` and display the final lesson plan in the existing chat UI.

However, for the full Agentic GraphRAG value, a second UI iteration should add:

- a Planner/progress section;
- a Retriever coverage indicator;
- a Critic approval/revision signal;
- a source/media sidebar;
- clear AI-generated content marking.

These are incremental Django template additions, not a full frontend rewrite.

---

## 8. Conclusion

The standalone WebUI is the best frontend for the **pilot phase** because it exposes and tests the agentic architecture directly.

The native AixLearning frontend is the best frontend for the **long-term product phase** because it owns the real teacher workflow, accounts, classroom data, and platform experience.

The important principle is: the final UX should not reduce Agentic GraphRAG to a single chatbot bubble. The frontend must surface the reasoning and quality-control signals that make the architecture valuable.
