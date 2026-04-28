# Agent Mode ↔ Domain Config Prompt Integration

**Author:** AI Team (Louis Mono)
**Date:** January 2026
**Status:** Design Document — Decision Pending

---

## Problem Statement

The Agent Mode and GraphRAG Mode maintain **two independent prompt systems** that never communicate:

| | GraphRAG Mode | Agent Mode |
|---|---|---|
| **Domain expertise** | `domains/udl_domain.py` → `get_system_prompt()` (rich, 200+ lines) | `agent/configs/domain_prompts.py` → hardcoded extensions (UDL: 25 lines, Neuro: 85 lines) |
| **Output format** | `domains/udl_domain.py` → `get_response_template()` (advisory/consultative) | `agent/prompts/writer_prompt.py` → `WRITER_SYSTEM_PROMPT_LESSON` (structured lesson plan) |
| **Where used** | `llm_chain.py` via `get_domain_config()` | `writer_agent.py` / `critic_agent.py` via `get_domain_extension()` |

**Consequence:** When Agent Mode generates a UDL lesson plan, the Writer has the KG retrieval data (nodes, relationships, recommendations) but lacks the detailed UDL expert instructions about variability profiles (ADHD, autism, dyslexia, dyscalculia, giftedness, foreign students), checkpoint references, meta-rules, and pedagogical framing that `udl_domain.py` provides.

---

## Current Architecture

### GraphRAG Mode Flow

```
llm_chain.py
    → get_domain_config("udl")
    → domain_config.get_system_prompt()       ← WHO you are (UDL expert identity, 200+ lines)
    → domain_config.get_response_template()   ← HOW to format output (advisory structure)
    → Combined into ChatPromptTemplate with KG context
    → Single LLM call → Advisory response
```

### Agent Mode Flow

```
writer_agent.py
    → WRITER_SYSTEM_PROMPT_LESSON             ← Generic "Educational Content Writer"
    → + get_domain_extension("udl", "writer") ← 25 lines of basic UDL principles
    → + KG retrieval data from RetrieverAgent
    → Single LLM call → Structured lesson plan
```

### What Each Prompt Contains

#### `udl_domain.py` → `get_system_prompt()` (NOT used by Agent)

- Role: "Esperta di Pedagogia Inclusiva e Progettazione Universale per l'Apprendimento"
- TAG-CLOUD with prioritized keywords
- 3 UDL Principles broken down into checkpoints:
  - A. Coinvolgimento (reclutare interesse, sostenere sforzo, autoregolazione, motivazione intrinseca)
  - B. Rappresentazione (percezione, linguaggio/simboli, comprensione)
  - C. Azione ed Espressione (azione fisica, espressione/comunicazione, funzioni esecutive)
- Detailed variability profiles:
  - ADHD: attenzione, controllo inibitorio, memoria di lavoro, sensibilità alla ricompensa
  - Spettro autistico: teoria della mente, pragmatica linguistica, sensibilità sensoriale, flessibilità cognitiva
  - Discalculia: comprensione quantità, ragionamento spaziale, recupero fatti matematici
  - Dislessia: carico cognitivo lettura, monitoraggio metacognitivo, efficacia accademica
  - Plusdotazione: ritmo rapido, curiosità intensa, rischio noia, sensibilità emotiva
  - Studenti stranieri: barriere linguistiche, differenze culturali, vocabolario accademico
- Methodological context: CAST framework, scaffolding, ZPD, cooperative learning, digital/analog tools, authentic assessment
- Meta-rules: personalize, link to UDL principles, distinguish SUGGESTS vs NO_SUGGESTS, use KG data as priority source

#### `udl_domain.py` → `get_response_template()` (NOT used by Agent)

8-section advisory format:
1. Introduzione Empatica
2. Analisi della Variabilità (caratteristiche, barriere, manifestazioni comportamentali)
3. Strategie Raccomandate (linked to Principi UDL A/B/C, with digital + analog tools)
4. Approcci da Evitare (from NO_SUGGESTS relationships)
5. Strategie di Mitigazione
6. Valutazione Inclusiva
7. Ordine di Implementazione
8. Note sulla Fiducia

#### `neuro_domain.py` → `get_system_prompt()` (NOT used by Agent)

Similar structure for neuroscience: cognitive processes (attention, memory, executive functions), emotional/motivational factors, learning strategies, neuroplasticity principles.

#### `neuro_domain.py` → `get_response_template()` (NOT used by Agent)

7-section advisory format:
1. Introduzione Empatica
2. Metodologie Principali (base neuroscientifica, implementazione, adattamenti)
3. Schema Lezione **(se pertinente)**: Warm-up → I Do → We Do → You Do
4. Consolidamento (metacognitive questions, spaced repetition)
5. Basi Teoriche
6. Ordine di Implementazione
7. Note sulla Fiducia

#### `domain_prompts.py` → `NEURO_WRITER_EXTENSION` (CURRENTLY used by Agent)

85 lines covering: working memory limits, attention spans, "I Do/We Do/You Do" model, scaffolding, ZPD, metacognition, lesson structure (Before/During/After Teaching), assessment guidelines (MCQ format).

#### `domain_prompts.py` → `UDL_WRITER_EXTENSION` (CURRENTLY used by Agent)

25 lines covering: 3 UDL principles at surface level, basic accessibility requirements. **No variability profiles, no checkpoints, no tools, no meta-rules.**

#### Agent Writer's own prompt (`WRITER_SYSTEM_PROMPT_LESSON`)

Generic lesson plan format:
- Titolo, Livello, Durata
- Obiettivi di Apprendimento (2-4 SMART)
- Materiali Necessari
- Introduzione (X minuti)
- Attività 1: [Nome] (X minuti) — step by step
- Attività 2: [Nome] (X minuti)
- Valutazione
- Differenziazione (per studenti con difficoltà / per studenti avanzati)
- Fonti

---

## Output Comparison: Advisory vs Lesson Plan

| Aspect | GraphRAG Output (advisory) | Agent Output (lesson plan) |
|--------|---------------------------|---------------------------|
| **Purpose** | "Here's what the research says and how to apply it" | "Here's a lesson you can print and follow" |
| **Timing** | No per-section timing | Every section has timing (X minuti) |
| **Materials** | Not listed | Explicit materials list |
| **Grade level** | Implicit | Explicit |
| **Activities** | Described conceptually | Step-by-step with instructions |
| **Assessment** | "Valutazione inclusiva" (general) | Specific assessment instrument |
| **Differentiation** | Woven throughout | Dedicated section |
| **Audience** | Teacher seeking guidance | Teacher ready to implement |

---

## Proposed Solutions

### Option 1: Agent Produces Same Output as GraphRAG

**Approach:** Replace Agent Writer prompts entirely with `get_system_prompt()` + `get_response_template()` from domain configs.

**Changes:**
- `writer_agent.py`: replace `WRITER_SYSTEM_PROMPT_LESSON` with `domain_config.get_system_prompt()`
- `writer_agent.py`: use `domain_config.get_response_template()` as output instructions
- `domain_prompts.py`: no longer needed for writer

**Pros:**
- Single source of truth
- Zero prompt maintenance
- Guaranteed consistency between modes

**Cons:**
- Loses structured lesson plan format (timing, materials, activities)
- Agent becomes "better GraphRAG" rather than a different product
- Multi-intent support (definition, comparison, etc.) would all get the same advisory template
- Undermines the Agent mode's value proposition

**Effort:** ~2h
**Risk:** Low (technical), High (product — loses differentiation)
**Recommendation:** Not recommended unless you decide both modes should produce the same output.

---

### Option 2: Agent Keeps Own Format, Absorbs Domain Expertise (Quick Win)

**Approach:** Load `get_system_prompt()` ONLY from domain configs into the Agent Writer. Keep the Agent's own lesson plan output format. Do NOT load `get_response_template()`.

**Changes (1 file only):**

```python
# agent/configs/domain_prompts.py — modified get_domain_extension()

try:
    from domains import get_domain_config
    DOMAIN_CONFIG_AVAILABLE = True
except ImportError:
    DOMAIN_CONFIG_AVAILABLE = False

def get_domain_extension(domain: str, agent: str) -> str:
    if domain == "all" or not domain:
        return ""
    
    if DOMAIN_CONFIG_AVAILABLE and agent == "writer":
        domain_config = get_domain_config(domain)
        if domain_config:
            system_prompt = domain_config.get_system_prompt()
            return (
                f"\n\n## Domain Expert Knowledge ({domain.upper()})\n\n"
                f"{system_prompt}"
            )
    
    # Fallback to hardcoded extensions (always used for critic)
    domain_exts = DOMAIN_EXTENSIONS.get(domain.lower(), {})
    return domain_exts.get(agent.lower(), "")
```

**What the Agent Writer receives:**
1. Base: "You are an Educational Content Writer. Format as lesson plan with Title, Duration, Objectives, Activities with timing, Assessment, Differentiation..."
2. + Domain expertise: "You are a UDL expert. You know about ADHD (inhibitory control, working memory), autism (theory of mind, sensory sensitivity), the 3 UDL Principles, SUGGESTS vs NO_SUGGESTS, checkpoint references..."
3. + KG retrieval data (nodes, relationships, recommendations)

**Pros:**
- Only 1 file changed (`domain_prompts.py`)
- Zero changes to `writer_agent.py`, `critic_agent.py`, domain configs
- Graceful fallback if `domains/` import fails
- Agent keeps its differentiated lesson plan output
- UDL Writer immediately gets rich variability profiles, meta-rules
- Critic keeps its own Agent-specific evaluation criteria (unaffected)

**Cons:**
- Combined prompt may be long (~400+ lines system + domain)
- For Neuro: potential overlap between hardcoded extension (85 lines) and `neuro_domain.py`'s system prompt
- Response template expertise (advisory structure) is not used

**Handling Neuro overlap:**
- Option A: For neuro, keep hardcoded extension (it was written specifically for lesson generation). Only dynamically load for UDL.
- Option B: Always dynamically load for all domains (Neuro gets both, LLM synthesizes).

**Effort:** ~1h
**Risk:** Very Low
**Recommendation:** Implement this first as a quick win.

---

### Option 3: Domain-Specific Lesson Plan Templates (Clean Architecture)

**Approach:** Add a new method `get_lesson_plan_template()` to `BaseDomainConfig`. Each domain defines how a lesson plan should be structured within its pedagogical framework. The Agent Writer uses `get_system_prompt()` for expertise and `get_lesson_plan_template()` for output format.

**Changes:**
1. `domains/base_config.py`: add `get_lesson_plan_template()` (with default fallback)
2. `domains/neuro_domain.py`: implement `get_lesson_plan_template()` (Neuro lesson structure)
3. `domains/udl_domain.py`: implement `get_lesson_plan_template()` (UDL lesson structure)
4. `agent/configs/domain_prompts.py`: load both from domain config

#### Neuro Lesson Plan Template

```markdown
# [TITOLO LEZIONE]

**Livello:** [scuola primaria / secondaria I grado / secondaria II grado]
**Durata:** [X minuti]
**Dominio:** Neurodidattica

## Obiettivi di Apprendimento (SMART)
- Obiettivo 1
- Obiettivo 2

## Materiali Necessari
- [elenco]

## Pre-Teaching: Preparazione (prima della lezione)
- Conoscenze pregresse da attivare
- Valutazione allineata con gli obiettivi

## Fase 1: Warm-up / Gancio (5 min)
- Attivazione conoscenze pregresse
- Connessione emotiva / fattore sorpresa
- Collegamento con lezioni precedenti (consolidamento)
- Domanda guida (obiettivo di apprendimento come domanda)

## Fase 2: I Do — Io Faccio (X min)
- Presentazione segmentata del nuovo materiale
- Chunking (3-7 elementi)
- Analogie e metafore
- Dual coding (visivo + verbale)

## Fase 3: We Do — Facciamo Insieme (X min)
- Pratica guidata collaborativa
- Feedback immediato e formativo
- Scaffolding con rilascio graduale

## Fase 4: You Do — Fai Tu (X min)
- Applicazione autonoma
- Differenziazione didattica
- Opzioni per bisogni speciali

## Consolidamento (5 min)
- Attività di chiusura
- Autovalutazione dello studente
- 2 domande metacognitive

## Piano di Spaced Repetition
- Momento 1 (giorno +3): [attività 5-10 min]
- Momento 2 (giorno +7): [attività 5-10 min]
- Momento 3 (giorno +14): [attività 5-10 min]
- Momento 4 (giorno +28): [attività 5-10 min]

## Valutazione
- [strumento di valutazione formativa]
- [3-10 domande a scelta multipla se applicabile]

## Differenziazione
- **Per studenti con difficoltà:** [adattamenti]
- **Per studenti avanzati:** [estensioni]

---
*Fonti: [metodologie utilizzate dal Knowledge Graph]*
```

#### UDL Lesson Plan Template

```markdown
# [TITOLO LEZIONE]

**Livello:** [scuola primaria / secondaria I grado / secondaria II grado]
**Durata:** [X minuti]
**Dominio:** Universal Design for Learning
**Variabilità target:** [ADHD / spettro autistico / dislessia / discalculia / plusdotazione / studenti stranieri]

## Obiettivi di Apprendimento
- Obiettivo 1
- Obiettivo 2

## Analisi della Variabilità
- Caratteristiche specifiche degli studenti
- Barriere potenziali (sensoriali, linguistiche, cognitive, esecutive, tecnologiche)
- Manifestazioni comportamentali osservabili

## Materiali e Strumenti
| Strumento | Tipo | Per quale variabilità | Principio UDL |
|-----------|------|----------------------|---------------|
| | Digitale / Analogico | | A / B / C |

## Fase 1: Coinvolgimento — Principio A (X min)
*"Perché dell'apprendimento"*
- Reclutamento dell'interesse: [strategia]
- Sostenere sforzo e persistenza: [strategia]
- Opzioni per l'autoregolazione: [strategia]
- Gestione della motivazione intrinseca: [strategia]

## Fase 2: Rappresentazione — Principio B (X min)
*"Cosa dell'apprendimento"*
- Opzioni per la percezione (visivo, uditivo, tattile): [strategia]
- Opzioni per il linguaggio e i simboli: [strategia]
- Opzioni per la comprensione (conoscenze pregresse, pattern): [strategia]

## Fase 3: Azione ed Espressione — Principio C (X min)
*"Come dell'apprendimento"*
- Opzioni per l'azione fisica: [strategia]
- Opzioni per l'espressione e la comunicazione: [strategia]
- Opzioni per le funzioni esecutive (pianificazione, monitoraggio): [strategia]

## Approcci da Evitare
- [Approcci che il KG indica come NON raccomandati — da relazioni NO_SUGGESTS]
- Motivazione pedagogica per ciascuno

## Strategie di Mitigazione
| Sfida prevedibile | Strategia di mitigazione | Strumento |
|-------------------|--------------------------|-----------|
| | | |

## Valutazione Inclusiva
- Modalità multiple di espressione
- Valutazione autentica (task, progetti, performance)
- Opzioni flessibili di valutazione

## Differenziazione per Variabilità
- **ADHD:** [supporto esecutivo esternalizzato, pause movimento, cueing di iniziazione]
- **Spettro autistico:** [struttura prevedibile, supporti per pragmatica, gestione sensoriale]
- **Dislessia:** [riduzione carico cognitivo, formati alternativi, supporti metacognitivi]
- **Plusdotazione:** [compiti aperti, approfondimenti, mentoring]
- **Studenti stranieri:** [scaffolding linguistico, translanguaging, vocabolario accademico]

---
*Fonti: [metodologie e nodi del Knowledge Graph utilizzati]*
*Principi UDL di riferimento: [A / B / C]*
```

**Implementation:**

```python
# domains/base_config.py — add new method
def get_lesson_plan_template(self) -> str:
    """
    Return domain-specific lesson plan template for Agent Mode.
    
    Unlike get_response_template() (advisory format for GraphRAG),
    this returns a structured lesson plan format with timing,
    materials, and step-by-step activities.
    
    Override in domain subclasses.
    Default: generic lesson plan structure.
    """
    return """(generic fallback template)"""

# agent/configs/domain_prompts.py — use both
def get_domain_extension(domain: str, agent: str) -> str:
    if DOMAIN_CONFIG_AVAILABLE and agent == "writer":
        domain_config = get_domain_config(domain)
        if domain_config:
            expertise = domain_config.get_system_prompt()
            lesson_template = domain_config.get_lesson_plan_template()
            return (
                f"\n\n## Domain Expert Knowledge\n\n{expertise}"
                f"\n\n## Lesson Plan Structure\n\n{lesson_template}"
            )
    # Fallback to hardcoded
    ...
```

**What each mode uses from domain configs:**

| Method | GraphRAG Mode | Agent Mode |
|--------|--------------|------------|
| `get_system_prompt()` | YES (via `llm_chain.py`) | YES (via `domain_prompts.py`) |
| `get_response_template()` | YES (advisory format) | NO |
| `get_lesson_plan_template()` | NO | YES (lesson format) |

**Pros:**
- Single source of truth: all domain knowledge in `domains/`
- Each mode has purpose-specific output format
- Domain-specific lesson structures (Neuro follows neurodidactic model, UDL follows 3-Principle framework)
- Adding a new domain = 1 file (the domain config), works in both modes
- Existing `domain_prompts.py` hardcoded extensions become unnecessary (can be removed)

**Cons:**
- Requires changes to 4 files
- Must test both modes after changes
- Longer prompts (but GPT-4o handles 128K context)

**Effort:** ~4h
**Risk:** Low (additive — new method with default fallback, nothing breaks)
**Recommendation:** Best long-term solution. Implement after Option 2 is validated.

---

## Migration Path

```
Current State
    ├── GraphRAG: uses domains/ configs          ✅ working
    └── Agent: uses agent/configs/ (isolated)    ✅ working but thin UDL

Step 1: Option 2 (Quick Win) — ~1h
    ├── GraphRAG: unchanged
    └── Agent Writer: loads get_system_prompt() from domains/
        └── Critic: unchanged (keeps hardcoded evaluation criteria)

Step 2: Option 3 (Clean Architecture) — ~4h
    ├── Add get_lesson_plan_template() to base_config.py
    ├── Implement in neuro_domain.py (I Do/We Do/You Do structure)
    ├── Implement in udl_domain.py (3-Principle structure)
    ├── Update domain_prompts.py to load both
    └── Remove hardcoded NEURO_WRITER_EXTENSION / UDL_WRITER_EXTENSION

Final State
    ├── GraphRAG: get_system_prompt() + get_response_template()    (advisory)
    └── Agent:    get_system_prompt() + get_lesson_plan_template()  (lesson plan)
    └── Both share domain expertise, differ only in output format
```

---

## Files Affected

| File | Option 1 | Option 2 | Option 3 |
|------|----------|----------|----------|
| `agent/configs/domain_prompts.py` | Major rewrite | 1 function change | 1 function change |
| `agent/agents/writer_agent.py` | Major rewrite | No change | No change |
| `agent/agents/critic_agent.py` | No change | No change | No change |
| `domains/base_config.py` | No change | No change | Add 1 method |
| `domains/neuro_domain.py` | No change | No change | Add 1 method |
| `domains/udl_domain.py` | No change | No change | Add 1 method |
| `llm_chain.py` (GraphRAG) | No change | No change | No change |
| `agent/prompts/writer_prompt.py` | Remove | No change | No change |

---

## Decision Criteria

| If you want... | Choose |
|----------------|--------|
| Fastest implementation, minimal risk | **Option 2** |
| Same output from both modes | Option 1 |
| Domain-specific lesson plans, clean architecture | **Option 3** |
| Both modes share expertise but have different purposes | Option 2 → Option 3 |

---

*Document generated by AI Team — for internal architectural decision*
