# AI Evaluation Metrics Strategy — Agentic GraphRAG

**Date:** 2 August 2026  
**Status:** Proposed strategy; implementation not started  
**Audience:** FEM Direction, AI Team, AixLearning DEV Team, domain experts,
Product, and pilot coordinators  
**Scope:** Agentic GraphRAG pipeline
`Planner → Retriever → Writer → Critic`, including WebUI, API and MCP usage  

---

## 1. Executive summary

The Agentic GraphRAG system is not completely without evaluation. It already
contains:

- an online Critic Agent that scores generated lessons and can request
  revisions;
- heuristic retrieval confidence and coverage indicators;
- an optional Corrective RAG grader;
- a `MetricsCalculator` with heuristic context relevance and faithfulness
  calculations;
- persisted run metadata such as approval, revisions, node counts and duration.

However, these components do not yet form a complete evaluation system.

The principal gaps are:

- no versioned, expert-reviewed golden dataset;
- no systematic offline benchmark for prompt/model/KG changes;
- no production calculation of standard RAG faithfulness, context precision or
  context recall;
- no quality regression thresholds in CI;
- no independent calibration of the Critic against human reviewers;
- no run-level Langfuse traces, token/cost roll-up or evaluation scores;
- no structured teacher feedback or product-outcome metrics;
- incomplete persistence of retrieval and evaluation outcomes.

The recommended strategy is not to select a single “AI quality score”. It is to
combine four complementary evidence sources:

1. **deterministic operational metrics** for every run;
2. **reference-based offline metrics** on a versioned golden dataset;
3. **LLM-as-a-judge rubrics**, calibrated against humans;
4. **human and product outcomes** from teachers and domain experts.

The initial deliverable should be a small but trustworthy evaluation backbone:

- 20 reviewed golden cases, split between Neuro and UDL;
- a framework-neutral evaluation dataset and result schema;
- deterministic run telemetry;
- retrieval Precision/Recall using expected Neo4j concept IDs;
- offline faithfulness, context quality and pedagogical scoring;
- advisory regression reports before any metric becomes a blocking release
  gate.

Ragas, DeepEval and Langfuse can all contribute, but they serve different
roles:

- **Ragas:** RAG-specific offline metrics;
- **DeepEval:** pytest-like evaluation, agent/tool checks and custom G-Eval
  rubrics;
- **Langfuse:** traces, datasets, experiments, scores and production feedback.

The dataset, metric definitions and acceptance criteria must remain owned by
FEM, rather than being defined implicitly by one framework.

---

## 2. Objectives and non-objectives

### 2.1 Objectives

The evaluation program must:

1. detect regressions before prompt, model, KG or retrieval changes reach
   production;
2. distinguish retrieval failures from generation failures;
3. measure pedagogical usefulness, not only linguistic quality;
4. provide comparable evidence across domains, models and releases;
5. support both offline experiments and sampled production monitoring;
6. preserve enough run context to diagnose failures;
7. include human calibration and teacher outcomes;
8. remain cost-controlled and compatible with a multi-worker deployment;
9. support Italian and English requests;
10. respect GDPR, data-minimization and retention requirements.

### 2.2 Non-objectives

The first implementation should not:

- reduce quality to one composite score;
- run expensive LLM judges on every production request;
- make all quality metrics blocking immediately;
- treat BLEU, ROUGE or keyword overlap as sufficient for open-ended lessons;
- send unredacted teacher/student information to external evaluation services;
- replace expert pedagogical review;
- lock FEM into Ragas, DeepEval, Langfuse or one model provider.

---

## 3. Current-state audit

### 3.1 Production pipeline

The current agent flow is:

```text
Planner
   ↓
Retriever
   ↓
Optional Corrective RAG grader
   ↓
Writer
   ↓
Critic
   ├── REVISE → Writer
   └── APPROVE → final output
```

The key implementation locations are:

- `src/aix/agent/agents/planner_agent.py`;
- `src/aix/agent/agents/retriever_agent.py`;
- `src/aix/agent/agents/retrieval_grader_agent.py`;
- `src/aix/agent/agents/writer_agent.py`;
- `src/aix/agent/agents/critic_agent.py`;
- `src/aix/agent/graph/nodes.py`;
- `src/aix/webui/agent/service.py`.

### 3.2 Existing evaluation mechanisms

| Mechanism | Current behavior | Runtime status | Main limitation |
|---|---|---|---|
| Critic Agent | Scores structure, evidence, pedagogy, practicality and differentiation on 1–5 | Every agent run | It belongs to the same generation system and is not independently calibrated |
| Revision loop | Requests a rewrite when Critic thresholds fail | Every agent run | Maximum revisions can force completion; approval does not guarantee correctness |
| Retrieval confidence | Produces qualitative confidence tiers | Every retrieval | Heuristic and not calibrated against expected concepts |
| Coverage tier | Classifies retrieval as healthy, limited or out-of-scope | WebUI run | Mainly based on counts, not concept relevance |
| Corrective RAG grader | Relevant/ambiguous/irrelevant judgment and optional rewrite | Feature flag; off by default | LLM judge failures degrade to permissive behavior |
| UDL coverage badge | Counts nodes associated with three UDL principle families | UDL WebUI runs | Label-count heuristic, not a pedagogical UDL rubric |
| `MetricsCalculator` | Context relevance, faithfulness, query complexity, graph coverage | Streamlit path only | Not integrated into production agent; metrics are custom heuristics |
| Run metadata | Approval, scores, revisions, nodes, media and duration | Persisted partially | Missing stage latency, retrieval grade, parse errors, tokens, cost and evaluator versions |
| Sentry/GlitchTip | Exceptions and sampled performance traces | Optional production integration | Reliability observability, not content-quality evaluation |
| Langfuse | Fetches versioned prompts | Production prompt loading | No agent traces, observations, experiment runs or quality scores |

### 3.3 Important interpretation

Several existing names resemble standardized RAG metrics, but they must not be
reported as equivalent without qualification:

- current `context_relevance` is a keyword/embedding heuristic;
- current `faithfulness` can use token overlap or a custom LLM judge;
- `retrieval_confidence` is an internal tier, not calibrated probability;
- `UDL coverage` counts labels, not instructional quality;
- Critic scores are useful pipeline signals, not independent ground truth.

The documentation should distinguish:

```text
project heuristic
standard framework metric
human-reviewed metric
production KPI
```

### 3.4 Persistence available today

The WebUI currently provides two useful persistence points:

- `Lesson`: status, error, domain, profile and final lesson;
- `LessonMessage.meta_json`: approval, revisions, Critic scores, node/media
  counts, search-query count, duration and UDL coverage.

This is sufficient for an MVP operational dashboard, but not for a complete
evaluation history.

### 3.5 Missing assets

No current repository asset provides:

- a golden query/lesson dataset;
- expert relevance labels for retrieved concepts;
- expected planner intent or scope labels;
- expected tool calls;
- reference pedagogical rubrics;
- a Ragas evaluation runner;
- a DeepEval test suite;
- CI quality thresholds;
- paired experiment reports;
- structured teacher ratings.

---

## 4. Evaluation design principles

### 4.1 Evaluate the failure surface, not only the final answer

A final lesson can be weak because:

- Planner misunderstood the intent;
- domain/scope classification was wrong;
- Text2Cypher generated invalid Cypher;
- retrieval missed relevant concepts;
- irrelevant concepts polluted the context;
- Writer ignored good context;
- Writer introduced unsupported claims;
- Critic failed to detect the problem;
- the lesson is correct but impractical for the teacher.

Metrics must identify which stage failed.

### 4.2 Use multiple evidence types

| Evidence | Best use | Strength | Limitation |
|---|---|---|---|
| Deterministic checks | Reliability, schema, IDs, timing, counts | Cheap and reproducible | Cannot judge rich pedagogical quality |
| Reference-based metrics | Retrieval and expected behavior | Explainable and stable | Requires maintained golden data |
| LLM-as-a-judge | Faithfulness, relevance, pedagogy, safety | Scales to open-ended output | Judge bias, cost and version drift |
| Human review | Pedagogy and real usefulness | Highest domain validity | Slow, expensive and potentially inconsistent |
| Product behavior | Acceptance, export, edits, follow-up | Reflects real usage | Indirect and affected by UX factors |

### 4.3 Baseline before gating

New metrics should initially be:

1. collected;
2. reviewed for data quality;
3. compared against human judgments;
4. segmented by domain/language/query type;
5. used as advisory warnings;
6. promoted to release gates only after stability is demonstrated.

### 4.4 Prefer paired comparisons

For prompt, model and retrieval changes, compare candidate and baseline on the
same dataset items. Paired differences are more useful than comparing two
unrelated aggregate scores.

### 4.5 Version everything

Every evaluation result should identify:

- pipeline commit/SHA;
- dataset name and version;
- prompt names and versions;
- model/provider configuration;
- embedding model;
- KG snapshot/version;
- evaluator framework and version;
- judge model and rubric version;
- metric implementation version.

Without this metadata, scores cannot be reproduced.

### 4.6 No single composite score

A weighted “AI Quality Score” can conceal critical failures. A result with
excellent style but unsupported scientific claims must not pass because its
average is high.

Use:

- a scorecard of independent dimensions;
- explicit critical-failure rules;
- aggregates only for reporting, never as the sole release criterion.

---

## 5. Metric framework

## 5.1 System reliability and operational metrics

These metrics should be deterministic and collected for every run.

### Run success rate

```text
completed runs / all started runs
```

Segment by:

- domain;
- endpoint (WebUI/API/MCP);
- model;
- pipeline version;
- new lesson vs follow-up;
- error category.

### Stage success rate

Track success/failure separately for:

- Planner;
- retrieval/Text2Cypher;
- Corrective RAG;
- Writer;
- Critic;
- media enrichment;
- persistence/streaming.

### Parse-error rate

```text
parse errors for agent X / invocations of agent X
```

Current structured `agent_parse_error` logs should become persisted counters.
Critic parse errors must not be silently included in approval-rate reporting.

### Latency

Record:

- queue wait;
- Planner duration;
- retrieval duration;
- Writer time-to-first-token;
- Writer total duration;
- Critic duration;
- end-to-end p50, p90 and p95;
- follow-up vs initial-run duration.

### Availability and load

- concurrent-run utilization;
- queue timeout rate;
- rate-limit rejection rate;
- stale-run recovery count;
- database/checkpointer failures;
- Neo4j availability;
- external API failure rate.

### Token and cost metrics

Per run and stage:

- input tokens;
- output tokens;
- cached tokens where available;
- model/provider;
- estimated cost;
- cost per successful lesson;
- judge-evaluation cost;
- cost by domain and user cohort.

### Initial pilot targets

| Metric | Provisional target |
|---|---:|
| Run success rate | ≥95% |
| Parse-error rate per agent | <2% |
| p50 end-to-end duration | <90 seconds |
| p95 end-to-end duration | <180 seconds |
| Critical persistence/data-loss failures | 0 |

Targets are provisional until real staging and pilot baselines exist.

---

## 5.2 Retrieval and GraphRAG metrics

Retrieval evaluation must distinguish quantity, relevance, completeness,
ranking and graph behavior.

### Domain-routing accuracy

```text
correct domain decisions / labeled routing cases
```

Dataset labels should include:

- `neuro`;
- `udl`;
- `all/cross-domain`;
- `out_of_scope`.

### Scope-classification accuracy

Measure whether the Planner correctly identifies:

- in-scope educational-methodology requests;
- partially supported requests;
- requests outside the available KG;
- ambiguous requests requiring clarification.

### Cypher validity rate

```text
Cypher statements passing validation / generated Cypher statements
```

Also record:

- repair rate;
- fallback rate;
- invalid-label rate;
- syntax-error rate;
- zero-result rate;
- generation retries.

### Retrieval availability

```text
in-scope queries retrieving at least one relevant concept / in-scope queries
```

### Healthy coverage rate

Current WebUI logic uses a threshold such as five nodes. Keep this operational
signal, but do not treat it as relevance.

```text
in-scope runs with nodes_count ≥ configured threshold / in-scope runs
```

### ID-based Precision@K

When expected Neo4j concept IDs are labeled:

```text
relevant concept IDs in top K / K retrieved concept IDs
```

This is deterministic and should be the primary initial retrieval metric.

### ID-based Recall@K

```text
expected concept IDs retrieved in top K / all expected concept IDs
```

Recall needs a reviewed denominator. Do not calculate it from an incomplete
reference list.

### Mean Reciprocal Rank

For cases where one principal concept is expected:

```text
MRR = mean(1 / rank of first expected concept)
```

### nDCG@K

Use normalized discounted cumulative gain when experts assign graded relevance
such as:

- 3 = essential;
- 2 = relevant;
- 1 = useful context;
- 0 = irrelevant.

This captures both relevance and ranking.

### Context precision

Measures whether useful context is ranked above irrelevant context. It may be:

- deterministic using expected concept IDs;
- LLM-based using a reference response/context;
- response-conditioned when no reference exists.

The metric implementation type must be recorded.

### Context recall

Measures whether retrieval contains all information needed by the reference.
It requires:

- reference claims;
- reference contexts;
- or expected document/concept IDs.

### Context relevance

Measures direct query-context alignment. It is useful without a reference, but
it is not the same as context precision or recall.

### Graph diversity and coverage

Possible GraphRAG-specific diagnostics:

- distinct labels/categories represented;
- duplicate concept rate;
- direct vs neighbor-expanded evidence;
- relationship-type distribution;
- average and maximum hop distance;
- isolated-node retrieval;
- proportion of cross-domain nodes;
- recommendation/evidence ratio.

These are diagnostics, not standalone quality judgments.

### Corrective RAG effectiveness

When enabled:

```text
quality after rewrite - quality before rewrite
```

Track:

- grader decision distribution;
- rewrite frequency;
- retrieval improvement after retry;
- unnecessary retries;
- grader parse/fallback rate.

### Suggested starting targets

| Metric | Advisory target |
|---|---:|
| Domain routing accuracy | ≥90% |
| Cypher validation success | ≥98% |
| Retrieval availability for in-scope cases | ≥90% |
| Healthy coverage (`nodes ≥ 5`) | ≥70% |
| ID Precision@5 | ≥0.70 |
| ID Recall@10 | ≥0.70 |
| Context precision | ≥0.60 |
| Context recall | ≥0.70 |

These thresholds must be recalibrated by domain and query slice.

---

## 5.3 Generation and grounding metrics

### Faithfulness

Faithfulness answers:

> Are factual claims in the generated response supported by the retrieved
> context?

Typical formulation:

```text
supported response claims / all evaluated response claims
```

It does not establish that the retrieved context itself is true. It measures
consistency with the evidence supplied to the Writer.

### Unsupported-claim severity

Not all unsupported claims are equally important. Classify:

- critical: scientific/safety/legal claim;
- major: pedagogical recommendation presented as established evidence;
- minor: harmless elaboration or transition.

Release gates should focus on critical and major failures, not only averages.

### Answer relevance

Measures whether the final output addresses the teacher request without
irrelevant drift.

### Completeness

For lesson-generation intents, check required sections:

- objectives;
- prerequisites/context;
- timing;
- materials;
- activities;
- assessment;
- differentiation/accessibility;
- evidence or recommendations;
- next steps where required.

Completeness should combine deterministic section checks with rubric scoring.

### Instruction adherence

Verify:

- requested language;
- class/age;
- duration;
- subject/topic;
- pedagogical intent;
- requested output type;
- refinement instruction;
- forbidden or explicitly excluded content.

### Citation/evidence correctness

When citations are present:

- citation exists;
- URL/DOI format valid;
- cited source supports the associated claim;
- citation belongs to retrieved/approved evidence;
- no fabricated source metadata.

### Consistency

- objectives align with activities;
- activities align with assessment;
- timing sums are realistic;
- difficulty matches class/age;
- recommendations do not contradict one another.

### Language quality

- Italian/English compliance;
- clarity;
- terminology consistency;
- teacher-facing tone;
- absence of unnecessary verbosity.

### Format/schema validity

- Markdown structure;
- JSON schema validity where structured output is required;
- exports render correctly;
- no truncated sections;
- AI disclosure marker remains present.

### Suggested starting targets

| Metric | Advisory target |
|---|---:|
| Faithfulness | ≥0.75 |
| Answer relevance | ≥0.75 |
| Required-section completeness | ≥95% |
| Language compliance | ≥98% |
| Critical unsupported claims | 0 |
| Critic average | ≥3.5/5 |
| Runs requiring revision | <30% |

Framework scores are not interchangeable. A Ragas score of `0.75` and a
DeepEval score of `0.75` must not be assumed equivalent without calibration.

---

## 5.4 Agent and tool-behavior metrics

### Planner intent accuracy

Compare predicted intent and lesson type to expert labels.

### Plan completeness

Check required Planner fields:

- query intent;
- confidence;
- scope status;
- key concepts;
- search queries;
- lesson type;
- language;
- educational constraints.

### Tool-selection correctness

For each golden case, define:

- required tools;
- permitted optional tools;
- forbidden/unnecessary tools;
- expected call order where relevant.

### Task completion

Determine whether the full user goal was completed, independently of style.

### Retry and revision efficiency

- retrieval retry count;
- Critic revision count;
- score improvement after revision;
- repeated no-improvement loops;
- successful recovery after transient failure.

### Revision effectiveness

```text
post-revision score - pre-revision score
```

Also check whether requested Critic corrections were actually applied.

### State consistency

- no missing required `AgentState` fields;
- no stale retrieval context after rewrite;
- follow-up turn preserves correct conversation context;
- thread/session IDs remain correlated.

### Graceful degradation

Expected behavior when:

- Neo4j is unavailable;
- retrieval returns zero nodes;
- media assets are missing;
- external APIs time out;
- judge output cannot be parsed;
- request is out of scope.

---

## 5.5 Pedagogical and UDL metrics

Generic RAG metrics cannot determine whether a lesson is educationally useful.
A domain-specific pedagogical rubric is required.

### Core pedagogical dimensions

Score each dimension from 1–5 with anchored examples:

1. **Objective clarity:** objectives are specific and observable.
2. **Objective/activity alignment:** activities practice the stated objectives.
3. **Assessment alignment:** assessment measures those objectives.
4. **Learning progression:** sequence moves coherently from activation to
   practice and consolidation.
5. **Age/class suitability:** vocabulary, complexity and autonomy are
   appropriate.
6. **Time feasibility:** activity durations are realistic.
7. **Teacher practicality:** instructions and materials are immediately usable.
8. **Differentiation:** adaptations address learner variability.
9. **Evidence grounding:** pedagogical recommendations connect to retrieved
   methodologies.
10. **Inclusivity/accessibility:** barriers and alternative participation modes
    are considered.

### UDL-specific dimensions

Evaluate quality—not just presence—of:

- multiple means of engagement;
- multiple means of representation;
- multiple means of action and expression;
- learner choice and agency;
- accessibility barriers;
- flexible assessment;
- alignment with the actual learner/context profile.

### Neuro-specific dimensions

Evaluate:

- accuracy of neuroscience claims;
- avoidance of neuromyths;
- translation from evidence to classroom practice;
- appropriateness of cognitive-load recommendations;
- realistic treatment of attention, memory, emotion and executive functions.

### Calibration

The current Critic rubric is a strong starting point but must be calibrated:

1. domain experts score a shared sample;
2. resolve disagreements and improve scale anchors;
3. run the LLM judge on the same sample;
4. measure exact/adjacent agreement and Cohen’s kappa where suitable;
5. revise rubric and few-shot examples;
6. repeat whenever judge model or rubric changes.

---

## 5.6 Safety, privacy and compliance metrics

### Safety

- harmful or inappropriate educational content;
- discriminatory or exclusionary recommendations;
- unsafe classroom activities;
- age-inappropriate content;
- unsupported clinical/medical claims;
- instructions outside the system’s educational scope.

### Privacy

- student names or identifying information in traces;
- secrets/credentials in prompts or outputs;
- uploaded-file PII;
- accidental persistence of sensitive text;
- data sent to external judge providers.

### Compliance

- AI-generated-content disclosure present;
- trace/correlation identifier available;
- configured retention respected;
- consent/legal basis for pilot telemetry;
- external evaluator subprocessors documented;
- human escalation path for critical findings.

### Critical rule

Safety and privacy should use explicit pass/fail gates. They should not be
averaged into general quality.

---

## 5.7 Human and product-success metrics

Automated metrics measure proxies. Pilot success requires teacher evidence.

### Teacher feedback

Minimal UI:

- useful / not useful;
- optional 1–5 rating;
- optional short comment;
- selectable issue categories.

Suggested categories:

- incorrect content;
- irrelevant KG recommendations;
- missing content;
- too generic;
- impractical timing/materials;
- unsuitable for class/age;
- accessibility/UDL weakness;
- formatting/export issue;
- other.

### Behavioral signals

- lesson exported/downloaded;
- lesson saved;
- refinement requested;
- copied or reused profile;
- abandoned run;
- repeated generation for same request;
- time to first accepted result.

These signals are not proof of quality but help interpret ratings.

### Human acceptance criteria

Initial pilot targets:

- ≥80% “useful” or “very useful”;
- ≥70% acceptable without substantial rewrite;
- 0 unresolved critical safety/content errors;
- expert-reviewed sample average ≥3.5/5.

---

## 6. Golden evaluation dataset

## 6.1 Why it is the first strategic asset

Without a versioned golden dataset:

- prompt/model comparisons are subjective;
- retrieval precision/recall have no denominator;
- LLM judges cannot be calibrated;
- CI thresholds cannot be trusted;
- production failures are not converted into regression tests.

The dataset is more important than selecting Ragas or DeepEval.

## 6.2 Initial size

Start with 20 cases:

- 10 Neuro;
- 10 UDL.

Expand to 50–100 cases after pilot failures and real usage patterns become
available.

## 6.3 Required slices

Include:

- Italian and English;
- straightforward in-scope lesson;
- informational explanation;
- comparison request;
- accessibility/special-needs context;
- follow-up refinement;
- sparse-KG case;
- ambiguous request;
- out-of-scope topic;
- cross-domain request;
- malformed/long input;
- query requiring warning or abstention;
- request with explicit duration/class constraints.

## 6.4 Dataset record

Each record should contain:

```json
{
  "id": "neuro-it-001",
  "dataset_version": "1.0.0",
  "domain": "neuro",
  "language": "it",
  "user_input": "Richiesta del docente...",
  "educational_profile": {},
  "expected": {
    "scope_status": "in_scope",
    "intent": "lesson_creation",
    "lesson_type": "full_lesson",
    "concept_ids_required": ["..."],
    "concept_ids_relevant": ["..."],
    "concept_ids_forbidden": ["..."],
    "required_sections": ["objectives", "activities", "assessment"],
    "required_warnings": [],
    "forbidden_claims": []
  },
  "reference_contexts": [],
  "reference_rubric": {
    "pedagogy": {},
    "udl": {},
    "safety": {}
  },
  "tags": ["italian", "lesson", "in_scope"]
}
```

Do not require one exact reference lesson. Open-ended lesson design can have
many valid outputs. Prefer:

- required properties;
- acceptable concept/evidence sets;
- forbidden behavior;
- rubric anchors.

## 6.5 Dataset ownership

| Role | Responsibility |
|---|---|
| AI Team | Dataset schema, technical validation, runners |
| Neuro/UDL experts | Expected concepts, claims and pedagogical rubric |
| Product/pilot coordinator | Representative user scenarios and priority |
| DEV/QA | CI integration and reproducibility |
| Data protection owner | Privacy review and retention |

## 6.6 Dataset lifecycle

1. Draft case from domain expert or production failure.
2. Technical validation against the current KG.
3. Independent review by a second person.
4. Assign version and tags.
5. Run baseline.
6. Record known limitations.
7. Add new regressions without silently rewriting historical expectations.

---

## 7. Offline evaluation architecture

## 7.1 Framework-neutral runner

Create a canonical runner that:

1. loads a dataset version;
2. invokes the same production agent entry point;
3. captures all stage outputs;
4. records model/prompt/KG versions;
5. executes configured evaluators;
6. stores item-level scores and reasons;
7. calculates aggregate and slice-level results;
8. compares candidate vs baseline;
9. generates JSON and human-readable reports;
10. returns a CI result according to the selected policy.

Suggested location:

```text
scripts/eval/run_agent_eval.py
src/aix/evaluation/
data/eval/
tests/eval/
```

## 7.2 Canonical captured fields

- original input/profile;
- Planner output;
- generated and executed Cypher;
- retrieved node IDs, ranks, labels and relationship evidence;
- retrieval grade/rewrite;
- Writer draft/final;
- Critic scores and revisions;
- final output;
- stage durations;
- tokens and costs;
- errors/fallbacks;
- pipeline and evaluator versions.

## 7.3 Result schema

```json
{
  "evaluation_id": "...",
  "dataset_item_id": "neuro-it-001",
  "pipeline_version": "git-sha",
  "dataset_version": "1.0.0",
  "mode": "offline",
  "scores": [
    {
      "name": "id_precision_at_5",
      "value": 0.8,
      "passed": true,
      "evaluator_type": "deterministic",
      "evaluator_version": "1.0.0",
      "reason": null
    }
  ],
  "critical_failures": [],
  "metadata": {}
}
```

## 7.4 Comparison report

Report:

- candidate and baseline means;
- paired difference per case;
- pass/fail counts;
- regressions by domain/language/tag;
- bootstrap confidence interval where practical;
- largest improved/regressed examples;
- judge disagreement and parse failures;
- total evaluation cost.

---

## 8. Online production evaluation

## 8.1 Evaluation tiers

### Tier 1 — every run

Deterministic and inexpensive:

- status/error;
- stage durations;
- model/prompt version;
- tokens/cost;
- parse errors;
- node/relationship counts;
- Cypher validation;
- retrieval outcome;
- revision count;
- output/schema checks;
- AI disclosure;
- user feedback.

### Tier 2 — sampled runs

LLM-as-a-judge:

- faithfulness;
- answer relevance;
- pedagogy/UDL;
- safety.

Suggested initial sample:

- 5% of successful production runs;
- 100% of manually flagged runs;
- 100% of low-confidence/zero-node runs;
- 100% of new prompt/model canary traffic.

### Tier 3 — human review

- random stratified sample;
- all critical automated failures;
- all negative teacher feedback;
- periodic calibration set.

## 8.2 Persistence model

### MVP

Extend `LessonMessage.meta_json` with:

- `average_critic_score`;
- `retrieval_grade`;
- `retrieval_attempts`;
- `retrieval_warning`;
- parse-error flags;
- stage durations;
- token/cost totals;
- prompt/model versions;
- evaluator sample status.

### Recommended durable model

Add normalized tables:

#### `evaluation_run`

- ID;
- lesson/message/trace reference;
- dataset item if offline;
- online/offline mode;
- pipeline/dataset version;
- domain/language;
- start/end time;
- evaluator configuration.

#### `evaluation_score`

- evaluation-run ID;
- metric name;
- numeric/categorical/boolean value;
- threshold and pass/fail;
- evaluator type/version;
- judge model;
- reason;
- metadata;
- creation time.

This preserves framework independence and makes SQL aggregation practical.

## 8.3 Sampling and asynchronous execution

LLM judges should not delay the teacher response.

Recommended sequence:

1. complete and persist the lesson;
2. decide deterministically whether the run should be sampled;
3. enqueue or schedule evaluation;
4. write scores asynchronously;
5. alert only on configured severe outcomes.

For the internal pilot, a lightweight background process or scheduled batch may
be sufficient. A durable queue becomes necessary only when volume and delivery
guarantees justify it.

---

## 9. Instrumentation plan

## 9.1 Agent-state additions

Add or normalize:

- per-stage start/end timestamps;
- parse-error flags;
- model/prompt versions;
- token usage;
- cost;
- retrieval rank/evidence metadata;
- pre/post-revision Critic scores;
- evaluator sampling decision.

Avoid placing secrets or full sensitive inputs in state.

## 9.2 WebUI service

Extend `_extract_meta()` in `src/aix/webui/agent/service.py` to persist:

- average Critic score;
- retrieval grade and attempts;
- coverage tier;
- parse-error flags;
- stage durations;
- total tokens/cost;
- prompt/model versions.

## 9.3 LLM wrapper

Instrument the common LLM-call layer to record:

- agent/stage;
- model/provider;
- latency;
- input/output token counts;
- retries;
- finish reason;
- error type;
- trace/span IDs.

This is more reliable than adding bespoke telemetry separately to each agent.

## 9.4 Langfuse

The current integration fetches prompts only. Add:

- one trace per teacher interaction;
- observations/spans for Planner, retrieval, Writer and Critic;
- session/thread correlation;
- model/token/cost metadata;
- prompt-version linkage;
- evaluation scores attached to traces;
- dataset experiment runs;
- user feedback scores.

Before sending production content:

- determine cloud vs self-hosted Langfuse;
- confirm DPA and data location;
- redact or pseudonymize teacher/student data;
- configure sampling and retention.

## 9.5 API schemas

Align WebUI and API exposure for:

- Critic average score;
- retrieval grade/outcome;
- UDL coverage;
- evaluation status;
- trace ID;
- optional non-sensitive metric summary.

Do not expose internal judge reasoning or sensitive trace content by default.

---

## 10. Ragas

## 10.1 Role

Ragas is an open-source RAG evaluation framework. Current documentation
includes metrics such as:

- Faithfulness;
- Response/Answer Relevancy;
- Context Precision;
- Context Recall;
- Context Relevance/Utilization variants;
- ID-based context precision and recall;
- noise sensitivity and other specialized metrics.

Current Ragas documentation recommends the newer collections-based API for new
work. The dependency and metric APIs should be version-pinned because the
library has evolved rapidly.

## 10.2 Strengths for this project

- directly addresses retriever vs generator quality;
- ID-based metrics fit Neo4j concept IDs well;
- faithfulness is suited to KG-grounded lesson content;
- supports datasets and batch evaluation;
- reference-free variants can help bootstrap;
- provides explanations for some judge-based scores.

## 10.3 Limitations

- many metrics require judge LLM calls;
- reference-dependent recall needs high-quality human data;
- standard text chunks do not map perfectly to graph nodes/relationships;
- it does not evaluate Planner routing or full agent behavior by itself;
- it does not provide FEM-specific pedagogy/UDL criteria;
- metric values vary with judge model, prompt and library version;
- running Ragas online for every lesson would add latency and cost.

## 10.4 Recommended use

Use Ragas primarily for:

- offline golden-set retrieval evaluation;
- faithfulness and answer relevance;
- comparison of retrieval/model/prompt versions;
- sampled production audits after calibration.

Prioritize deterministic ID-based precision/recall before LLM-only context
judgments.

Do not replace the current Critic with Ragas. They measure different things.

---

## 11. DeepEval

## 11.1 Role

DeepEval is an open-source LLM evaluation framework designed similarly to
pytest. Its current metric catalog includes:

- Answer Relevancy;
- Faithfulness;
- Contextual Precision;
- Contextual Recall;
- Contextual Relevancy;
- Hallucination;
- bias and toxicity;
- Tool Correctness;
- Task Completion;
- conversational metrics;
- G-Eval for custom natural-language rubrics.

## 11.2 Strengths for this project

- natural integration with pytest and CI;
- threshold-based assertions;
- agent/tool metrics beyond basic RAG;
- G-Eval supports FEM-specific pedagogical and UDL rubrics;
- item-level reasons aid debugging;
- local/custom judge models can be configured;
- suitable for regression tests.

## 11.3 Limitations

- many metrics are LLM-as-a-judge and non-deterministic;
- built-in RAG metrics overlap with Ragas;
- custom G-Eval rubrics still require human calibration;
- cloud features introduce another service/provider decision;
- evaluation can become expensive if every metric runs on every case;
- framework thresholds can create false confidence.

## 11.4 Recommended use

Use DeepEval for:

- pytest-style evaluation tests;
- Planner/tool/task-completion checks;
- custom pedagogical, UDL and safety G-Eval rubrics;
- CI orchestration after baselines stabilize.

During a short calibration phase, compare Ragas and DeepEval on the same
faithfulness/context cases. Then select one canonical implementation for each
headline metric rather than reporting duplicate conflicting values.

---

## 12. Langfuse

## 12.1 Role

Langfuse is primarily the observability and evaluation operations layer:

- traces and observations;
- prompts and versions;
- datasets;
- experiments;
- numeric/categorical/boolean/text scores;
- LLM-as-a-judge evaluations;
- annotation queues;
- user feedback.

## 12.2 Current project status

The repository currently uses Langfuse to fetch prompts with a short cache.
There is no evidence of full agent trace/span instrumentation or score
submission.

Documentation that describes per-agent tracing should therefore be treated as
target architecture rather than completed implementation.

## 12.3 Recommended use

- production trace store;
- prompt/model/KG comparison;
- dataset experiment tracking;
- attaching Ragas/DeepEval/custom scores;
- human annotation workflow;
- dashboarding and failure exploration.

## 12.4 Source-of-truth policy

Keep:

- canonical golden dataset in version-controlled, privacy-reviewed form;
- canonical metric definitions in this repository;
- essential run/evaluation summaries in FEM-controlled Postgres.

Langfuse may mirror datasets and scores for operations, but should not be the
only copy required to reproduce release decisions.

---

## 13. Framework selection recommendation

### Proposed stack

| Need | Recommended component |
|---|---|
| Deterministic runtime telemetry | Custom instrumentation + Postgres |
| Production traces and experiments | Langfuse |
| Neo4j concept Precision/Recall | Custom deterministic evaluators |
| RAG faithfulness/context benchmark | Ragas |
| Agent/tool/pedagogical CI checks | DeepEval + custom G-Eval rubrics |
| Human calibration | Domain-expert review + Langfuse annotations or controlled export |
| Product feedback | WebUI feedback persisted in Postgres and optionally sent to Langfuse |

### Avoiding unnecessary complexity

Do not introduce all framework features at once.

Recommended adoption:

1. custom telemetry and golden data;
2. custom deterministic retrieval metrics;
3. one Ragas offline benchmark;
4. DeepEval only for metrics that add unique agentic/pedagogical value;
5. Langfuse traces and score storage;
6. CI gates after calibration.

---

## 14. LLM-as-a-judge governance

LLM judges are scalable but are not ground truth.

### Known risks

- self-preference toward outputs from the same model family;
- verbosity bias;
- position bias in pairwise comparisons;
- language-dependent scoring;
- sensitivity to rubric wording;
- judge-model version drift;
- inconsistent reasoning and score;
- reward/grader hacking.

### Required controls

1. Pin judge model and configuration.
2. Use low/zero temperature where supported.
3. Blind candidate model identity.
4. Randomize pairwise answer order.
5. Use explicit scale anchors and critical-failure rules.
6. Require structured output with score and concise reason.
7. Validate against human-labeled examples.
8. Measure exact agreement, adjacent agreement and disagreement patterns.
9. Recalibrate after judge/rubric/domain changes.
10. Preserve judge version with every score.

### Promotion rule

An LLM metric should not become a release gate until:

- the rubric has been reviewed by domain experts;
- the judge has been tested on representative Italian and English examples;
- disagreement with humans has been analyzed;
- thresholds have demonstrated stable behavior across repeated runs;
- cost and latency are understood.

---

## 15. CI/CD quality-gate strategy

## 15.1 Pull-request checks

Fast and deterministic:

- schemas;
- routing fixtures;
- expected concept-ID checks on a small static retrieval fixture;
- required sections;
- parse robustness;
- safety/PII deterministic checks;
- no external LLM dependency where possible.

## 15.2 Nightly evaluation

Run live models and retrieval against the full golden set:

- Ragas/DeepEval/custom judges;
- model/provider health;
- dataset slices;
- token/cost report;
- candidate vs baseline.

Initially advisory.

## 15.3 Release evaluation

Before production:

- full golden dataset;
- required human review for high-impact prompt/KG changes;
- no critical safety/privacy failures;
- no material regression on headline metrics;
- staging smoke test.

### Example blocking policy after calibration

Block when:

- any critical safety/privacy case fails;
- run success falls below 95%;
- Cypher validity falls below 98%;
- a headline metric regresses by more than an agreed paired margin;
- parse errors exceed threshold;
- required dataset slices are missing.

Do not block on a small aggregate-score fluctuation without item-level review.

---

## 16. Pilot scorecard

The following is a proposed starting scorecard, not an SLA.

| Layer | Metric | Initial target | Frequency |
|---|---|---:|---|
| Reliability | Run success | ≥95% | Every run |
| Reliability | Parse errors per agent | <2% | Every run |
| Performance | End-to-end p95 | <180 s | Every run |
| Retrieval | Domain-routing accuracy | ≥90% | Golden set |
| Retrieval | Cypher validity | ≥98% | Every run/golden |
| Retrieval | ID Precision@5 | ≥0.70 | Golden set |
| Retrieval | ID Recall@10 | ≥0.70 | Golden set |
| Retrieval | Context precision | ≥0.60 | Golden set |
| Retrieval | Context recall | ≥0.70 | Golden set |
| Generation | Faithfulness | ≥0.75 | Golden + sample |
| Generation | Answer relevance | ≥0.75 | Golden + sample |
| Generation | Required-section completeness | ≥95% | Every run |
| Pedagogy | Expert/judge rubric | ≥3.5/5 | Golden + sample |
| Safety | Critical failures | 0 | Every release |
| Product | Useful/very useful | ≥80% | Pilot |
| Product | Acceptable without substantial rewrite | ≥70% | Pilot sample |

Report every metric by:

- Neuro vs UDL;
- Italian vs English;
- intent/query type;
- in-scope/out-of-scope;
- initial/follow-up turn;
- model/provider;
- pipeline version.

---

## 17. Human evaluation process

### 17.1 Rubric

Use 1–5 anchored scales:

- 1 = unacceptable/unsafe;
- 2 = major correction required;
- 3 = usable with meaningful edits;
- 4 = strong, minor edits;
- 5 = excellent, ready to use.

### 17.2 Reviewer workflow

1. Blind reviewer to model/version.
2. Provide user request, profile, retrieved evidence and output.
3. Score independent dimensions.
4. Mark critical issues.
5. Add short reason.
6. Review disagreements.
7. Feed confirmed failures into the golden dataset.

### 17.3 Inter-rater quality

For double-reviewed samples:

- percent agreement;
- adjacent-score agreement;
- Cohen’s kappa for categorical outcomes where appropriate;
- disagreement reasons.

Human rubric inconsistency must be addressed before blaming automated judges.

---

## 18. Privacy, security and EU governance

### Data minimization

Do not send to evaluation providers unless required:

- user identity;
- student names;
- full uploaded documents;
- authentication/session data;
- secrets;
- unrelated conversation history.

### Pseudonymization

Use:

- internal trace IDs;
- hashed/pseudonymous user IDs;
- redacted prompts;
- minimal required evidence.

### Retention

Define separately:

- operational logs;
- production traces;
- evaluation scores;
- golden dataset;
- human comments;
- uploaded documents.

### Provider review

For Langfuse cloud, judge LLMs and hosted evaluation products, document:

- processor/subprocessor;
- data region;
- retention;
- training-on-data policy;
- DPA;
- deletion procedure;
- access controls.

### EU AI Act support

Evaluation evidence can support:

- transparency;
- logging and traceability;
- human oversight;
- quality/risk monitoring;
- documented change management.

It does not itself establish legal compliance; legal review remains separate.

---

## 19. Delivery roadmap

### Phase 0 — Definitions and ownership

**Effort:** 2–3 AI-team days plus expert workshop

- approve metric dictionary;
- define critical failures;
- select pilot scorecard;
- nominate domain owners;
- confirm privacy rules;
- resolve current documentation vs implementation discrepancies.

### Phase 1 — Deterministic telemetry

**Effort:** 4–6 development days

- stage timing;
- parse-error persistence;
- retrieval outcome/grade;
- tokens/cost;
- model/prompt versions;
- extend metadata/API schemas;
- initial SQL/operational report.

### Phase 2 — Golden dataset v1

**Effort:** 5–8 team days plus expert labeling

- 20 cases;
- expected concepts/routing;
- rubric and forbidden behavior;
- double-review selected cases;
- versioning and privacy review.

### Phase 3 — Evaluation runner

**Effort:** 5–8 development days

- framework-neutral runner;
- deterministic retrieval evaluators;
- candidate/baseline comparison;
- reports;
- advisory CI integration.

### Phase 4 — Ragas and DeepEval integration

**Effort:** 4–7 development days

- pin versions;
- map captured fields;
- calibrate overlapping metrics;
- implement selected RAG and G-Eval evaluators;
- cost controls and caching.

### Phase 5 — Langfuse tracing and experiments

**Effort:** 4–7 development days

- traces/spans;
- scores;
- datasets/experiments;
- token/cost;
- redaction and retention;
- production sampling.

### Phase 6 — Human/pilot feedback

**Effort:** 3–5 development days plus pilot operation

- feedback UI;
- persistence;
- annotation workflow;
- weekly review;
- promote failures to golden cases.

### Overall planning range

Technical implementation:

```text
approximately 22–41 development days
```

Calendar duration with expert review, privacy decisions and pilot feedback:

```text
approximately 6–10 weeks
```

This can be split into independent increments. Phase 1–3 already deliver major
value without waiting for the full stack.

---

## 20. Proposed code and repository structure

```text
src/aix/evaluation/
  __init__.py
  schemas.py
  deterministic.py
  retrieval.py
  generation.py
  pedagogical.py
  safety.py
  ragas_adapter.py
  deepeval_adapter.py
  persistence.py

scripts/eval/
  run_agent_eval.py
  compare_runs.py
  export_report.py

data/eval/
  golden_v1.jsonl
  rubrics/
    pedagogy_v1.yaml
    udl_v1.yaml
    safety_v1.yaml

tests/eval/
  test_golden_smoke.py
  test_retrieval_metrics.py
  test_pedagogical_rubric.py
```

Existing files likely modified:

- `src/aix/agent/graph/state.py`;
- `src/aix/agent/graph/nodes.py`;
- `src/aix/webui/agent/service.py`;
- `src/aix/webui/lessons/models.py`;
- `src/aix/api/schemas/agent.py`;
- common LLM call/config layer;
- Langfuse prompt client or a new observability module;
- CI workflow;
- environment and deployment templates;
- technical documentation.

---

## 21. Acceptance criteria

### Evaluation backbone

- Golden dataset has a version, owner and review status.
- Runner produces reproducible item-level JSON results.
- Candidate and baseline run against identical cases.
- Results identify pipeline, prompts, models, KG and evaluators.

### Retrieval

- Retrieved concept IDs and ranks are captured.
- Precision@K and Recall@K are deterministic and unit-tested.
- Empty/out-of-scope behavior is represented.

### Generation

- Faithfulness and relevance are computed offline.
- Required-section and instruction checks run deterministically.
- Critical unsupported claims are surfaced separately.

### Agent behavior

- Planner routing and tool expectations can be evaluated.
- Revision/retry effectiveness is reported.
- Parse errors cannot inflate approval metrics silently.

### Operations

- Stage latency, tokens and cost are captured.
- Evaluation sampling does not block teacher responses.
- Scores can be aggregated by domain/language/version.

### Governance

- Judge/rubric versions are stored.
- At least one human-calibrated sample exists.
- Privacy/redaction rules are documented and tested.
- Initial CI is advisory.

### Pilot

- Teacher feedback can be collected.
- Weekly pilot report contains reliability, retrieval, generation and human
  outcomes.
- Confirmed failures become new regression cases.

---

## 22. Risks and mitigations

| Risk | Mitigation |
|---|---|
| Framework score treated as truth | Human calibration, explicit metric definitions |
| Same-model Critic bias | Independent judge and expert sample |
| Judge/model drift | Version pinning and recalibration |
| Eval cost growth | Sampling, metric tiers, caching, nightly batches |
| Slow CI | Deterministic PR set; live judges nightly/release |
| Poor golden references | Dual review and versioned changes |
| Overfitting to 20 cases | Expand from production failures and maintain slices |
| Privacy leakage | Redaction, pseudonymization, provider/DPA review |
| Vendor lock-in | Canonical local schemas and adapters |
| Conflicting Ragas/DeepEval scores | Select one canonical implementation per metric |
| Misleading averages | Slice reports and critical-failure gates |
| Metrics become targets rather than quality | Periodic qualitative review and rubric refresh |

---

## 23. Decisions required

Before implementation, FEM should decide:

1. Who owns the golden dataset?
2. Which Neuro and UDL experts approve expected concepts and rubrics?
3. Which metrics are pilot KPIs vs internal diagnostics?
4. Which failures are immediately blocking?
5. Which judge model/provider is permitted?
6. May production prompts/outputs be sent to Langfuse cloud?
7. What sampling rate and retention are acceptable?
8. Should Langfuse be cloud-hosted or self-hosted?
9. Which framework metrics become canonical after calibration?
10. Who reviews weekly pilot failures and updates the dataset?

---

## 24. Recommended immediate next step

Do not begin by adding every framework.

The next concrete work package should be:

1. extract and normalize the metrics already produced by the pipeline;
2. create the 20-case golden dataset schema;
3. label expected routing and concept IDs;
4. implement deterministic Precision@K/Recall@K;
5. run and review the first baseline;
6. only then add Ragas faithfulness/context metrics and DeepEval custom rubrics.

This sequence produces useful evidence early and prevents the project from
building an expensive evaluation layer on undefined success criteria.

---

## Appendix A — Metric dictionary

| Metric | Type | Required input | Primary failure diagnosed |
|---|---|---|---|
| Run success | Deterministic | Status | Reliability |
| Parse-error rate | Deterministic | Structured errors | Agent output contract |
| Stage latency | Deterministic | Timestamps | Performance |
| Token/cost | Deterministic | Provider usage | Efficiency |
| Domain accuracy | Reference | Expected/predicted domain | Planner routing |
| Precision@K | Reference | Expected and retrieved IDs | Retrieval noise |
| Recall@K | Reference | Expected and retrieved IDs | Missing evidence |
| MRR | Reference | Principal expected ID/rank | Ranking |
| nDCG@K | Reference | Graded relevance/rank | Ranking quality |
| Context precision | Reference/judge | Contexts + reference/response | Retrieval ranking |
| Context recall | Reference/judge | Contexts + reference | Missing context |
| Faithfulness | LLM judge | Output + contexts | Unsupported claims |
| Answer relevance | LLM judge | Input + output | Drift |
| Completeness | Deterministic/judge | Requirements + output | Missing sections |
| Instruction adherence | Deterministic/judge | Input/profile + output | Constraint violation |
| Tool correctness | Reference/deterministic | Expected/actual calls | Agent behavior |
| Task completion | Judge/human | Goal + output | End-to-end failure |
| Pedagogical score | Judge/human | Rubric + output/context | Educational quality |
| UDL alignment | Judge/human | UDL rubric + output/profile | Inclusive design |
| Safety | Deterministic/judge/human | Policy + output | Harm |
| Teacher usefulness | Human | Rating | Product success |

---

## Appendix B — Failure taxonomy

```text
REL-01  Run exception
REL-02  Timeout
REL-03  Parse failure
REL-04  Persistence/stream interruption

PLAN-01 Wrong intent
PLAN-02 Wrong domain/scope
PLAN-03 Missing constraints
PLAN-04 Poor search-query decomposition

RET-01  Invalid Cypher
RET-02  Zero retrieval for in-scope request
RET-03  Irrelevant context
RET-04  Missing expected concepts
RET-05  Duplicate/noisy context
RET-06  Incorrect cross-domain expansion

GEN-01  Unsupported claim
GEN-02  Irrelevant answer
GEN-03  Missing required section
GEN-04  Constraint violation
GEN-05  Citation error
GEN-06  Language/format failure

PED-01  Objective/activity misalignment
PED-02  Assessment misalignment
PED-03  Impractical timing/materials
PED-04  Age/class mismatch
PED-05  Insufficient differentiation/accessibility
PED-06  Neuromyth or weak evidence translation

AGT-01  Wrong tool/route
AGT-02  Ineffective retry
AGT-03  Ineffective revision
AGT-04  Loop/state inconsistency

SAFE-01 Harmful/inappropriate content
SAFE-02 PII/privacy leak
SAFE-03 Unsupported clinical claim
SAFE-04 Missing AI disclosure
```

---

## Appendix C — References

### Current repository

- `src/aix/retrieval/query_metrics.py`
- `src/aix/agent/prompts/critic_prompt.py`
- `src/aix/agent/agents/critic_agent.py`
- `src/aix/agent/agents/retrieval_grader_agent.py`
- `src/aix/agent/graph/nodes.py`
- `src/aix/webui/agent/service.py`
- `src/aix/webui/lessons/models.py`
- `src/aix/domains/langfuse_prompts.py`

### External documentation

- Ragas documentation: <https://docs.ragas.io/>
- Ragas Context Precision:
  <https://docs.ragas.io/en/latest/concepts/metrics/available_metrics/context_precision/>
- Ragas Faithfulness:
  <https://docs.ragas.io/en/stable/concepts/metrics/available_metrics/faithfulness/>
- DeepEval documentation: <https://deepeval.com/docs>
- DeepEval Answer Relevancy:
  <https://deepeval.com/docs/metrics-answer-relevancy>
- DeepEval Faithfulness:
  <https://deepeval.com/docs/metrics-faithfulness>
- DeepEval repository:
  <https://github.com/confident-ai/deepeval>
- Langfuse documentation: <https://langfuse.com/docs>
- Langfuse datasets:
  <https://langfuse.com/docs/evaluation/experiments/datasets>
- Langfuse scores:
  <https://langfuse.com/docs/evaluation/scores/overview>
- Langfuse SDK experiments:
  <https://langfuse.com/docs/evaluation/experiments/experiments-via-sdk>
- OpenAI evaluation best practices:
  <https://developers.openai.com/api/docs/guides/evaluation-best-practices>
- OpenAI graders:
  <https://developers.openai.com/api/docs/guides/graders>

