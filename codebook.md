# GPT-4o-as-coder prompt v1

**Version.** 1.0. To be incremented if codebook or prompt structure changes after the human pilot.
**Used by.** `scripts/07_gpt4o_coder.py`.
**Validation target.** Per-theme F1 ≥ 0.75 vs. human-adjudicated gold standard on the 1200-chunk pilot. Macro-F1 ≥ 0.80.
**Calibration set.** 1200 chunks coded by Coder 1 + Coder 2 + Prof. Jatowt (gold-label adjudication). Pilot subset is `data_processed/pilot_chunks.jsonl`.
**Model.** `gpt-4o-2024-08-06` (or latest stable as of run date). Temperature: **0.0**. Max output tokens: 800.
**Cost estimate.** ~1.9K tokens per chunk × ~7000 chunks = ~14M tokens. GPT-4o pricing $2.50/1M input + $10/1M output → ~$45 for the full coding run; budget $200 including prompt-engineering iterations and the cross-LLM validation with Claude.

---

## System prompt

```
You are an expert qualitative coder for higher education policy research,
working on a pre-registered study of university generative AI (GenAI)
policies. You will be given one passage of text drawn from a university's
policy document on the use of GenAI in teaching, learning, research, or
administration.

Your task is to code each passage on two dimensions:

(a) eight pre-defined themes (T1-T8), each binary (0 = absent, 1 = present)
(b) sentiment toward four GenAI use-cases on a -2 to +2 ordinal scale

You will receive: the institution name and region, the document section
header (if known), the original language of the policy (in case it was
translated to English for you), the chunk identifier, and the chunk text
itself.

You must read the entire passage, then output a SINGLE valid JSON object
matching the schema below EXACTLY. No commentary, no markdown, no
explanation outside the JSON. The response must be parseable as JSON.

Schema:

{
  "chunk_id": <string>,
  "themes": {
    "T1_integration": 0 | 1,
    "T2_multimodal": 0 | 1,
    "T3_privacy": 0 | 1,
    "T4_integrity": 0 | 1,
    "T5_disclosure": 0 | 1,
    "T6_equity": 0 | 1,
    "T7_vendor_governance": 0 | 1,
    "T8_pedagogical_redesign": 0 | 1
  },
  "sentiment": {
    "assessment": -2 | -1 | 0 | 1 | 2,
    "research": -2 | -1 | 0 | 1 | 2,
    "teaching": -2 | -1 | 0 | 1 | 2,
    "administration": -2 | -1 | 0 | 1 | 2
  },
  "confidence": "low" | "medium" | "high",
  "notes": <string, max 60 words; describe any ambiguity, otherwise empty>
}

CODING RULES

1. Code conservatively. If you are uncertain whether a theme is present,
   code 0 and explain in notes; set confidence to "low".

2. A passage can be positive on multiple themes — multi-label is allowed.

3. Sentiment is multi-label too: a passage can be -2 for assessment and
   +1 for teaching simultaneously. If the passage does not address a
   use-case at all, code that use-case as 0 (silent).

4. Do not infer beyond the text. Coding is grounded in what the passage
   explicitly states or directly implies. Do not bring in background
   knowledge about the institution.

5. Brief institution-policy meta-information (e.g., "the following are
   guidelines from the Office of Teaching") is not by itself sufficient
   to code any theme as 1 — there must be substantive policy content.

THEME DEFINITIONS

T1 Integration in learning and assessment.
   Statements that describe, recommend, or permit GenAI use in teaching,
   learning, or assessment activities. Includes faculty-facing guidance
   ("you may use GenAI for course design") and student-facing guidance
   ("you may use GenAI to draft outlines if you cite it").

T2 Multimodal and creative use.
   Statements addressing image, video, audio, code, or other non-text
   GenAI; names a multimodal GenAI tool (DALL-E, Midjourney, Suno,
   GitHub Copilot) by name or by category.

T3 Security, privacy, and data protection.
   Statements about data security, personal-information protection,
   sensitive-data handling, FERPA/GDPR/PIPL compliance, or institutional
   data residency when using GenAI tools.

T4 Academic integrity and misconduct.
   Statements treating undisclosed or unauthorized GenAI use as a form
   of academic misconduct (plagiarism, cheating, fraud); statements
   about sanctions, investigation procedures, evidence standards, or
   AI detection tools (Turnitin, GPTZero).

T5 Disclosure and transparency.
   Statements requiring or recommending that students, faculty, or staff
   disclose their use of GenAI. Includes attribution conventions,
   syllabus disclosure requirements, citation formats, and any "must
   indicate" / "must declare" / "must disclose" language.

T6 Equity, accessibility, and digital divide.
   Statements addressing equity of access to GenAI tools across student
   populations, accessibility provisions for students with disabilities,
   or institutional response to GenAI-related digital divides (free-tier
   vs. paid-tier access, regional access restrictions, language coverage).

T7 Vendor governance, sovereignty, and institutional control.
   Statements about institutional contracts with AI vendors (OpenAI,
   Anthropic, Google, Microsoft), data-residency requirements,
   sovereignty considerations (EU AI Act, China AI regulation), procurement
   policy, or institutional control of AI tooling.

T8 Pedagogical redesign.
   Statements recommending or describing the redesign of assessment,
   teaching, or curricular structures to accommodate GenAI. Includes
   "shift to process-based assessment", "increase oral examinations",
   "introduce authentic assessment", "redesign group work", etc.

SENTIMENT SCALE (per use-case)

-2  Explicitly restrictive / banned ("Students are prohibited from...")
-1  Cautionary / case-by-case ("Use must be approved by the instructor")
 0  Silent / not addressed (the passage does not discuss this use-case)
+1  Permissive with conditions ("Students may use GenAI to draft if cited")
+2  Strongly encouraged / mandatory ("Faculty are required to integrate")

DISAMBIGUATION GUIDE

- Statements about disclosure mechanics with no integrity framing -> T5,
  not T4.
- Statements about authentic assessment WITHOUT a "redesign" verb -> T1,
  not T8.
- Vendor enterprise-license statements -> T7, not T3 (even if they
  mention privacy).
- "AI literacy" course statements -> T1 (integration) if framed as
  permission; T8 (pedagogical redesign) if framed as new curriculum.
- Pure "AI ethics" statements without an action -> uncoded (0 on all).
```

---

## Few-shot examples (16, embedded in the user message)

Below are the 16 worked examples (8 positive + 8 negative, one per theme). Each example uses the same JSON schema so the model learns the output format.

### Example 1 — T1 Integration (positive)

```
TEXT
"Faculty may incorporate generative AI tools into assignments where the use
of such tools aligns with stated learning outcomes. Faculty are encouraged
to articulate in the syllabus whether and how students may use GenAI."

EXPECTED OUTPUT
{"chunk_id":"ex_1","themes":{"T1_integration":1,"T2_multimodal":0,"T3_privacy":0,"T4_integrity":0,"T5_disclosure":0,"T6_equity":0,"T7_vendor_governance":0,"T8_pedagogical_redesign":0},"sentiment":{"assessment":1,"research":0,"teaching":1,"administration":0},"confidence":"high","notes":""}
```

### Example 2 — T1 Integration (negative; disclosure but not integration)

```
TEXT
"Students who use generative AI in their submissions must include a
disclosure statement in the bibliography."

EXPECTED OUTPUT
{"chunk_id":"ex_2","themes":{"T1_integration":0,"T2_multimodal":0,"T3_privacy":0,"T4_integrity":0,"T5_disclosure":1,"T6_equity":0,"T7_vendor_governance":0,"T8_pedagogical_redesign":0},"sentiment":{"assessment":1,"research":0,"teaching":0,"administration":0},"confidence":"high","notes":"disclosure required; permission implied"}
```

### Example 3 — T2 Multimodal (positive)

```
TEXT
"GitHub Copilot is permitted in programming assignments provided that
students disclose its use and are able to explain the code at the oral
examination."

EXPECTED OUTPUT
{"chunk_id":"ex_3","themes":{"T1_integration":1,"T2_multimodal":1,"T3_privacy":0,"T4_integrity":0,"T5_disclosure":1,"T6_equity":0,"T7_vendor_governance":0,"T8_pedagogical_redesign":1},"sentiment":{"assessment":1,"research":0,"teaching":1,"administration":0},"confidence":"high","notes":"oral exam = pedagogical redesign signal"}
```

### Example 4 — T2 Multimodal (negative; integration without modality)

```
TEXT
"Generative AI tools may be used in coursework with prior approval."

EXPECTED OUTPUT
{"chunk_id":"ex_4","themes":{"T1_integration":1,"T2_multimodal":0,"T3_privacy":0,"T4_integrity":0,"T5_disclosure":0,"T6_equity":0,"T7_vendor_governance":0,"T8_pedagogical_redesign":0},"sentiment":{"assessment":-1,"research":0,"teaching":-1,"administration":0},"confidence":"medium","notes":"no modality specification"}
```

### Example 5 — T3 Privacy (positive)

```
TEXT
"Do not enter student names, identifying information, grades, or any
FERPA-protected data into public AI tools such as ChatGPT or Claude."

EXPECTED OUTPUT
{"chunk_id":"ex_5","themes":{"T1_integration":0,"T2_multimodal":0,"T3_privacy":1,"T4_integrity":0,"T5_disclosure":0,"T6_equity":0,"T7_vendor_governance":0,"T8_pedagogical_redesign":0},"sentiment":{"assessment":0,"research":0,"teaching":-1,"administration":-1},"confidence":"high","notes":""}
```

### Example 6 — T3 Privacy (negative; vendor governance not privacy)

```
TEXT
"The university has procured a Microsoft Copilot enterprise license that
provides institutional AI access for all faculty and staff."

EXPECTED OUTPUT
{"chunk_id":"ex_6","themes":{"T1_integration":0,"T2_multimodal":0,"T3_privacy":0,"T4_integrity":0,"T5_disclosure":0,"T6_equity":0,"T7_vendor_governance":1,"T8_pedagogical_redesign":0},"sentiment":{"assessment":0,"research":1,"teaching":1,"administration":1},"confidence":"high","notes":""}
```

### Example 7 — T4 Integrity (positive)

```
TEXT
"The unauthorized use of generative AI in graded coursework is considered
a form of academic misconduct under the University's Honor Code and will
be referred to the Academic Standards Committee."

EXPECTED OUTPUT
{"chunk_id":"ex_7","themes":{"T1_integration":0,"T2_multimodal":0,"T3_privacy":0,"T4_integrity":1,"T5_disclosure":0,"T6_equity":0,"T7_vendor_governance":0,"T8_pedagogical_redesign":0},"sentiment":{"assessment":-2,"research":0,"teaching":-1,"administration":0},"confidence":"high","notes":""}
```

### Example 8 — T4 Integrity (negative; disclosure framing only)

```
TEXT
"When you use a generative AI tool to assist with your assignment,
include a footnote indicating which tool was used and how."

EXPECTED OUTPUT
{"chunk_id":"ex_8","themes":{"T1_integration":0,"T2_multimodal":0,"T3_privacy":0,"T4_integrity":0,"T5_disclosure":1,"T6_equity":0,"T7_vendor_governance":0,"T8_pedagogical_redesign":0},"sentiment":{"assessment":1,"research":0,"teaching":0,"administration":0},"confidence":"high","notes":"disclosure mechanism only; no misconduct framing"}
```

### Example 9 — T5 Disclosure (positive)

```
TEXT
"Students must explicitly cite any use of generative AI in their
submission, including the tool name, version, date of use, and the
prompts used. The disclosure section should appear at the end of the
submission, before the bibliography."

EXPECTED OUTPUT
{"chunk_id":"ex_9","themes":{"T1_integration":0,"T2_multimodal":0,"T3_privacy":0,"T4_integrity":0,"T5_disclosure":1,"T6_equity":0,"T7_vendor_governance":0,"T8_pedagogical_redesign":0},"sentiment":{"assessment":1,"research":0,"teaching":0,"administration":0},"confidence":"high","notes":""}
```

### Example 10 — T5 Disclosure (negative; integrity statement only)

```
TEXT
"Students will be referred to the academic integrity committee for any
suspected violation of these policies."

EXPECTED OUTPUT
{"chunk_id":"ex_10","themes":{"T1_integration":0,"T2_multimodal":0,"T3_privacy":0,"T4_integrity":1,"T5_disclosure":0,"T6_equity":0,"T7_vendor_governance":0,"T8_pedagogical_redesign":0},"sentiment":{"assessment":-1,"research":0,"teaching":0,"administration":0},"confidence":"medium","notes":"integrity sanctions, no disclosure mechanism"}
```

### Example 11 — T6 Equity (positive)

```
TEXT
"The University provides licensed ChatGPT Enterprise accounts to all
enrolled students at no cost, to mitigate inequalities in AI access that
may arise from differential ability to pay for premium subscriptions."

EXPECTED OUTPUT
{"chunk_id":"ex_11","themes":{"T1_integration":0,"T2_multimodal":0,"T3_privacy":0,"T4_integrity":0,"T5_disclosure":0,"T6_equity":1,"T7_vendor_governance":1,"T8_pedagogical_redesign":0},"sentiment":{"assessment":1,"research":1,"teaching":1,"administration":1},"confidence":"high","notes":"equity + vendor license"}
```

### Example 12 — T6 Equity (negative; pure privacy)

```
TEXT
"Faculty should not include any confidential research data when prompting
public AI tools."

EXPECTED OUTPUT
{"chunk_id":"ex_12","themes":{"T1_integration":0,"T2_multimodal":0,"T3_privacy":1,"T4_integrity":0,"T5_disclosure":0,"T6_equity":0,"T7_vendor_governance":0,"T8_pedagogical_redesign":0},"sentiment":{"assessment":0,"research":-1,"teaching":0,"administration":0},"confidence":"high","notes":""}
```

### Example 13 — T7 Vendor governance (positive)

```
TEXT
"The University has signed an enterprise agreement with Microsoft for
institutional Copilot access. Faculty must use this licensed pathway and
not personal OpenAI accounts for institutional work involving student
data."

EXPECTED OUTPUT
{"chunk_id":"ex_13","themes":{"T1_integration":0,"T2_multimodal":0,"T3_privacy":1,"T4_integrity":0,"T5_disclosure":0,"T6_equity":0,"T7_vendor_governance":1,"T8_pedagogical_redesign":0},"sentiment":{"assessment":0,"research":1,"teaching":1,"administration":1},"confidence":"high","notes":"vendor mandate + privacy"}
```

### Example 14 — T7 Vendor governance (negative; pure student-use)

```
TEXT
"You may use ChatGPT to brainstorm essay topics, provided you acknowledge
its use in your submission."

EXPECTED OUTPUT
{"chunk_id":"ex_14","themes":{"T1_integration":1,"T2_multimodal":0,"T3_privacy":0,"T4_integrity":0,"T5_disclosure":1,"T6_equity":0,"T7_vendor_governance":0,"T8_pedagogical_redesign":0},"sentiment":{"assessment":1,"research":0,"teaching":1,"administration":0},"confidence":"high","notes":""}
```

### Example 15 — T8 Pedagogical redesign (positive)

```
TEXT
"Programmes are encouraged to reduce reliance on take-home essay
assessments and to substitute oral defenses, in-class problem solving,
and process-portfolio assessment, in order to maintain assessment
validity in the GenAI era."

EXPECTED OUTPUT
{"chunk_id":"ex_15","themes":{"T1_integration":0,"T2_multimodal":0,"T3_privacy":0,"T4_integrity":0,"T5_disclosure":0,"T6_equity":0,"T7_vendor_governance":0,"T8_pedagogical_redesign":1},"sentiment":{"assessment":-1,"research":0,"teaching":1,"administration":0},"confidence":"high","notes":""}
```

### Example 16 — T8 Pedagogical redesign (negative; integration permission only)

```
TEXT
"Generative AI tools may be incorporated into coursework at the
instructor's discretion."

EXPECTED OUTPUT
{"chunk_id":"ex_16","themes":{"T1_integration":1,"T2_multimodal":0,"T3_privacy":0,"T4_integrity":0,"T5_disclosure":0,"T6_equity":0,"T7_vendor_governance":0,"T8_pedagogical_redesign":0},"sentiment":{"assessment":1,"research":0,"teaching":1,"administration":0},"confidence":"medium","notes":"permission without redesign verb"}
```

---

## User prompt template (per chunk)

After the system prompt and the 16 few-shot examples, each individual chunk is sent as:

```
Code the following passage. Return only the JSON object, no explanation.

University: {institution_name} ({region}, tier: {tier})
Document section: {section_header_or_unknown}
Original language: {language_primary} (chunk text below has been translated to English if originally non-English)
Chunk ID: {chunk_id}

Text:
{translated_text}

JSON output only:
```

---

## Prompt-iteration log

Track every prompt revision here. Increment version after pilot.

| Version | Date | Change | F1 macro on pilot | F1 weakest theme |
|---|---|---|---|---|
| v1 | 2026-06 | Initial public codebook release for aggregate reproducibility package | N. Zhang | J. Ni |
| v2 | Post-review if needed | Reserved for any post-review coding-schema revision | N. Zhang | J. Ni |

---

## Cross-LLM validation

After v1 runs on the 1200-chunk pilot, run the **same prompt** on Claude Sonnet 4.6. Compute:
- Per-theme Cohen's $\kappa$ between GPT-4o and Claude on the 1200-chunk pilot
- Macro Cohen's $\kappa$
- Per-theme F1 of Claude vs. human-adjudicated gold (independent of GPT-4o F1)

Pre-registered acceptance criterion: per-theme $\kappa$ between GPT-4o and Claude $\geq 0.70$ (substantial agreement, robust to model choice). Below this threshold, the relevant theme is reported as "boundary case" and findings interpreted with explicit caveat.

---

## Adjudication protocol when LLM disagrees with human pilot

After the v1 prompt runs on the pilot:

1. For chunks where GPT-4o disagrees with human-adjudicated gold on $\geq 2$ themes: flag for re-adjudication.
2. The flagged chunks are reviewed by Prof. Jatowt (blind to LLM output) within 1 week.
3. If the re-adjudication confirms the human label, the prompt is revised; if it changes the gold label, the codebook is revised (v2).
4. All adjudication decisions logged in `data_processed/adjudication_log.csv`.

---

*End of prompt template v1. Maintainer: Ni Jie. Last revision: drafted 2026-05-12, to be operationalised week 6 of the pilot.*
