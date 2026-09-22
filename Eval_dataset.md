I want you to guide me through building the evaluation ground-truth dataset for the Umbrella RAG project.


==================================================
GOAL
==================================================

Create a fixed 50-question evaluation dataset for the Umbrella RAG project.

Use the GitHub repository:

https://github.com/44bits/awesome-opensource-documents

The workflow should be:

Selected documents from the same domain
        ↓
RAGAS synthetic testset generator
        ↓
60 candidate questions
        ↓
Quality filtering
        ↓
50 final questions
        ↓
Manual ground-truth answer creation/verification
        ↓
Frozen evaluation dataset

This dataset will later be used to evaluate and compare different Umbrella RAG configurations.


==================================================
DOCUMENT SELECTION
==================================================

We need approximately 5–10 documents from the SAME coherent domain/topic from the repository.

Do not randomly combine unrelated documents.

Analyze the repository and recommend a suitable group of documents based on:
- coherent domain
- sufficient information density
- ability to generate meaningful retrieval questions
- document quality
- availability/accessibility
- licensing suitability

For each proposed document, show:
- title
- source URL
- domain/topic
- why it is useful
- license information if available

Do not invent URLs, licenses, or document information.

Before downloading anything, explain why these documents were selected.

Then guide me through the component responsible for obtaining/downloading the selected documents.

Do not implement the complete download system automatically.

==================================================
RAGAS SYNTHETIC TESTSET GENERATION
==================================================

Use RAGAS's synthetic testset generation functionality to generate approximately 60 candidate questions from the selected documents.

Important:

RAGAS is being used primarily as the synthetic QUESTION GENERATION mechanism.

Do not blindly accept RAGAS-generated reference answers as the final ground truth.

I want the final ground-truth answers to be manually verified.

First explain:
- what RAGAS synthetic testset generation does
- what input it expects
- what output it produces
- how it generates questions from documents
- what role the LLM plays
- what role the document chunks/context play
- how the generated questions will eventually become our evaluation dataset


==================================================
QUESTION FILTERING LOGIC: 60 → 50
==================================================

After RAGAS generates approximately 60 candidate questions, curate them down to exactly 50 high-quality evaluation questions.

Do NOT simply keep the first 50 and do NOT use an arbitrary numerical quality score.

The goal is to create a reliable, diverse, source-grounded dataset that meaningfully evaluates Umbrella's RAG retrieval and answer generation.

For every candidate, evaluate the following:

1. GROUNDING / ANSWERABILITY — MANDATORY
- The question must be answerable from the selected source documents.
- The source must contain enough information to establish a clear answer.
- Remove questions requiring outside or general knowledge.
- Remove questions whose answer cannot be verified from the source.

2. CLARITY / AMBIGUITY — MANDATORY
- The question must have one clear interpretation.
- Remove questions with multiple reasonable interpretations.
- Remove subjective questions unless the source explicitly provides the relevant claim.

3. DUPLICATION — MANDATORY
- Remove exact duplicates.
- Remove semantically equivalent or near-duplicate questions that test essentially the same information.
- Judge duplication based on meaning and expected answer, not only string similarity.
- Do not introduce an arbitrary embedding-similarity threshold; at this dataset size, use semantic/LLM-assisted review and human review where necessary.

4. RETRIEVAL VALUE
Prefer questions that require retrieving relevant information from the corpus.

Prefer:
- specific information retrieval
- section-level retrieval
- contextual retrieval
- connecting related information
- multi-step/multi-hop questions when naturally supported

Avoid questions that an LLM could reliably answer without accessing the corpus.

5. DIFFICULTY
- Remove questions that are excessively trivial or provide almost no meaningful retrieval challenge.
- Keep some straightforward factual retrieval questions because they are valid RAG queries.
- Do not artificially make questions difficult.

6. QUESTION-TYPE DIVERSITY
Prefer a useful mixture of:
- factual retrieval
- specific/contextual retrieval
- section-specific questions
- interpretation of retrieved context
- multi-step/multi-hop questions where naturally supported

Do not force a question type when the source material does not support it.

7. SOURCE COVERAGE
Prefer reasonable coverage across the selected documents.

Do NOT enforce an exact number of questions per document.

Do NOT keep weak questions merely to satisfy document quotas.

Guide me in creating the filtering/validation logic.

--------------------------------------------------
PRIORITY WHEN CRITERIA CONFLICT
--------------------------------------------------

Use this priority order:

1. Grounding / answerability
2. Clarity / lack of ambiguity
3. No duplication / near-duplication
4. Retrieval value
5. Appropriate difficulty
6. Question-type diversity
7. Source coverage

Grounding, clarity, and duplication are mandatory.

Retrieval value takes priority over artificially balancing documents or question types.


--------------------------------------------------
CURATION PROCESS
--------------------------------------------------

For each candidate, classify:

KEEP
REMOVE
BORDERLINE

For every REMOVE decision, record a concise reason:
- duplicate
- near-duplicate
- ambiguous
- insufficiently grounded
- outside knowledge required
- too trivial
- poor retrieval value
- poor question quality

For BORDERLINE questions, record why the decision is uncertain and present them for my review.

Do not silently resolve important borderline cases.

--------------------------------------------------
WHEN MORE THAN 50 VALID QUESTIONS REMAIN
--------------------------------------------------

If more than 50 candidates survive the mandatory checks:

1. Prefer stronger retrieval-value questions.
2. If retrieval value is comparable, prefer the question that adds more useful question-type diversity.
3. If still comparable, prefer the question that improves source coverage.
4. If still effectively equivalent, flag the choice for human review.

"Comparable" or "effectively equivalent" does not need an arbitrary numerical threshold. Use qualitative judgment and surface genuinely borderline choices for human review.

Do not fabricate precision with arbitrary scores.

--------------------------------------------------
WHEN FEWER THAN 50 VALID QUESTIONS REMAIN
--------------------------------------------------

Do NOT lower the quality criteria just to reach 50.

If fewer than 50 valid questions remain:

- Generate at most 10 replacement candidates in one round.
- Apply exactly the same filtering criteria to the replacements.
- Allow a maximum of 2 replacement rounds.
- After each round, report how many valid questions remain.
- Do not keep questions known to be ambiguous, ungrounded, repetitive, or poor retrieval tests merely to reach 50.

If fewer than 50 valid questions still remain after 2 replacement rounds:

STOP.

Report that the selected corpus may not contain enough suitable material for a 50-question benchmark and flag the document selection/corpus as the likely issue.

Do not continue generating questions indefinitely.

--------------------------------------------------
FINAL CURATION RECORD
--------------------------------------------------

Before finalizing the dataset, show the complete curation results for all candidates, including:

- question ID
- question
- KEEP / REMOVE / BORDERLINE
- reason
- source document
- question type, if clearly identifiable

If the question type is genuinely unclear, record it as "unclear" rather than omitting the field.

I will review the borderline cases before the final 50 are locked.

Only after the final 50 are approved should they move to the manual ground-truth verification stage.



==================================================
MANUAL GROUND TRUTH
==================================================

This is VERY IMPORTANT.

I want the final ground-truth answers to be manually verified.

For each final question:

1. Identify the source document.
2. Identify the relevant page, section, heading, or location.
3. Read the relevant source material.
4. Create/verify the ground-truth answer.
5. Ensure the answer is supported ONLY by the source material.
6. Ensure it directly answers the question.
7. Remove the question if a clear source-grounded answer cannot be established.

Ground-truth answers should be:
- concise
- directly responsive to the question
- self-contained
- factually supported by the source
- primarily paraphrased rather than copied verbatim
- detailed enough to include all essential facts needed to judge correctness
- free of information not supported by the source

Do not make answers unnecessarily long.

Do not use direct quotations unless the exact wording itself is important to the question.

Do not automatically trust an LLM-generated answer as ground truth.

The final answer key should be based on the actual source documents.

Guide me through creating the dataset structure and the manual verification workflow.

I will review the excerpt and answer, then either approve or edit the ground-truth answer.

Do not make me manually search through the full documents unless the relevant source excerpt cannot be reliably identified.

==================================================
FINAL DATASET
==================================================

The final dataset should contain exactly 50 questions.

Prefer a structure similar to:

{
    "id": "q001",
    "question": "...",
    "ground_truth": "...",
    "source_document": "...",
    "source_section": "...",
    "source_page": 12
}

If page numbers are unavailable, use the most precise available location.

Preserve useful source metadata such as:
- document title
- section/chapter
- page where available

Explain briefly why each field exists and how the future RAGAS evaluation pipeline will use it.

==================================================
VALIDATION
==================================================

Guide me through validation of the final dataset.

We should verify:

- exactly 50 questions exist
- every question has a ground-truth answer
- every question has a source
- every answer is supported by its source
- no duplicate questions
- no near-duplicate questions
- no ambiguous questions
- no unnecessarily trivial questions
- questions are reasonably distributed across documents
- questions actually test retrieval/context usage

Explain the validation logic before giving me the code.

==================================================
FROZEN DATASET
==================================================

Once the 50 questions are finalized, treat the dataset as a FROZEN evaluation dataset.

Do not regenerate or modify the questions for every experiment.

The same 50 questions must later be used for:

Baseline RAG
        ↓
Evaluation

Semantic chunking
        ↓
Same 50 questions
        ↓
Evaluation

Hybrid retrieval
        ↓
Same 50 questions
        ↓
Evaluation

Reranking
        ↓
Same 50 questions
        ↓
Evaluation

This allows meaningful before/after comparisons.

Do not modify questions or ground-truth answers simply because a particular RAG configuration performs poorly.

==================================================
CODE STYLE
==================================================

Use production-grade architectural decisions with beginner-readable implementation.

Prefer:
- explicit functions
- simple data structures
- readable control flow
- straightforward LangChain components
- simple RAGAS usage
- clear input → processing → output flow

Avoid unnecessary abstractions or complex design patterns.

Do not introduce new technologies, dependencies, frameworks, or architecture unless there is a clear requirement or engineering reason.

If an advanced technique is genuinely necessary:
1. explain the problem it solves
2. explain why we need it
3. explain the simpler alternative
4. then show the simplest understandable implementation

==================================================
MOST IMPORTANT IMPLEMENTATION RULE
==================================================

For each component:

1. Explain what we are building.
2. Explain why we need it.
3. Show where it fits in the pipeline.

==================================================
IMPORTANT ACCURACY RULES
==================================================

Do not hallucinate.

Do not invent:
- documents
- URLs
- licenses
- source locations
- RAGAS APIs
- configuration options
- metrics
- question answers
- dataset contents

If you are unsure about a RAGAS API or current library behavior, verify the current documentation/source before telling me to use it.

If something cannot be verified, say so clearly.

==================================================
DOCUMENTATION
==================================================

After the implementation is complete, guide me in creating concise documentation covering:

- selected documents
- source URLs
- licenses
- domain/topic
- why these documents were selected
- RAGAS synthetic generation process
- number of candidate questions generated
- filtering criteria
- why some questions were removed
- manual ground-truth verification process
- final 50-question dataset structure
- limitations

Do not write the documentation automatically without explaining what should go into it.

==================================================
FINAL STOP CONDITION
==================================================

Do not consider the task complete merely because code exists.

The task is complete only after:

- documents have been selected and obtained
- approximately 60 candidate questions have been generated
- poor/repetitive/ambiguous/trivial candidates have been filtered
- exactly 50 questions remain
- all 50 ground-truth answers have been manually verified
- source references have been verified
- dataset validation has passed
- the frozen dataset is ready for the existing evaluation pipeline


Start by analyzing the awesome-opensource-documents repository.





