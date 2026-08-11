# Forbidden prose patterns

Apply this file to README files, design documents, comments, reports, and release
notes. Add a pattern when it survives review twice or makes a claim harder to
verify.

The standard is simple: name the actor, action, evidence, and limit in the fewest
words that preserve meaning.

## Banned forms

### Staccato pairs

Short fragments arranged for drama waste space and hide the relationship between
ideas.

Bad:

> The result? Clear. The cost? High.

Use:

> The result was clear and expensive.

### Antithesis reframes and negative parallelism

Do not introduce a claim by theatrically denying a weaker claim. State the claim
directly.

Bad:

> This is not a leaderboard. It is a decision tool.

Use:

> This study estimates the value of retrieval for each model.

Also reject repeated forms such as “not X, but Y,” “less X, more Y,” and “doesn't
just X; it Y.” Keep a contrast only when both sides are necessary facts.

### Isocolon metaphor-pairs

Parallel slogans sound polished while saying little.

Bad:

> A lens for quality, a map for cost.

Use:

> The report compares accuracy and cost.

### Backward references

Phrases such as “as noted above,” “the former,” “the latter,” and “the point
below” force the reader to search for context.

Bad:

> Apply the former rule before publication.

Use:

> Pin the dataset version before publication.

Use a link or repeat the exact noun when the target is outside the paragraph.

### Throat clearing

Delete openings that announce the importance, obviousness, or purpose of the
sentence.

Bad:

> It is important to note that the dataset must be pinned.

Use:

> Pin the dataset.

Common offenders include “the point is,” “why this matters,” “it should be noted,”
“in order to,” and “the fact that.”

### Unsupported superlatives

Reject “best,” “sharpest,” “most important,” “unique,” “unprecedented,” and
“nobody does this” unless a cited comparison establishes the claim.

Bad:

> This is the sharpest test available.

Use:

> This contrast pairs each model with its no-search baseline.

### Inflated abstractions

Prefer the concrete field, action, or result over an abstract label.

Bad:

> The instrumentation creates an observability contribution.

Use:

> Each row records which search-result fields the provider returned.

Watch for “framework,” “paradigm,” “surface,” “mechanism,” “contribution,” and
“dimension.” Technical terms may stay when the code or analysis defines them.

### Repeated caveats

State a limit once in the section where a reader will use it. Link to that
section elsewhere. Repetition makes the document longer and lets wording drift.

### Vague actors and verbs

Reject sentences where the reader cannot tell who does what.

Bad:

> It is handled during processing.

Use:

> `analyze_results.py` drops unpaired rows during analysis.

### Decorative rhetoric

Remove rhetorical questions, dramatic commands, metaphors, and editorial asides
from technical prose.

Bad:

> Do not let this become a leaderboard.

Use:

> Do not compare native search across vendors because the model changes too.

### Claims without a boundary

Name the models, dataset, date, and metric that support a result. Avoid claims
about broad classes when the study tests one member.

Bad:

> Independent search beats native search.

Use:

> On the pinned LiveNewsBench snapshot, the You.com harness arm improved paired
> answer match over the OpenAI native arm by X points.

### Precision without provenance

Prices, model names, API behavior, dataset sizes, and live-check results need a
source in code, a pinned artifact, or a dated verification note. Remove stale
numbers instead of polishing them.

## Editing pass

Before merging prose:

1. Delete repeated claims and caveats.
2. Replace vague references with exact nouns.
3. Split sentences that carry more than one condition or exception.
4. Remove slogans, scene-setting, and self-evaluation.
5. Check every number, model name, command, link, and date against the repository.
6. Read the text aloud. Rewrite any sentence that sounds like a pitch.
7. Add newly observed slop to this file with a bad example and a direct
   replacement.
