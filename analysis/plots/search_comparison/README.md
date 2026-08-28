# You.com normalized vs provider-native search

These figures compare the shared You.com harness against each vendor's own
search toolchain, on the two models that have both: GPT-5.6 Terra and Claude
Sonnet 5. They are built from the tables that `analysis/analyze_variables.py`
writes, and the values behind them are in `search_comparison_plot_data.csv`.

Read the Native comparison as a system comparison. The native arms can open
pages and the harness arms cannot, so the contrast bundles a capability
difference with the search-implementation difference. The
[repository README](../../../README.md#conformance-with-the-benchmarks-defaults)
records the observed counts.

## Lead figure

![Editorial comparison](00_search_comparison_editorial_hero.png)

**Suggested caption:** The normalized You.com harness is faster for both models.
It also improves GPT-5.6 Terra's answer match, while Claude Sonnet 5's quality is
statistically tied with provider-native search.

This is the recommended visual for the main discussion. It shows the complete
argument in one reading direction: native in pink, You.com in indigo, and
improvement moving left-to-right in both panels.

## Supporting detail

### Paired quality intervals

![Paired quality effects](01_native_vs_you_quality_effects.png)

**Suggested caption:** On GPT-5.6 Terra, the normalized You.com harness improves
strict answer match by 3.1 percentage points and semantic-judge accuracy by 3.5
points versus provider-native search; both task-bootstrap intervals exclude
zero. Claude Sonnet 5 moves in the same direction, but neither accuracy interval
excludes zero.

### Paired latency intervals

![Paired latency savings](02_native_vs_you_latency_savings.png)

**Suggested caption:** The normalized You.com harness is faster for both models:
4.0 seconds per answer for GPT-5.6 Terra and 1.4 seconds for Claude Sonnet 5.
Both 95% task-bootstrap intervals exclude zero.

### Quality-latency decision space

![Quality and latency decision space](03_quality_latency_decision_space.png)

**Suggested caption:** Across 1,329 rows per condition, switching from
provider-native search to the normalized You.com harness moves both models
toward the preferred upper-left region: higher strict answer match and lower
mean latency.

## Notes

Quote the values in these figures and in `search_comparison_plot_data.csv`.
Earlier drafts circulated different condition means; `variable_condition_summary.csv`
is the reproducible source, and it puts GPT You.com latency at 9.7 seconds.

These figures omit cost. The root-span export they were built from carries no
trace-aggregated cost field. Per-row cost is available in the release table that
`analysis/build_hf_dataset.py` writes.

Recreate every figure:

```bash
.venv/bin/python analysis/plot_search_comparison.py
```

## Attribution

Questions and reference answers come from LiveNewsBench, MIT licensed, the work
of Yunfan Zhang, Kathleen McKeown, and Smaranda Muresan —
[arXiv:2602.13543](https://arxiv.org/abs/2602.13543). Cite the benchmark
alongside any figure reused from here; the BibTeX entry is in the
[repository README](../../../README.md#citation).
