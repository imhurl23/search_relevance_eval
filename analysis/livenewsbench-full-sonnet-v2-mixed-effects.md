# Task-random-intercept mixed-effects regression

## Specification

A binomial logistic mixed model predicts `gated_answer_match` from the 14 deployed model-by-retrieval conditions, with a random intercept for `task_key`. The model uses a variational-Bayes fit from `statsmodels`. Instrumentation-truncated rows are excluded as preregistered. Intervals below are approximate 95% posterior intervals from the mean-field variational approximation.

- Rows: 18,412
- Unique tasks: 1,129
- Excluded truncated rows: 194
- Estimated task-intercept SD: 4.110
- Latent-scale task ICC: 83.7%

## Registered contrasts

Odds ratios above 1 favor the first arm named in the comparison.

| Model | Comparison | Odds ratio | Approx. 95% interval | P(effect > 0) |
|---|---|---:|---:|---:|
| DeepSeek V4 | Normalized vs no search | 240.35 | [161.10, 358.59] | 1.000 |
| DeepSeek V4 | Wide vs normalized | 1.05 | [0.78, 1.40] | 0.620 |
| GLM-5.2 | Normalized vs no search | 1468.16 | [833.71, 2585.42] | 1.000 |
| GLM-5.2 | Wide vs normalized | 0.99 | [0.74, 1.33] | 0.477 |
| GPT-5.6 Terra | Normalized vs no search | 9.96 | [7.47, 13.28] | 1.000 |
| GPT-5.6 Terra | Wide vs normalized | 0.98 | [0.73, 1.31] | 0.441 |
| GPT-5.6 Terra | Native vs normalized | 0.62 | [0.46, 0.82] | 0.001 |
| Claude Sonnet 5 | Normalized vs no search | 35.19 | [25.80, 47.99] | 1.000 |
| Claude Sonnet 5 | Wide vs normalized | 0.97 | [0.72, 1.30] | 0.414 |
| Claude Sonnet 5 | Native vs normalized | 0.87 | [0.65, 1.17] | 0.177 |

## Condition estimates

`Conditional probability` evaluates each condition at a task random intercept of zero; it is not a population-marginal probability.

| Condition | n | Observed accuracy | Conditional probability |
|---|---:|---:|---:|
| DeepSeek V4 \| No search | 1,160 | 4.9% | 0.1% |
| DeepSeek V4 \| Normalized | 1,315 | 36.1% | 15.7% |
| DeepSeek V4 \| Wide | 1,324 | 36.3% | 16.3% |
| GLM-5.2 \| No search | 1,329 | 1.4% | 0.0% |
| GLM-5.2 \| Normalized | 1,328 | 35.6% | 15.2% |
| GLM-5.2 \| Wide | 1,329 | 35.6% | 15.1% |
| GPT-5.6 Terra \| No search | 1,329 | 26.2% | 4.3% |
| GPT-5.6 Terra \| Normalized | 1,329 | 41.9% | 31.0% |
| GPT-5.6 Terra \| Wide | 1,329 | 41.8% | 30.6% |
| GPT-5.6 Terra \| Native | 1,329 | 38.6% | 21.7% |
| Claude Sonnet 5 \| No search | 1,328 | 12.0% | 0.5% |
| Claude Sonnet 5 \| Normalized | 1,328 | 35.4% | 14.8% |
| Claude Sonnet 5 \| Wide | 1,329 | 35.2% | 14.4% |
| Claude Sonnet 5 \| Native | 1,326 | 34.5% | 13.1% |

## Interpretation limits

The model accounts for repeated conditions on the same task, but it does not turn post-treatment retrieval metrics into causal mediators. Native conditions remain structurally unavailable for DeepSeek and GLM. Variational-Bayes intervals may be narrower than likelihood-based intervals, so the task-bootstrap analysis remains the registered reproducible baseline.

---

## Attribution

Questions and reference answers come from LiveNewsBench, release `jan_2026_release_2`, pinned at commit `8a6b96e`, MIT licensed. The benchmark is the work of Yunfan Zhang, Kathleen McKeown, and Smaranda Muresan — [arXiv:2602.13543](https://arxiv.org/abs/2602.13543). Cite it alongside any result taken from this report; the BibTeX entry is in the [repository README](../README.md#citation).

The rows carry the upstream canary asking that the benchmark never enter a training corpus. These runs use five searches and zero page visits, which differs from the benchmark's default allowance of five each. The matrix spans all four splits of the release, and LiveNewsBench defines its splits by event age, so `livenewsbench_split` and event age are close to collinear.
