# Scope-FedTextual

Anonymous reference implementation of **Scoped-FedTextGrad**, a scope-preserving method for federated textual optimization.

The implementation follows the paper algorithm directly:

1. Each client runs the FedTextGrad-compatible local textual optimizer.
2. Local optimization trajectories are atomized into compact, independently routable rule cards.
3. Every target client measures each rule's validation utility relative to the shared round-start prompt.
4. Rules with utility above `epsilon` are eligible. Local rules are considered first; eligible cross-client rules are ranked by validation utility per exact model token.
5. Each client receives a client-specific prompt that satisfies a hard token budget.
6. **Safe-Scoped** compares the complete scoped prompt with the local prompt on validation data and falls back to Local when the scoped prompt is worse by more than `delta`.

No training, validation, or test example is transmitted by the method. Candidate textual rules and scalar validation statistics are the communicated artifacts.

## Repository layout

```text
configs/
  paper_grid.yaml       Qwen3-8B paper grid
  smoke_mock.yaml       offline smoke test; no API/model calls
scripts/
  run_scoped_grid.py    resumable grid runner
src/
  scoped_fedtextgrad/   atomization, utility, routing, gate, runner
  textgrad/             FedTextGrad-compatible local optimizer
  data/                 official BBH loader and non-IID client construction
  llm/                  cached Responses and vLLM-compatible backends
  evaluation/           deterministic answer extraction and metrics
tests/
  test_scoped_method.py method and offline tests
```

A line-by-line algorithm map is provided in [`docs/METHOD.md`](docs/METHOD.md).

## Installation

Python 3.10 or newer is recommended.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

On Windows PowerShell, activate with `.venv\Scripts\Activate.ps1`.

## Offline verification

The smoke configuration uses deterministic synthetic tasks and a mock backend. It makes no network or LLM calls.

```bash
python -m unittest discover -s tests -v
python scripts/run_scoped_grid.py --config configs/smoke_mock.yaml --dry-run
```

The completed run writes its resolved configuration, candidate rules, client-rule utility matrix, selected/rejected rules, Safe-Scoped gate decisions, final prompts, predictions, metrics, and LLM/cache counters under `results/`.

## Reproducing the main grid

The primary Qwen3-8B factorial grid is declared in `configs/paper_grid.yaml`:

- BBH tasks: Tracking Shuffled Objects (three objects), Movie Recommendation, and Sports Understanding;
- heterogeneity: Task Skew and Compound Skew (`mixed` in configuration files);
- clients: `K in {6, 9}`;
- seeds: `{13, 42, 73}`;
- local TextGrad steps: `E in {1, 3}`;
- local batch sizes: `{1, 3}`.

The official BBH index pools are `0:50` for training, `50:150` for validation, and `150:` for testing. For `K=6`, every client receives 20/20/50 train/validation/test examples; for `K=9`, every client receives 16/20/33. Client allocation is deterministic and without replacement.

`mixed` is Compound Skew, not a mixture of tasks inside each client. Each client still solves exactly one task. Clients additionally receive the shortest or longest unused examples within that task according to the fixed word-count proxy.

### Local vLLM-compatible server

Start the model server separately, then set its URL and API key. The example configuration expects Qwen3-8B at an OpenAI-compatible Chat Completions endpoint.

```bash
export LOCAL_LLM_API_KEY=local-vllm
python scripts/run_scoped_grid.py --config configs/paper_grid.yaml
```

Edit `llm.base_url`, `llm.model_name`, and `llm.tokenizer_name` for the deployed server. Formal prompt-budget claims require the exact tokenizer; the runner fails rather than silently using an estimate when `require_exact_tokenizer: true`.

### Responses-compatible endpoint

Set `llm.provider: openai_responses` and provide credentials through environment variables only:

```bash
export OPENAI_API_KEY=...
export OPENAI_BASE_URL=https://api.openai.com/v1
export OPENAI_MODEL=your-model
```

Never store API keys in YAML or commit `.env` files.

## Method outputs

Each run is self-contained. The main auditable artifacts are:

- `candidate_rules.json`: serialized rule cards and provenance;
- `rule_utility_matrix.csv`: target-client utility for every client-rule pair;
- `selected_rules_per_client.json` and `rejected_rules.json`: complete routing decisions;
- `safe_gate_decisions.json`: validation-only Local/Scoped choices;
- `final_client_prompts.json`: deployed prompts for every method and client;
- `per_client_scores.csv` and `main_results_table.csv`: client and aggregate metrics;
- `resolved_config.json`, `run_status.json`, and `llm_usage.json`: reproducibility metadata.

The scope mask is `S[i,g] = 1[U[i,g] > epsilon]`, where utility is validation-accuracy improvement of `P + g` over the round-start prompt `P`. Test labels are never used for routing or validation gating.

## Upstream fidelity

The local optimizer and common-prompt baselines are protocol-compatible with the official [FedTextGrad repository](https://github.com/ubc-tea/FedTextGrad). The pinned upstream commit is recorded in `upstream.json`; upstream code is not vendored.

## Double-blind release

This repository intentionally contains no author names, affiliations, personal email addresses, private infrastructure paths, API credentials, raw model caches, or private run logs. Please preserve that property in contributions while the work is under double-blind review.
