# Method-to-code map

This note maps the paper formulation to the public implementation. It makes the algorithm auditable without requiring private experiment infrastructure.

## Inputs

- clients `1,...,K`, each with local train, validation, and test data;
- a shared round-start prompt `P`;
- prompt token budget `B`;
- scope threshold `epsilon`;
- optional Safe-Scoped margin `delta`.

## 1. Local textual optimization and atomization

`src/textgrad/local_optimizer.py` follows the pinned FedTextGrad forward, feedback, TGD-update, same-batch evaluation, and proximal-revert loop. Each local step records a `GradientCard` containing the source client, feedback, candidate rule, evidence IDs, local scores, source task, and rule-token length.

`src/scoped_fedtextgrad/rule_extraction.py` converts each candidate into one portable imperative rule. The output is normalized and hard-capped with the target model tokenizer. The main router does not use source-task identity to infer applicability.

## 2. Client-dependent scope

For every target client `i` and candidate rule `g`, `src/scoped_fedtextgrad/utility.py` evaluates:

```text
U[i,g] = Accuracy_i(P + g) - Accuracy_i(P)
S[i,g] = 1[U[i,g] > epsilon]
```

The baseline `Accuracy_i(P)` is evaluated once per target client and reused across rules. Validation examples remain local; the coordinator receives scalar utilities.

## 3. Hard-budget routing

`src/scoped_fedtextgrad/selection.py` implements the deterministic first-order routing policy:

```text
eligible = [g for g in all_rules if U[i,g] > epsilon]
local    = sort(eligible from client i, by U[i,g] / tokens(g), descending)
shared   = sort(other eligible rules, by U[i,g] / tokens(g), descending)

selected = []
for g in local + shared:
    if tokens(compose(P, selected + [g])) <= B:
        selected.append(g)
```

The fit check counts the complete composed prompt, including the base prompt and section syntax. Singleton utilities are used only as a practical routing score; the implementation does not assume that utilities remain additive after composition.

## 4. Safe-Scoped validation gate

`src/scoped_fedtextgrad/gate.py` implements:

```text
if Acc_val(scoped_prompt) < Acc_val(local_prompt) - delta:
    final_prompt = local_prompt
else:
    final_prompt = scoped_prompt
```

The runner applies this decision before test evaluation. Test labels are never available to the router or gate.

## 5. End-to-end runner

`src/scoped_fedtextgrad/runner.py` executes the complete sequence and writes all intermediate artifacts. `scripts/run_scoped_grid.py` expands the declared factorial grid, chooses per-client sample sizes that preserve disjoint official BBH pools, skips already completed cells, and records failures without discarding earlier results.

## Main invariants

- scope is target-client validation utility, not source-task metadata or text similarity;
- all selected rules satisfy `U[i,g] > epsilon`;
- locally sourced eligible rules are considered before cross-client rules;
- the exact complete prompt never exceeds `B` when a finite budget is used;
- Safe-Scoped uses validation data only;
- client examples are allocated without replacement and are not communicated;
- every run stores its resolved configuration and complete routing decisions.
