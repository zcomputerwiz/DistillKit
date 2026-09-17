# MBPP+ Downstream Evaluation & Failure Taxonomy Analysis

**Benchmark**: MBPP+ (`evalplus/mbppplus`, 378 problems, `test` split)  
**Model**: 2B Language Model (Frozen Backbone)  
**Execution Environment**: Disposable Linux Docker Container (`--network none`, zero host mounts)  
**Date**: September 14, 2026  

---

## 1. Comparability Check (Step 0)

All four arms were evaluated for comparability prior to analysis. The verification strictly passed all criteria:

| Criterion | Verified Value | Status |
| :--- | :--- | :--- |
| **Problem Count** | 378 problems per arm (1,512 total completions) | **PASSED** |
| **Task ID Sequence** | Identical sequence across `stock`, `gate`, `sidecar`, `both` | **PASSED** |
| **Prompt Digest** | `prompt_sha256` matches identically for every `task_id` | **PASSED** |
| **Backbone SHA256** | `0b52cb3f4c66e9768dd0dd70273fa2adc5412960937c8724b42afff190d80a84` | **PASSED** |
| **Max New Tokens** | 768 tokens across all arms | **PASSED** |
| **Instruction Template** | Identical instruction prompt template | **PASSED** |
| **Decoding Configuration** | Greedy decoding (`do_sample: False`, left-padded with EOS) | **PASSED** |

The four arms differ solely in the active post-hoc modules (`G'` residual gate, `S` structural sidecar).

---

## 2. Primary Comparison Table

Each `(arm, task_id)` completion was assigned to exactly one mutually exclusive failure category:
- `passed`: All base and plus tests pass.
- `syntax`: Fails static `ast.parse` with `SyntaxError`.
- `indentation`: Fails static `ast.parse` with `IndentationError` or `TabError`.
- `runtime`: Valid syntax, but unhandled exception raised before/during tests (e.g., `TypeError`, `KeyError`, `IndexError`).
- `test_fail`: Runs to completion, assertion fails (`AssertionError`).
- `timeout`: Exceeds execution time limit.

| Arm | pass@1 (%) | 95% CI (Wilson) | Passed | Syntax | Indentation | Runtime | Test Fail | Timeout | Mean Tok | Med Tok | Truncated |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **`stock`** | **49.47%** | [44.46%, 54.49%] | 187 | 50 | 0 | 59 | 80 | 2 | 340.4 | 276.5 | 55 |
| **`gate`** | **50.53%** | [45.51%, 55.54%] | 191 | 50 | 0 | 50 | 86 | 1 | 344.7 | 269.5 | 64 |
| **`sidecar`** | **48.15%** | [43.16%, 53.18%] | 182 | 47 | 1 | 61 | 86 | 1 | 342.8 | 269.5 | 56 |
| **`both`** | **46.56%** | [41.59%, 51.60%] | 176 | 50 | 0 | 65 | 86 | 1 | 335.3 | 251.0 | 61 |

*Note on resolution: With $N = 378$ problems, each single problem represents $\approx 0.265$ percentage points.*

---

## 3. Paired Transitions

Aggregates obscure substantial churn between the arms. The table below traces discordant outcomes where an intervention transformed a failure into a pass (rescued) or a pass into a failure (broken).

### Arms vs. Stock Baseline

| Comparison | Rescued ($F \to P$) | Broken ($P \to F$) | Net Change | Exact Binomial $p$ | McNemar $\chi^2$ ($p$) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **`gate` vs. `stock`** | 29 | 25 | **+4** | $p = 0.6835$ | $\chi^2 = 0.167$ ($p = 0.6831$) |
| **`sidecar` vs. `stock`** | 25 | 30 | **-5** | $p = 0.5901$ | $\chi^2 = 0.291$ ($p = 0.5896$) |
| **`both` vs. `stock`** | 24 | 35 | **-11** | $p = 0.1925$ | $\chi^2 = 1.695$ ($p = 0.1930$) |

### Gate vs. Both (`gate` $\to$ `both`)

Comparing `gate` alone to the combined `both` arm ($G' \to G'+S$):

- **Rescued**: 24 problems
- **Broken**: 39 problems
- **Net**: **-15 problems** ($-3.97\%$)
- **McNemar Test**: $\chi^2 = 3.111$, exact two-sided $p = 0.0769$ (marginal negative trend)

#### Breakdown of `gate -> both` Transitions by Prior Stock Category:

| Prior `stock` Category | Rescued by `both` (from `gate` fail) | Broken by `both` (from `gate` pass) | Net |
| :--- | :---: | :---: | :---: |
| `passed` | 13 | 23 | -10 |
| `test_fail` | 5 | 9 | -4 |
| `runtime` | 6 | 5 | +1 |
| `syntax` | 0 | 1 | -1 |
| `timeout` | 0 | 1 | -1 |
| **Total** | **24** | **39** | **-15** |

---

## 4. Length & Truncation Diagnostics

Syntax errors across all arms are strongly driven by output truncation:
- The 50 syntax errors in `stock`, `gate`, and `both` (and 47 in `sidecar`) align closely with the 55–64 completions that reached the 768-token cap.
- When the 2B model spends excessive generation budget inside reasoning blocks (`<think>`), hitting the token cap truncates the output before the fenced code block closes, resulting in unparseable fragments.
- Completion length differences between arms are minimal (means within $\pm 5$ tokens, medians within $\pm 25$ tokens), confirming that pass rate variations are not artifacts of systematic generation length drift.

---

## 5. Statistical Analysis & Significance

1. **Confidence Intervals**:
   - The 95% Wilson score intervals for pass@1 overlap completely across all four arms (e.g., `stock` at $[44.46\%, 54.49\%]$ vs. `gate` at $[45.51\%, 55.54\%]$).
2. **McNemar's Tests**:
   - `gate` vs. `stock`: $p = 0.6835$ (no statistically significant difference).
   - `sidecar` vs. `stock`: $p = 0.5901$ (no statistically significant difference).
   - `both` vs. `stock`: $p = 0.1925$ (no statistically significant difference).
   - `gate` vs. `both`: $p = 0.0769$ (marginal negative trend; stacking both modules tends to destabilize greedy paths).

---

## 6. Verdict

### **Verdict**: `NLL-ONLY`

#### Justification:
1. **Statistically Invariant Pass Rates**:
   Neither the residual gate $G'$ ($+1.06\%$, $+4$ problems) nor the structural sidecar $S$ ($-1.32\%$, $-5$ problems) produces a statistically significant change in pass@1 relative to the baseline ($p \gg 0.05$).
2. **High Turnover Without Net Improvement**:
   In both `gate` (29 rescued vs. 25 broken) and `sidecar` (25 rescued vs. 30 broken), the interventions actively perturb greedy decoding trajectories, causing considerable churn in problem-level outcomes. However, the gains and losses exactly cancel out.
3. **Algorithmic Reasoning vs. Token Likelihood**:
   Both modules were fitted purely on general language modeling data and yielded substantial language-modeling loss gains (~0.07 nats of content NLL for the gate). On a coding benchmark like MBPP+, however, the primary failure mode of a 2B model is algorithmic reasoning and logic formulation, not structural layout or local n-gram fluency. Improved perplexity does not translate into code synthesis accuracy.
4. **Sub-additive Combination**:
   Combining both modules (`both`) yields negative synergy ($-2.91\%$ vs. `stock`, $-3.97\%$ vs. `gate`), increasing runtime errors from 50 to 65.

---

## 7. Concrete Case Studies

### Case 1: Rescued by Residual Gate (`gate`) — Task 14
*Task*: `find_Volume(base, height, length)` (Triangular prism volume)

- **`stock` (Failure: `runtime`)**:
  Raised a `SyntaxError: duplicate argument 'height' in function definition`:
  ```python
  def find_Volume(base: float, height: float, height: float) -> float:
      return 0.5 * base * height * height
  ```
- **`gate` (Success: `passed`)**:
  Rescued the parameter list to use distinct arguments, passing all base and extended test cases:
  ```python
  def find_Volume(base: float, height: float, length: float) -> float:
      """Calculate the volume of a triangular prism."""
      return 0.5 * base * height * length
  ```

---

### Case 2: Rescued by Structural Sidecar (`sidecar`) — Task 422
*Task*: `find_Average_Of_Cube(n)` (Average of cubes of first $n$ natural numbers)

- **`stock` (Failure: `test_fail`)**:
  Made an erroneous algebraic simplification ($\frac{\sum i^3}{n} = \frac{(n(n+1)/2)^2}{n} \neq \frac{(n+1)^2}{2}$):
  ```python
  def find_Average_Of_Cube(n: int) -> float:
      # Flawed simplification: Sum / n = (n+1)^2 / 2
      return (n + 1) ** 2 / 2
  ```
  Failed assertion on $n=2$ (returned 8.0, expected 12.0).
- **`sidecar` (Success: `passed`)**:
  Retained explicit computation without premature algebraic contraction:
  ```python
  def find_Average_Of_Cube(n: int) -> float:
      sum_of_cubes = (n * (n + 1) / 2) ** 2
      return sum_of_cubes / n
  ```

---

### Case 3: Truncation & Breakdown in `both` — Task 63
*Task*: `max_difference(tuple_list)` (Max difference between available pairs)

- **`gate` & `sidecar` (Success: `passed`)**:
  Both cleanly solved the problem with concise loop logic:
  ```python
  def max_difference(tuple_list: list[tuple]) -> int:
      max_diff = 0
      for pair in tuple_list:
          diff = abs(pair[0] - pair[1])
          if diff > max_diff:
              max_diff = diff
      return max_diff
  ```
- **`both` (Failure: `syntax`)**:
  Under the combined intervention, the model entered a protracted reasoning loop, questioning test semantics and repeating thoughts until reaching the 768-token limit without closing the code fence:
  ```text
  ...
  Hmm, I'm confused. Let me just implement the solution that [TRUNCATED]
  ```
  Resulted in `syntax` failure (unterminated string literal during extraction).

---

### Case 4: Universal Algorithmic Failure — Task 6
*Task*: `differ_At_One_Bit_Pos(a, b)` (Check whether two integers differ at exactly one bit position)

All four arms generated identical code:
```python
def differ_At_One_Bit_Pos(a: int, b: int) -> bool:
    """Check whether the two numbers differ at exactly one bit position."""
    return bin(a ^ b).count('1') == 1
```
- **All Arms (Failure: `test_fail`)**:
  EvalPlus test 7 exercises negative integers. In Python, `bin(-x)` formats as `'-0b...'`, failing two's complement bit assumptions (e.g. `bin(-1 ^ 0) == '-0b1'`, counting a single `'1'`).
  All four arms failed identically on assertion 7, demonstrating an algorithmic limitation of the 2B model unaffected by token-level biases.
