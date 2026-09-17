# Phase 1e category admission

**Acceptance: FAIL.** All Phase 1 through 1d artifacts and verdicts remain unchanged.

## Calibration-only arm selection

| Condition | Answer accuracy | Answer NLL | Overall NLL | Whitespace NLL | False-bit rate |
| --- | ---: | ---: | ---: | ---: | ---: |
| unchanged_phase1d | 0.7591 | 0.842052 | 0.950807 | 0.735118 | 0.0000 |
| constant | 0.7669 | 0.665151 | 0.950150 | 0.879000 | 0.0342 |
| contextual | 0.7617 | 0.703788 | 0.950071 | 0.817677 | 0.0000 |

Selected on calibration: `None`. Confirmation was evaluated zero times because neither arm qualified.

| Arm | Parameters | Updates | Causal-target exposures | Answer targets | Train seconds |
| --- | ---: | ---: | ---: | ---: | ---: |
| constant | 1 | 176 | 1048499 | 11220 | 0.472 |
| contextual | 27 | 176 | 1048516 | 11225 | 0.527 |

Calibration interventions retained correct conditional changes in 1.0000 for both arms across 248 cases. Prefix causality passed 2265 comparisons.

Maximum contextual within-bit/non-bit conditional changes were 8.941e-08/2.384e-07; inactive identity was exact. Contextual admission overhead was 9.248 us per active prefix.

Frozen backbone/specialist/readout hashes: `0c81b60f20fdb790dea46d76fc1146f36611d993b9ebdb76d8eeca3c246f0e7f` / `2c64666f579a89def2a66b900fff4113ffac8f8c0ef12203dd7d83c39fad2db3` / `eb101ea56901fd6d6f2e5d525dc6d5ab2092ea73e8053487879aca02daaadb98`. Admission checkpoint hash: `237f8068d1d936643f5dca8379e4bbdfc52374d8e5c2cf42728fde9af31ce525`. All prior artifact hashes are unchanged.

Passing establishes useful category admission only on this frozen backbone. Cross-scale reuse, matched-quality efficiency, XOR, and backbone-layer integration remain out of scope.

Reproduce:

`python -m experiments.modular_phase1e --config experiments/modular_phase1e/configs/diagnostic.json --run-dir scratch/modular-phase1e --force`
