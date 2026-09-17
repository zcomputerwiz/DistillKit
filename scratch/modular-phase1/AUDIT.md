# Phase 1 audit

**Corrected conclusion: the Phase 1 verdict remains failed composition.** No checkpoint was retrained or modified. The specialist passed only the implemented 0.95 macro-F1 gate; it did not satisfy a 0.99 event-accuracy requirement. The reader consumes changed pointer inputs, but answer probabilities are effectively insensitive to a one-reference, opposite-value lookup intervention.

## 1. Specialist gate reconciliation

Implemented threshold: macro-F1 >= 0.95, excluding pad and whitespace. Reconstructed macro-F1: 0.970893; structural-event accuracy excluding pad/whitespace: 0.984787 (overall token accuracy 0.985228). Both structural metrics are below 0.99. Binding remains 1.000000.

The historical `passed` field is retained because it reflects the implemented criterion. Treating it as proof of >=99% structural-event accuracy was incorrect.

| Event | Support | Predicted | TP | FP | FN | Precision | Recall | F1 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| bos | 2048 | 2048 | 2048 | 0 | 0 | 1.0000 | 1.0000 | 1.0000 |
| eos | 1863 | 1876 | 1846 | 30 | 17 | 0.9840 | 0.9909 | 0.9874 |
| whitespace | 73906 | 72915 | 72861 | 54 | 1045 | 0.9993 | 0.9859 | 0.9925 |
| open | 3934 | 3931 | 3931 | 0 | 3 | 1.0000 | 0.9992 | 0.9996 |
| close | 3696 | 3763 | 3624 | 139 | 72 | 0.9631 | 0.9805 | 0.9717 |
| let | 14456 | 14437 | 14436 | 1 | 20 | 0.9999 | 0.9986 | 0.9993 |
| declaration | 14433 | 13519 | 13518 | 1 | 915 | 0.9999 | 0.9366 | 0.9672 |
| assign | 14418 | 14392 | 14392 | 0 | 26 | 1.0000 | 0.9982 | 0.9991 |
| value | 16379 | 16347 | 16347 | 0 | 32 | 1.0000 | 0.9980 | 0.9990 |
| query | 1921 | 1913 | 1908 | 5 | 13 | 0.9974 | 0.9932 | 0.9953 |
| use | 2324 | 3187 | 2296 | 891 | 28 | 0.7204 | 0.9880 | 0.8332 |
| xor | 2413 | 2419 | 2403 | 16 | 10 | 0.9934 | 0.9959 | 0.9946 |
| semicolon | 16227 | 16177 | 16176 | 1 | 51 | 0.9999 | 0.9969 | 0.9984 |
| answer_marker | 1863 | 1885 | 1847 | 38 | 16 | 0.9798 | 0.9914 | 0.9856 |
| output | 1863 | 2115 | 1846 | 269 | 17 | 0.8728 | 0.9909 | 0.9281 |
| invalid | 7989 | 8809 | 7599 | 1210 | 390 | 0.8626 | 0.9512 | 0.9048 |

Validation slices:

| Slice | Documents | Tokens | Overall accuracy | Structural accuracy | Macro-F1 |
| --- | ---: | ---: | ---: | ---: | ---: |
| all | 2048 | 179733 | 0.9852 | 0.9848 | 0.9709 |
| clean | 1863 | 163736 | 0.9877 | 0.9884 | 0.9810 |
| corrupted | 185 | 15997 | 0.9600 | 0.9576 | 0.8779 |
| depth_0 | 410 | 9976 | 0.9772 | 0.9780 | 0.9568 |
| depth_1 | 412 | 40945 | 0.9967 | 0.9975 | 0.9891 |
| depth_2 | 430 | 43566 | 0.9973 | 0.9965 | 0.9947 |
| depth_3 | 438 | 47189 | 0.9913 | 0.9908 | 0.9842 |
| depth_4 | 358 | 38057 | 0.9535 | 0.9521 | 0.9314 |
| literal | 410 | 9976 | 0.9772 | 0.9780 | 0.9568 |
| no_shadowing | 431 | 11274 | 0.9779 | 0.9793 | 0.9600 |
| query_literal | 17 | 173 | 0.9595 | 1.0000 | 0.9786 |
| query_literal_xor | 393 | 9803 | 0.9776 | 0.9776 | 0.9565 |
| query_lookup | 745 | 77817 | 0.9825 | 0.9838 | 0.9632 |
| query_xor | 893 | 91940 | 0.9884 | 0.9864 | 0.9782 |
| scope_resolution | 1638 | 169757 | 0.9857 | 0.9852 | 0.9735 |
| shadowing | 1617 | 168459 | 0.9857 | 0.9852 | 0.9734 |
| whitespace_mixed | 567 | 47927 | 0.9852 | 0.9853 | 0.9711 |
| whitespace_newlines | 472 | 40636 | 0.9896 | 0.9895 | 0.9803 |
| whitespace_spaces | 522 | 47209 | 0.9846 | 0.9865 | 0.9684 |
| whitespace_tabs | 487 | 43961 | 0.9819 | 0.9779 | 0.9651 |

## 2. Controlled pointer intervention

Eligible lookup-only cases: 302 of 304 base lookup documents; 2718 checkpoint/case pairs. Every case changes exactly one reference to a declaration holding the opposite bit; XOR is excluded (149 0->1 and 153 1->0 documents).

Mean change in full-vocabulary correct-answer probability: +0.00001092 (mean absolute 0.00006976, range -0.00146955 to +0.00284006). Mean change after normalizing over bit tokens only: +0.00001153 (mean absolute 0.00005954). Top-1 answer accuracy changed by exactly zero in all nine checkpoints.

Exact input checks across 2718 checkpoint/case pairs: pointer index changed 2718; gather indices changed 2718; gathered embedding state changed 2718; weighted context changed 2718; final reader output changed 2718. No specialist feature or gathered-state cache exists.

## 3. How pointers enter

Pointers are deterministic token indices from the programmed lexical-scope table, not learned predictions. At each use, the reader gathers a seven-token window of raw token-plus-position embeddings beginning at the declaration identifier, scores that window, forms a weighted context, gates/scales it, and adds it to a learned projection of structural features. Pointer distance is also an encoded scalar in the feature vector. Learned fields are event probabilities, validity probability, and confidence; programmed fields are exact depth, pointer presence/distance, and the pointer address.

## 4. Oracle features and invariance localization

Oracle causal event/validity labels were passed through the same 30-column interface, frozen reader, and frozen backbone. Semantic binding comparisons use declaration ordinals rather than raw token positions.

| Size | Mode | Accuracy | Answer NLL | Overall NLL | Rename agreement | Whitespace agreement |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| l2_w64 | learned | 0.0552 | 2.6685 | 1.9151 | 0.2294 | 0.2144 |
| l2_w64 | oracle | 0.0514 | 2.6759 | 1.9147 | 0.2381 | 0.2109 |
| l2_w64 | plain existing | - | - | - | 0.2251 | 0.2109 |
| l4_w128 | learned | 0.1681 | 1.8825 | 1.4683 | 0.3117 | 0.2648 |
| l4_w128 | oracle | 0.1621 | 1.9640 | 1.4598 | 0.2597 | 0.2422 |
| l4_w128 | plain existing | - | - | - | 0.3117 | 0.2743 |
| l6_w256 | learned | 0.3556 | 1.1923 | 1.1749 | 0.5541 | 0.5312 |
| l6_w256 | oracle | 0.3535 | 1.2261 | 1.1753 | 0.5541 | 0.5191 |
| l6_w256 | plain existing | - | - | - | 0.5455 | 0.5078 |

Aligned specialist features:

- whitespace: semantic binding ordinal agreement 1.0000; programmed depth agreement 1.0000; learned event argmax agreement 0.9946; event-probability MAE 0.001669.
- renamed: semantic binding ordinal agreement 1.0000; programmed depth agreement 1.0000; learned event argmax agreement 0.9935; event-probability MAE 0.001933.

## 5. Synchronized online timing

Device: NVIDIA GeForce RTX 3090.

| Size | Specialist + state (us/token) | Reader | Backbone | Sum | Overhead vs backbone |
| --- | ---: | ---: | ---: | ---: | ---: |
| l2_w64 | 4.225 | 2.842 | 6.582 | 13.648 | 107.4% |
| l4_w128 | 5.157 | 2.787 | 12.634 | 20.579 | 62.9% |
| l6_w256 | 4.438 | 2.296 | 16.899 | 23.633 | 39.9% |

Each component was warmed for 20 iterations, timed for 100, and bounded by torch.cuda.synchronize on cuda:0. State updates were timed on CPU. Transfers and tokenization are excluded; all learned features are recomputed.

## Code references

- programmed_state_and_pointers: `D:\DeepThought\Projects\HybridModel\DistillKit\experiments\modular_phase1\specialist.py:55`
- learned_interface: `D:\DeepThought\Projects\HybridModel\DistillKit\experiments\modular_phase1\specialist.py:244`
- oracle_interface: `D:\DeepThought\Projects\HybridModel\DistillKit\experiments\modular_phase1\specialist.py:275`
- reader_gather: `D:\DeepThought\Projects\HybridModel\DistillKit\experiments\modular_phase1\models.py:111`
- composed_forward: `D:\DeepThought\Projects\HybridModel\DistillKit\experiments\modular_phase1\models.py:166`
- audit_entrypoint: `D:\DeepThought\Projects\HybridModel\DistillKit\experiments\modular_phase1\audit.py:950`
- frozen_config: `D:\DeepThought\Projects\HybridModel\DistillKit\experiments\modular_phase1\configs\pilot.json`

## Final corrected conclusions

- Preserve the original negative quality and cost verdict.
- The specialist checkpoint met its implemented 0.95 macro-F1 gate, not a 0.99 structural-event criterion.
- Programmed semantic bindings are invariant after remapping positions. Any learned feature drift is small, and oracle features do not repair output invariance. The similar plain-backbone failures localize renaming/whitespace sensitivity primarily to the raw-token/position backbone, not the specialist.
- The trained reader receives genuinely different pointer-derived tensors yet has negligible answer-probability sensitivity to an opposite-value lookup pointer. It did not learn useful causal pointer retrieval; this is a separate reader failure.
- No further training or architecture change is justified by this audit alone.

Reproduce the audit (evaluation only):

`python -m experiments.modular_phase1.cli --config experiments/modular_phase1/configs/pilot.json --run-dir scratch/modular-phase1 audit`
