# Modular Phase 1 result

**Verdict: failed composition.** The specialist cleared its gate, but predeclared matched-quality checks failed.

Runs: 27/27 trained, 27/27 evaluated.

## Frozen specialist

Checkpoint SHA-256: `e0dcb68ba8c236533460ac2a8c2ac7f4319d4c4585cdd5ed4fc85ae26578ade0`

Learned parameters: 12,652; binding accuracy: 1.0000; structural-event macro-F1: 0.9709; depth accuracy: 0.8742; validity accuracy: 0.9719.

## Non-inferiority (specialist minus matched)

### l2_w64

Decision: fail (2/12 checks).

| Slice | Accuracy diff [95% CI] | Answer NLL diff [95% CI] | Overall NLL diff [95% CI] |
| --- | ---: | ---: | ---: |
| confirmation | -0.0022 [-0.0135, +0.0072] | -0.0235 [-0.0783, +0.0183] | +0.0658 [+0.0480, +0.0791] |
| depth_ood_5_8 | +0.0146 [-0.0017, +0.0341] | -0.0423 [-0.1221, +0.0377] | +0.0506 [+0.0436, +0.0593] |
| long_history | +0.0078 [-0.0157, +0.0286] | -0.0371 [-0.1045, +0.0299] | +0.0543 [+0.0413, +0.0645] |
| heldout_combo | -0.0065 [-0.0221, +0.0091] | -0.0518 [-0.1466, +0.0201] | +0.0559 [+0.0450, +0.0685] |

### l4_w128

Decision: fail (8/12 checks).

| Slice | Accuracy diff [95% CI] | Answer NLL diff [95% CI] | Overall NLL diff [95% CI] |
| --- | ---: | ---: | ---: |
| confirmation | +0.0642 [+0.0337, +0.1010] | -0.2704 [-0.4339, -0.1585] | +0.0213 [+0.0092, +0.0385] |
| depth_ood_5_8 | +0.0520 [+0.0098, +0.0992] | -0.2482 [-0.4094, -0.0967] | +0.0796 [+0.0562, +0.1069] |
| long_history | +0.0729 [+0.0182, +0.1367] | -0.3603 [-0.7081, -0.1604] | +0.0467 [+0.0054, +0.0747] |
| heldout_combo | +0.0430 [+0.0000, +0.0951] | -0.2056 [-0.3349, -0.0970] | +0.0217 [-0.0081, +0.0424] |

### l6_w256

Decision: fail (7/12 checks).

| Slice | Accuracy diff [95% CI] | Answer NLL diff [95% CI] | Overall NLL diff [95% CI] |
| --- | ---: | ---: | ---: |
| confirmation | +0.0065 [-0.0444, +0.0599] | -0.0943 [-0.1655, -0.0183] | -0.0291 [-0.0393, -0.0136] |
| depth_ood_5_8 | +0.0000 [-0.1333, +0.1317] | -0.0281 [-0.1040, +0.0612] | -0.0161 [-0.0326, -0.0020] |
| long_history | +0.0690 [+0.0039, +0.1563] | -0.3769 [-0.6323, -0.1874] | -0.0253 [-0.0656, +0.0093] |
| heldout_combo | -0.0143 [-0.1081, +0.0716] | -0.0560 [-0.3196, +0.1479] | -0.0243 [-0.0450, +0.0072] |

## Absolute confirmation metrics (mean across seeds)

| Size | Arm | Slice | Accuracy | Answer NLL | Overall NLL |
| --- | --- | --- | ---: | ---: | ---: |
| l2_w64 | plain | confirmation | 0.0444 | 2.7148 | 1.9220 |
| l2_w64 | plain | depth_ood_5_8 | 0.0390 | 2.7581 | 1.9797 |
| l2_w64 | plain | long_history | 0.0469 | 2.7277 | 1.9505 |
| l2_w64 | plain | heldout_combo | 0.0352 | 2.7761 | 1.7756 |
| l2_w64 | specialist | confirmation | 0.0552 | 2.6685 | 1.9151 |
| l2_w64 | specialist | depth_ood_5_8 | 0.0439 | 2.7139 | 1.9765 |
| l2_w64 | specialist | long_history | 0.0586 | 2.6808 | 1.9442 |
| l2_w64 | specialist | heldout_combo | 0.0430 | 2.7185 | 1.7708 |
| l2_w64 | matched | confirmation | 0.0575 | 2.6920 | 1.8639 |
| l2_w64 | matched | depth_ood_5_8 | 0.0293 | 2.7562 | 1.9265 |
| l2_w64 | matched | long_history | 0.0508 | 2.7179 | 1.9070 |
| l2_w64 | matched | heldout_combo | 0.0495 | 2.7703 | 1.7158 |
| l4_w128 | plain | confirmation | 0.1051 | 2.1045 | 1.4282 |
| l4_w128 | plain | depth_ood_5_8 | 0.1220 | 2.1705 | 1.4417 |
| l4_w128 | plain | long_history | 0.1393 | 2.1441 | 1.7612 |
| l4_w128 | plain | heldout_combo | 0.1107 | 2.2612 | 1.2425 |
| l4_w128 | specialist | confirmation | 0.1681 | 1.8825 | 1.4683 |
| l4_w128 | specialist | depth_ood_5_8 | 0.1707 | 1.9707 | 1.5321 |
| l4_w128 | specialist | long_history | 0.2109 | 1.8526 | 1.8158 |
| l4_w128 | specialist | heldout_combo | 0.1393 | 2.1162 | 1.2798 |
| l4_w128 | matched | confirmation | 0.1039 | 2.1529 | 1.4272 |
| l4_w128 | matched | depth_ood_5_8 | 0.1187 | 2.2189 | 1.4447 |
| l4_w128 | matched | long_history | 0.1380 | 2.2129 | 1.7514 |
| l4_w128 | matched | heldout_combo | 0.0964 | 2.3218 | 1.2550 |
| l6_w256 | plain | confirmation | 0.3618 | 1.3294 | 1.1980 |
| l6_w256 | plain | depth_ood_5_8 | 0.3610 | 1.3575 | 1.1443 |
| l6_w256 | plain | long_history | 0.3021 | 1.7545 | 1.6524 |
| l6_w256 | plain | heldout_combo | 0.3255 | 1.3677 | 0.9504 |
| l6_w256 | specialist | confirmation | 0.3556 | 1.1923 | 1.1749 |
| l6_w256 | specialist | depth_ood_5_8 | 0.3382 | 1.2926 | 1.1279 |
| l6_w256 | specialist | long_history | 0.3958 | 1.1932 | 1.6295 |
| l6_w256 | specialist | heldout_combo | 0.3034 | 1.3264 | 0.9210 |
| l6_w256 | matched | confirmation | 0.3490 | 1.2866 | 1.1978 |
| l6_w256 | matched | depth_ood_5_8 | 0.3382 | 1.3206 | 1.1415 |
| l6_w256 | matched | long_history | 0.3268 | 1.5700 | 1.6474 |
| l6_w256 | matched | heldout_combo | 0.3177 | 1.3824 | 0.9460 |

## Causal interventions (specialist arm, base confirmation)

| Size | Intervention | Accuracy delta | Answer NLL delta | Overall NLL delta |
| --- | --- | ---: | ---: | ---: |
| l2_w64 | ablate | -0.0039 | +0.0088 | +0.0239 |
| l2_w64 | wrong_pointer | -0.0013 | +0.0003 | +0.0000 |
| l4_w128 | ablate | -0.0465 | +0.3377 | +0.1576 |
| l4_w128 | wrong_pointer | +0.0000 | +0.0003 | +0.0000 |
| l6_w256 | ablate | -0.0592 | +0.7959 | +0.0465 |
| l6_w256 | wrong_pointer | -0.0007 | -0.0005 | +0.0000 |

The wrong-pointer control replaces every queried declaration address with a different active declaration (mean changed-pointer fraction is 1.000 at every size).

## Renaming and whitespace invariance

| Size | Arm | Renaming agreement | Whitespace agreement |
| --- | --- | ---: | ---: |
| l2_w64 | plain | 0.2251 | 0.2109 |
| l2_w64 | specialist | 0.2294 | 0.2144 |
| l2_w64 | matched | 0.2208 | 0.2205 |
| l4_w128 | plain | 0.3117 | 0.2743 |
| l4_w128 | specialist | 0.3117 | 0.2648 |
| l4_w128 | matched | 0.3247 | 0.2587 |
| l6_w256 | plain | 0.5455 | 0.5078 |
| l6_w256 | specialist | 0.5541 | 0.5312 |
| l6_w256 | matched | 0.5411 | 0.5304 |

## Measured cost

Specialist training: 21.08 accelerator-s; preprocessing/cache: 17.36 wall-s and 9,900,062 bytes.
Pilot total: 405.79 accelerator-s on one NVIDIA GeForce RTX 3090; maximum allocated memory: 0.240 GiB. All token, time, and 16 GiB memory checks passed.

| Size | First-use savings | Amortized training savings | Online savings |
| --- | ---: | ---: | ---: |
| l2_w64 | -101.4% | -75.2% | -81.4% |
| l4_w128 | -89.2% | -52.0% | -61.0% |
| l6_w256 | -57.9% | -25.6% | -44.5% |

## Checkpoint hashes

| Size | Arm | Seed | SHA-256 | Specialist SHA-256 |
| --- | --- | ---: | --- | --- |
| L2 W64 | matched | 11 | `06afce10afc9bc0b7655e6d02e732aa38443be5efaf0a1e126f63e834105dac0` | `e0dcb68ba8c236533460ac2a8c2ac7f4319d4c4585cdd5ed4fc85ae26578ade0` |
| L2 W64 | matched | 22 | `174b022a334adff439368aa6634788e49624e89fa54662a6c286db91b916cad1` | `e0dcb68ba8c236533460ac2a8c2ac7f4319d4c4585cdd5ed4fc85ae26578ade0` |
| L2 W64 | matched | 33 | `cbdda1f762bb6ab16c3c40b3c54d392de6515a71f85e000b1b18c6f8ac898b4a` | `e0dcb68ba8c236533460ac2a8c2ac7f4319d4c4585cdd5ed4fc85ae26578ade0` |
| L2 W64 | plain | 11 | `c045643756c809eadb69675417b6ff398343057aa74b5cfd7fbc0b04f27d6ecf` | `e0dcb68ba8c236533460ac2a8c2ac7f4319d4c4585cdd5ed4fc85ae26578ade0` |
| L2 W64 | plain | 22 | `07b05fc2f1735382013aba5a4b0b776fc05423cfde376e4224b63d661b1109cb` | `e0dcb68ba8c236533460ac2a8c2ac7f4319d4c4585cdd5ed4fc85ae26578ade0` |
| L2 W64 | plain | 33 | `5e08b115c32b21b8ea2322fb22cae351b246f95c79406387c458784eec9ce56b` | `e0dcb68ba8c236533460ac2a8c2ac7f4319d4c4585cdd5ed4fc85ae26578ade0` |
| L2 W64 | specialist | 11 | `3f1843b3f306ddca0c9c1b06d8151b400cb383da4f466813751d7a988ecc149e` | `e0dcb68ba8c236533460ac2a8c2ac7f4319d4c4585cdd5ed4fc85ae26578ade0` |
| L2 W64 | specialist | 22 | `c9d19f1243d87bc3f070c010b85e113df22634ed8157323c5088e11bb39fd214` | `e0dcb68ba8c236533460ac2a8c2ac7f4319d4c4585cdd5ed4fc85ae26578ade0` |
| L2 W64 | specialist | 33 | `0fbef0695d3cd6a6ec93db8e0ee4973593b4e330ad1ec52e22ce78152acac141` | `e0dcb68ba8c236533460ac2a8c2ac7f4319d4c4585cdd5ed4fc85ae26578ade0` |
| L4 W128 | matched | 11 | `e73c6f92fbd3e9b75d5328d19a59d551e255abbb5b4f895e3a52922fb7c55d56` | `e0dcb68ba8c236533460ac2a8c2ac7f4319d4c4585cdd5ed4fc85ae26578ade0` |
| L4 W128 | matched | 22 | `e8dde02f2bee43691e545ecde9f823a4d3d5ec8c1cfac5a65ee83f5b7a55aff2` | `e0dcb68ba8c236533460ac2a8c2ac7f4319d4c4585cdd5ed4fc85ae26578ade0` |
| L4 W128 | matched | 33 | `cff8e0b4208477942f247ae767835bc6fe1a857c0b29949ded5429792de5880f` | `e0dcb68ba8c236533460ac2a8c2ac7f4319d4c4585cdd5ed4fc85ae26578ade0` |
| L4 W128 | plain | 11 | `de4bce1557ba7fc8c301ffbe72b4436778ae445b93ef3bd0c98e39ab2c44e18d` | `e0dcb68ba8c236533460ac2a8c2ac7f4319d4c4585cdd5ed4fc85ae26578ade0` |
| L4 W128 | plain | 22 | `039726ab103dab5e7d7ecf0d597aa9ecfe09165cd04ca83227a2c3897b6262a2` | `e0dcb68ba8c236533460ac2a8c2ac7f4319d4c4585cdd5ed4fc85ae26578ade0` |
| L4 W128 | plain | 33 | `90469a692243e67e23fedcd3a0907af81b86fa87df98a3662f66d89f12f9d4df` | `e0dcb68ba8c236533460ac2a8c2ac7f4319d4c4585cdd5ed4fc85ae26578ade0` |
| L4 W128 | specialist | 11 | `0719abef025d1859907d7fbe9309481ce54a960be52ba542c923330857b90286` | `e0dcb68ba8c236533460ac2a8c2ac7f4319d4c4585cdd5ed4fc85ae26578ade0` |
| L4 W128 | specialist | 22 | `e3385fcf7bf643c6e2facb96a77c4cfb8c957be0efadca3865c052f2373c1349` | `e0dcb68ba8c236533460ac2a8c2ac7f4319d4c4585cdd5ed4fc85ae26578ade0` |
| L4 W128 | specialist | 33 | `e4cae157b524f8e24f470934d103dc84da2e3954241c37d306109e2ebf64705a` | `e0dcb68ba8c236533460ac2a8c2ac7f4319d4c4585cdd5ed4fc85ae26578ade0` |
| L6 W256 | matched | 11 | `e14348d81cb79a8e2a4decc5de7f30175e122997e42d2e6ed1b300628e9ae695` | `e0dcb68ba8c236533460ac2a8c2ac7f4319d4c4585cdd5ed4fc85ae26578ade0` |
| L6 W256 | matched | 22 | `537104a38a8ccbf69be8aa2df314a66871b9cd191d6df4c110d00a1c888e5b1b` | `e0dcb68ba8c236533460ac2a8c2ac7f4319d4c4585cdd5ed4fc85ae26578ade0` |
| L6 W256 | matched | 33 | `874203ffc5c6170af1cff327866234933575f3d45389271f52589def9e9bbf00` | `e0dcb68ba8c236533460ac2a8c2ac7f4319d4c4585cdd5ed4fc85ae26578ade0` |
| L6 W256 | plain | 11 | `0c81b60f20fdb790dea46d76fc1146f36611d993b9ebdb76d8eeca3c246f0e7f` | `e0dcb68ba8c236533460ac2a8c2ac7f4319d4c4585cdd5ed4fc85ae26578ade0` |
| L6 W256 | plain | 22 | `f14a61a2157394d51e512084ef08bbe4d26e0371124249da9c8a2ce393cf47cf` | `e0dcb68ba8c236533460ac2a8c2ac7f4319d4c4585cdd5ed4fc85ae26578ade0` |
| L6 W256 | plain | 33 | `8e531568ea25b8c86743f0db3d993ff27c959cfbdac9f2a78ab5b2b8993812e9` | `e0dcb68ba8c236533460ac2a8c2ac7f4319d4c4585cdd5ed4fc85ae26578ade0` |
| L6 W256 | specialist | 11 | `58e6765f88095938e001f1145067976424fc6c0016f114849066f45fae4bd664` | `e0dcb68ba8c236533460ac2a8c2ac7f4319d4c4585cdd5ed4fc85ae26578ade0` |
| L6 W256 | specialist | 22 | `54825d1ffc5cb60a60fe909085c9e2b776be40c2b29d9dd2dadff90b48bb4b91` | `e0dcb68ba8c236533460ac2a8c2ac7f4319d4c4585cdd5ed4fc85ae26578ade0` |
| L6 W256 | specialist | 33 | `3456cc4b33d4b5bf00372aa9138dbff95bf6f5339dc1b40df4a7d182417b7cf4` | `e0dcb68ba8c236533460ac2a8c2ac7f4319d4c4585cdd5ed4fc85ae26578ade0` |

The accuracy margin is 0.01 and both NLL margins are 0.02 nat. Intervals are paired hierarchical bootstraps over documents and seeds. First-use cost includes specialist training and corpus preprocessing; amortized cost spreads specialist training across the nine specialist-backed runs. Online timing includes tokenization, programmed state, recurrent inference, reader, and backbone where applicable.
