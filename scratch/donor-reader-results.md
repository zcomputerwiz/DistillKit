# Donor-reader CPU preflight (2026-09-11)

All calculations here ran with CUDA hidden. No model forward, training step, or GPU
kernel was used. Machine-readable outputs are the sibling
`donor-reader-weights.json`, `donor-reader-representations.json`, and
`donor-reader-collapses-*.json` files.

## Donor value projection

- Shape: 2560 x 2560; Frobenius norm 39.1393; spectral norm 2.68875.
- Effective rank: 1799.65; stable rank: 212.07.
- Exact condition number: 43,488.7; robust p99/p01 condition estimate: 335.54.

On 512 assistant prediction positions from the untouched screen bundle, donor values
were much larger than trained C1 values (median norms 0.41319 versus 0.01843). Their
row-wise cosine was near zero (median -0.00635), while linear CKA was 0.90458. Thus the
two readers preserve a strongly similar sample-similarity geometry but express it in
substantially different bases and scales; a direct residual write would not be a fair
compatibility test.

| Held-out group | rows | C1 median norm | donor median norm |
| --- | ---: | ---: | ---: |
| content | 492 | 0.01811 | 0.41161 |
| layout | 20 | 0.01980 | 0.44569 |
| all 16 rows seen in C1 training | 150 | 0.02168 | 0.45326 |
| at least one row novel | 362 | 0.01633 | 0.40183 |

## Donor temporal filters

The convolution is overwhelmingly instantaneous by energy, but not by channel count:

| tap | energy | channels dominated |
| --- | ---: | ---: |
| t | 97.9880% | 78.81% |
| t-3 | 0.9129% | 7.23% |
| t-6 | 0.5840% | 7.01% |
| t-9 | 0.5152% | 6.95% |

The deterministic classification labels 63.13% of stream/channels instantaneous,
5.41% averaging, 10.22% differencing, and 21.24% otherwise history-sensitive. Filter
banks are not four copies: flattened cosine ranges from -0.216 to +0.506 (stream 1 and
3 are the most similar at +0.506).

## Cheap four-stream collapses

PCA on donor-conv outputs explains 91.05% of stream energy with C1 values and 87.94%
with donor values; in both cases it is dominated by donor stream 1. This says the bank
is close to rank one in energy, not that its leading direction is useful to the 4B.

Using frozen C1 value + donor conv, a four-scalar least-squares collapse fit to the
frozen C1 value+C1-conv mean achieved cosine 0.82765 and MSE 1.70e-8 with weights:

```text
[-0.0003896, 0.0107654, -0.0042293, 0.0180039]
```

The same proxy fit with donor value + donor conv had cosine only 0.01561 and collapsed
nearly to zero. This is only a C1-feature compatibility proxy. Content/layout NLL,
real-versus-shuffled controls, and the frozen rho sweep remain the decision metrics and
must wait for GPU availability.
