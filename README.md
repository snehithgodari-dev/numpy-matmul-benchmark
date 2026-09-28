# Matrix Multiplication 4 Ways: Pure Python vs NumPy vs BLAS

Same 1000x1000 float64 matrix multiplication (2 x 10^9 floating-point operations),
four implementations, timed and verified for correctness. A JAX `jit` comparison is
included as a side experiment. Every number below was measured, not quoted.

## Methods

| # | Method                      | What happens |

| 1 | Pure Python nested loops | ~10^9 interpreted inner-loop iterations |
| 2 | NumPy row-by-row (`A[i] @ B`) | 1000 Python iterations, compiled inner product |
| 3 | NumPy vectorised (`A @ B`) | one call, dispatches to BLAS `dgemm` |
| 4 | BLAS `dgemm` called directly | same routine, no NumPy dispatch layer |

## Results (1 CPU core, OpenBLAS, Python 3.12, NumPy 2.4)

From `matmul_explained.ipynb`:

| Method | Runs | Mean time |

| Pure Python nested loops | 1 | 155.8 s |
| NumPy row-by-row | 10 | 257 ms |
| NumPy vectorised `A @ B` | 10 | 25.9 ms |
| BLAS `dgemm` direct | 10 | 25.5 ms |

1.Fastest vs slowest: about 6,100x in this run (see variance below).
2.`A @ B` sustained about 77 GFLOPS on one core.
3.Memory layout: summing a 4000x4000 array column-by-column (strided) was 1.7x slower than row-by-row (contiguous).

## Side experiment: JAX `jit`

| Method | Runs | Mean time |

| JAX `jit` (after warmup) | 10 | 27.5 ms |

For a single matmul, JAX `jit` was no faster than NumPy (27.5 ms vs 25.9 ms), because
both call an optimised kernel. The first call (trace + compile + run) took 41 ms.
`jit` pays off when it fuses many operations, not for one BLAS-bound op.

## Run-to-run variance

Pure Python is timed once because it takes minutes, and it is the noisy number.
Across five separate full runs it measured **72 s, 74 s, 75 s, 76 s and 156 s**, so the
fastest-vs-slowest ratio ranged from about **2,700x to 6,100x**. The NumPy-family
timings stayed within about 15% across runs.

For that reason the three notebooks in this repo show slightly different numbers:

| Notebook | Pure Python | Fastest vs slowest | Column vs row sums |

| `matmul_explained.ipynb` | 155.8 s | 6,122x | 1.7x |
| `matmul_code_only.ipynb` | 73.7 s | 3,015x | 2.0x |
| `matmul_benchmark.ipynb` | 76.5 s | 2,769x | not included |

Read the headline as "thousands of times faster", not as an exact figure. Re-run on
your own machine for your own numbers.

## What the results show

1. Row-by-row removes two of the three loops (the `j` and `k` loops), worth roughly 300-600x.
2. `A @ B` removes the last loop, worth another ~10x, because BLAS tiles the matrices to fit CPU cache and uses SIMD.
3.`A @ B` and direct `dgemm` are equal within noise: NumPy's dispatch overhead is negligible at this size.
4.A transpose (`.T`) costs nothing by itself, since it only flips strides, but the result is no longer C-contiguous, so the next operation can be slower.
5.BLAS thread scaling needs more than one core; the notebook prints scaling for however many cores you have.

## Repo structure

```
numpy-matmul-benchmark/
├── README.md
├── matmul_benchmark.py         # full benchmark as a script
├── matmul_benchmark.ipynb      # the script as a single notebook
├── matmul_explained.ipynb      # walkthrough with commentary
└── matmul_code_only.ipynb      # clean code, no explanation
```



JAX is only needed for the side experiment. Pure Python at n=1000 takes 1-3 minutes;
for a quick pass, set `N = 500` at the top.
