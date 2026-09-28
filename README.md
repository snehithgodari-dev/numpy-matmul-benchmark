# Matrix Multiplication 5 Ways: Pure Python vs NumPy vs BLAS vs JAX

Same 1000x1000 float64 matrix multiplication (2 x 10^9 floating-point operations),
five implementations, timed and verified for correctness. Every number below was
measured, not quoted.

## Methods

 #  Method                                               What happens 

 1  Pure Python nested loops                | ~10^9 interpreted inner-loop iterations |
 2  NumPy row-by-row (`A[i] @ B`)           | 1000 Python iterations, compiled inner product |
 3  NumPy vectorised (`A @ B`)              | one call, dispatches to BLAS `dgemm` |
 4  BLAS `dgemm` called directly            | same routine, no NumPy dispatch layer |
 5 JAX `jit`                                | XLA-compiled, compile once and reuse |

## Results (1 CPU core, OpenBLAS, Python 3.12, NumPy 2.4)

| Method | Runs | Mean time |

| Pure Python nested loops | 1 | 155.8 s |
| NumPy row-by-row | 10 | 257 ms |
| NumPy vectorised `A @ B` | 10 | 25.9 ms |
| BLAS `dgemm` direct | 10 | 25.5 ms |
| JAX `jit` (after warmup) | 10 | 27.5 ms |

1.Fastest vs slowest: about 6,100x in this run.
2.`A @ B` sustained about 77 GFLOPS on one core.
3.JAX first call (trace + compile + run): 41 ms.
4.Memory layout: summing a 4000x4000 array column-by-column (strided) was 1.7x slower than row-by-row (contiguous).

Caveat(warning) on the pure Python number:
it is timed once because it takes minutes, and
it varied a lot on this shared machine (72 s, 75 s and 156 s across three separate
runs). The NumPy-family numbers were stable (std under 10%). Treat the headline
ratio as "thousands of times", and re-run on your own machine for your own figure.

## What the results show

1.Removing the innermost Python loop (pure Python -> row-by-row) gives roughly 300-600x.
2.Removing the last Python loop (row-by-row -> `A @ B`) gives another ~10x, because BLAS
  tiles the matrices to fit CPU cache and uses SIMD.
3.`A @ B` and direct `dgemm` are equal within noise: NumPy's dispatch overhead is negligible at this size.
4.JAX `jit` does not beat NumPy for a single matmul: both call an optimised kernel.
  `jit` pays off when it fuses many ops, not for one BLAS-bound op.
5.BLAS thread scaling needs more than one core; the notebook prints scaling for however many cores you have.

## Repo structure

```
numpy-matmul-benchmark/
├── README.md
├── matmul_benchmark.py         # full benchmark as a script
├── matmul_explained.ipynb      # walkthrough with commentary
└── matmul_code_only.ipynb      # clean code, no explanation
```

## How to run

```bash
pip install numpy scipy threadpoolctl jax
python3 matmul_benchmark.py
```

Pure Python at n=1000 takes 1-3 minutes. For a quick pass, set `N = 500` at the top.
