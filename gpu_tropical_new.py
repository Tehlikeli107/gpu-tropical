"""
GPU Tropical Algebra: First GPU Library for Tropical Geometry
=============================================================
Tropical arithmetic: replace (*, +) with (+, min).
  Tropical addition:       a (+) b = min(a, b)
  Tropical multiplication: a (*) b = a + b

Tropical polynomial: f(x) = min(a0, a1+x, a2+2x, ..., ad+dx)
  = piecewise-linear convex lower envelope of d+1 lines

GPU batch computation -- NO existing GPU library (as of 2026).
CPU tools: SageMath, TropSing, Gfan -- all single-threaded CPU.
"""
