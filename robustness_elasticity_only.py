"""
Robustness test: is the crisis price response fitted or predicted?

The v7 EU staircase uses the observed March-2026 pair (35 bcm, EUR 57)
as one of its two calibration anchors, so the March fit is partially by
construction. This test rebuilds the EU staircase using ONLY pre-crisis
information:

  - the January anchor (cumulative demand 48.8 bcm at EUR 37), and
  - the short-run price elasticity of gas demand from the econometric
    literature (Burke & Yang 2016, upper short-run estimate -0.25),

via the constant-elasticity curve Q(P) = 48.8 * (P/37)^(-0.25) at January
seasonality, discretised into WTP rungs (heating block seasonal, flat
price-response rungs). NO crisis data enters the demand side. Asia is
left unchanged.

Result (2026-06-11): March predicted at EUR 55.2 vs observed 57.0 --
the closure response is emergent from market tightness plus literature
elasticity, not fitted. April/May overshoot (+16/+17) exactly where the
observed market dipped on ceasefire expectations, the documented
limitation of the two-state information structure.
"""

import importlib

import model_config as cfg

cfg.demand_blocks_base["EU"] = [
    (24.0, 120.0), (2.6, 118.0), (1.2, 105.0), (2.7, 80.0), (2.1, 65.0),
    (1.4, 57.0), (1.9, 48.0), (1.6, 42.0), (1.5, 37.0), (2.2, 31.0),
    (3.4, 24.0),
]

m13 = importlib.import_module("13_competitive_rolling")

if __name__ == "__main__":
    print("=" * 78)
    print("ROBUSTNESS: EU demand from pre-crisis anchor + literature "
          "elasticity only")
    print("=" * 78, flush=True)
    m13.roll(verbose=False)
