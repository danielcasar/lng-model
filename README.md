# Hormuz LNG Market Model

A rolling-horizon **competitive (welfare-maximising LP)** model of the global
liquefied natural gas (LNG) market under **two-sided Strait-of-Hormuz closure
uncertainty**. Market prices form endogenously as the **duals of the nodal
market-balance constraints**; a Bayesian scenario tree carries the agents'
beliefs about reopening together with a persistent escalation (deeper-
disruption) tail. A two-leader strategic Stackelberg-EPEC is retained as an
optional *comparison* experiment (see below).

Built for the TU Wien course *Advanced Energy System Modeling* (370.100).
Adapts the linear-programming supply-side framework of Zwickl-Bernhard and
Neumann (2024) to a multi-stage stochastic market under the 2026 closure.

---

## What the model does

The market core is **perfectly competitive**: at ~98% liquefaction
utilisation there is no slack to withhold, so price formation is driven by
market *tightness* meeting a demand-response ladder, not by strategic
behaviour (Fulwood 2024, OIES NG 195). Each monthly re-solve maximises
expected welfare over a Bayesian scenario (sub)tree:

```
max  E[ consumer surplus - supply cost - storage holding cost ]
```

subject to capacities, contract floors, storage dynamics, deliverability
limits and the EU Nov-1 storage mandate. It is a pure LP — no binaries, no
Big-M, no equilibrium selection — and one monthly re-solve takes well under a
second.

- **Competitive market clearing, prices = duals.** The price in each region
  and scenario node is the shadow price of its market-balance constraint.
- **Demand-response ladder.** A WTP staircase per region (Fulwood 2024):
  coal/oil switching, efficiency, industrial closures, and finally rationing.
  The crisis moves the marginal block up the ladder.
- **Storage with a precautionary refill.** An operational floor (30% of
  working gas) plus inter-temporal arbitrage; the *refilling* of storage
  through the crisis emerges from the escalation tail (below), matching the
  observed EU behaviour (stocks rose 30→46 bcm Mar–Jun 2026 despite the
  closure) rather than being imposed.
- **Two-sided closure uncertainty:**
  - *Reopening beliefs* — closure status is a two-state Markov chain whose
    transition rates are **unknown** to the agents; they hold Beta-Bernoulli
    posteriors updated from observation history, and the posterior means are
    the conditional branching probabilities.
  - *Persistent escalation tail* — an **absorbing** deeper-disruption state
    reachable from any closed node (a second chokepoint / wider conflict /
    damaged trains): a counterfactual the agents price in but which never
    realises on the observed path. Its persistence is what makes precautionary
    storage refilling economic.
  - *Realised crisis-deepening (reroute) derate* — a small duration-dependent
    loss of seaborne LNG along the realised closed path (shipping/rerouting
    fleet tie-up, Fulwood 2026; realised re-escalation), so the realised price
    profile rises through the closure as observed.
- **Calibrated** to observed monthly TTF/JKM prices and EU storage:
  realised-path price RMSE ≈ 3.9 EUR/MWh, storage RMSE ≈ 6 bcm.

### Strategic comparison (archived): the EPEC

A two-leader strategic Stackelberg-EPEC variant (USA + a Gulf composite as
price-makers, each leader an MPCC solved by Gauss–Seidel diagonalization) was
explored to test whether market power matters under the closure. It is **kept
locally in `code/archive/` and is not part of the tracked repo** — the
competitive LP is the model. The archived scripts import the shared structure
from `market.py`.

---

## Requirements

- Python 3.10+
- Pyomo 6.7+ (open-source optimisation modelling library)
- SciPy 1.10+ (for the Beta-Bernoulli prior)
- **Gurobi 11+ with a valid license** (free for academic use:
  https://www.gurobi.com/academia/)

Install Python dependencies:

```bash
pip install -r requirements.txt
```

Install Gurobi separately from the link above and set up the academic
license per their instructions.

---

## Quick start

```bash
python 13_competitive_rolling.py
```

Runs the full rolling-horizon calibration (t = −5 … +12, each re-solve
planning to t = +24) in a couple of seconds. Output: a calibration report
(model vs observed TTF/JKM with RMSE), an EU storage table (model vs observed,
with provenance flags), and three CSVs under `results/`
(`competitive_trajectory.csv`, `competitive_calibration.csv`,
`competitive_storage.csv`).

---

## File organisation

| File | Role |
|---|---|
| `13_competitive_rolling.py` | **THE model.** Rolling-horizon competitive welfare-LP per monthly belief subtree; prices = duals of the nodal market balance. Writes the calibration + storage reports and `results/` CSVs. |
| `model_config.py` | **Single configuration source.** Every tunable value (time horizon, priors, escalation/persistence, reroute derate, leaders, fringe, demand staircases, seasonality, storage, observed-storage series, Big-Ms, rolling settings) with inline source citations. All scripts import from here. |
| `parameters.csv` | **Single reference sheet of ALL configuration values and data inputs**, each with unit, type (Data / Derived / Calibrated / Assumption / Numerical) and explicit source. Start here for any sanity check. |
| `market.py` | **Shared market structure**: regions, fringe suppliers, delivered costs, liquefaction capacities, the closure/escalation/reroute capacity derates, the demand-response staircase, seasonality, and `make_ctx`. Imported by the LP. |
| `scenario_tree.py` | Builds the multi-stage **branching** Bayesian scenario tree (3-way from closed nodes: reopen/stay-closed/escalate; 2-way from open; escalated absorbing), full branching for `BRANCH_DEPTH` months then a modal tail. |
| `lng_data.py` | Supply-side data: break-even prices, transport costs, liquefaction capacities (2026 operational), event definitions. Zwickl-Bernhard & Neumann (2024) Table 6 / Appendix A. |
| `archive/` (untracked) | Local-only: the strategic-EPEC comparison (`11_epec_2leader.py`, `12_rolling_epec.py`) and legacy scripts. Gitignored. |
| `calibration_targets.csv` | Observed monthly TTF / JKM price targets (Sep 2025 – Jun 2026). |
| `eu_demand_monthly.csv` | Observed EU monthly gas demand 2019 – May 2026 (seasonality + demand-curve anchoring). |
| `ttf_history.csv` | Historical TTF spot prices for reference. |
| `results/` | Model outputs: realised-path trajectory, calibration table, and storage-vs-observed comparison (CSV). |

---

## Model parameters

All configuration lives in **`model_config.py`** (tunable values, with inline
citations) and **`lng_data.py`** (supply-side data). The companion sheet
**`parameters.csv`** documents every value row by row with unit, explicit
source, and a type flag distinguishing cited data from calibrated and stylised
values. When changing a value, edit `model_config.py` and update the
corresponding row in `parameters.csv`.

A note on calibration discipline: the *volumes* and *prices* (capacities,
delivered costs, WTP ladder) are held at their sourced values; the *scenario*
levers (Bayesian priors, escalation rate/loss/persistence, reroute derate) are
the free calibration knobs. The pre-crisis JKM−TTF gap is the US net-back
differential implied by the sourced delivered costs and is left untouched.

---

## Output interpretation

`13_competitive_rolling.py` prints, for the realised path t = −5 (Sep 2025) …
+12 (Feb 2027):

- **Calibration report** — model vs observed TTF (EU) and JKM (Asia) per
  month, with the overall price RMSE.
- **EU storage table** — model vs observed end-of-month stocks (bcm), with
  per-month provenance flags (firm / estimate / approx / projection) and the
  storage RMSE.
- **`results/` CSVs** — full trajectory (prices, leader dispatch, storage with
  `S_EU_obs`), the calibration table, and the dedicated storage comparison.

Nodes t = +13 … +24 are computed but not displayed — they exist only to push
the terminal storage condition far enough out that it does not contaminate the
reported window (standard horizon-extension fix for terminal-condition
artefacts in multi-stage stochastic optimisation).

---

## License

MIT — see `LICENSE` file.

## Author and citation

Daniel Caesar (TU Wien), 2026. If you use this code, please cite the
accompanying paper (in preparation) and reach out for the current draft.

### Acknowledged data source

The supply-side calibration (break-even prices, transport costs, liquefaction
capacities, and route-access shares) is taken from:

```
Zwickl-Bernhard, S. & Neumann, A. (2024). Modeling Europe's role in the
global LNG market 2040: Balancing decarbonization goals, energy security,
and geopolitical tensions. Energy, 301, 131612.
https://doi.org/10.1016/j.energy.2024.131612
```

The competitive multi-stage stochastic market formulation, the Bayesian
scenario tree with a persistent escalation tail, and the EPEC comparison are
original contributions of this model.
