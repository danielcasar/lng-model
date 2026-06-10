# Hormuz LNG Market Model

A multi-stage stochastic Stackelberg-EPEC model of the global liquefied
natural gas (LNG) market, applied to the 2026 Strait of Hormuz closure
scenario.

Built for the TU Wien course *Advanced Energy System Modeling* (370.100,). 
Adapts the linear-programming framework of
Zwickl-Bernhard and Neumann (2024) to a two-leader strategic equilibrium
under a Bayesian scenario tree.

---

## What the model does

- **Two strategic Stackelberg leaders**: USA and Qatar each choose monthly
  LNG export volumes to Europe and Asia, anticipating the regional
  followers' market-clearing response.
- **Two regional followers**: EU and Asia welfare-maximising market clearers
  with inter-temporal storage subject to the EU 90% Nov-1 mandate
  (Regulation 2017/1938).
- **Bayesian scenario tree**: closure status evolves as a two-state Markov
  chain whose transition rates are *unknown* to the agents — they form
  Beta-Bernoulli posterior beliefs from observation history, and the
  posterior means become the conditional branching probabilities of a
  59-node tree spanning Sep 2025 – Feb 2028.
- **Endogenous risk premium**: emerges naturally through the storage Euler
  condition under stochastic beliefs about closure persistence and
  recurrence — no exogenous "risk premium" parameter is imposed.

The leader's problem is a Mathematical Program with Complementarity
Constraints (MPCC), reformulated as a single-level Mixed-Integer Quadratic
Program via KKT conditions on the followers' problem and Fortuny-Amat–McCarl
Big-M linearisation of the complementarity products. The Equilibrium Problem
with Equilibrium Constraints (EPEC) across the two leaders is solved by
Gauss–Seidel diagonalization with damped best-response.

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
python 11_epec_2leader.py
```

Runs the full Gauss–Seidel diagonalization. Total wall time ~60–90 minutes
on a 16-thread workstation. Output includes per-iteration profits and a
realised-path equilibrium table at the end.

---

## File organisation

| File | Role |
|---|---|
| `lng_data.py` | Calibration data: break-even prices, transport costs, liquefaction capacities, region-access shares, and event definitions. Sourced from Zwickl-Neumann (2024) Table 6 and Appendix A. |
| `scenario_tree.py` | Builds the 59-node Bayesian scenario tree. Maintains posterior Beta-Bernoulli beliefs over the two-state Markov chain transition rates. |
| `11_epec_2leader.py` | **The model.** Two-leader stochastic Stackelberg EPEC with Gauss–Seidel diagonalization. |
| `ttf_history.csv` | Historical TTF spot prices for calibration / validation reference. |

---

## Model parameters (in `11_epec_2leader.py`)

| Parameter | Value | Source |
|---|---|---|
| EU demand block 1 | 28 bcm/mo at €80/MWh | Heating/petrochemical; sized for realistic winter drawdown (AGSI+) |
| EU demand block 2 | 10 bcm/mo at €50/MWh | Industrial coal-switch parity (EUA-inclusive) |
| EU demand block 3 | 7 bcm/mo at €25/MWh | Power-sector fuel switching |
| EU winter seasonality | 1.65 | Eurostat 2018–23 typical |
| EU storage cap | 100 bcm | GIE AGSI+ aggregate working gas |
| EU Nov-1 mandate | 90% | Reg. 2017/1938 target |
| Bayesian prior on closure rate | Beta(2, 40), mean 4.76%/mo | Post-2014 GPR-index regime; small effective sample for responsive updating |
| Bayesian prior on reopening rate | Beta(2, 12), mean 14.3%/mo | Historical chokepoint duration record |
| Damping for diagonalization | α = 0.4 | Cournot-type damping convention |

---

## Output interpretation

Each iteration prints leader expected profits and average realised-path
dispatch. After max_iter (or convergence), the final block shows:

- Per-leader expected profit over the entire scenario tree
- Realised-path equilibrium prices π_EU and π_Asia at each month
- Per-leader monthly dispatch q_USA_EU, q_USA_Asia, q_Qatar_EU, q_Qatar_Asia
- EU storage trajectory along the realised path
- Computing time (per solve, per iteration, and total)

The reported window is t = −5 (Sep 2025) to t = +12 (Feb 2027). Nodes
t = +13 to +24 are computed but not displayed — they exist only to push the
terminal storage condition far enough out that it does not contaminate the
reported window (standard horizon-extension fix for terminal-condition
artefacts in multi-stage stochastic optimisation).

---

## License

MIT — see `LICENSE` file.

## Author and citation

Daniel Caesar (TU Wien), 2026. If you use this code, please cite the
accompanying paper (in preparation) and reach out for the current draft.

### Acknowledged data source

The supply-side calibration (break-even prices, transport costs,
liquefaction capacities, and route-access shares) is taken from:

```
Zwickl-Bernhard, S. & Neumann, A. (2024). Modeling Europe's role in the
global LNG market 2040: Balancing decarbonization goals, energy security,
and geopolitical tensions. Energy, 301, 131612.
https://doi.org/10.1016/j.energy.2024.131612
```

The strategic-equilibrium formulation, the Bayesian scenario tree, and the
EPEC reformulation are original contributions of this model.
