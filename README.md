# Hormuz LNG Market Model

A multi-stage stochastic Stackelberg-EPEC model of the global liquefied
natural gas (LNG) market, applied to the 2026 Strait of Hormuz closure
scenario.

Built for the TU Wien course *Advanced Energy System Modeling* (370.100,
Unit 3 — Game Theory). Adapts the linear-programming framework of
Zwickl-Bernhard and Neumann (2024) to a two-leader strategic equilibrium
under a Bayesian scenario tree.

---

## What the model does

- **Two strategic Stackelberg leaders**: USA and Qatar each choose monthly
  LNG export volumes to Europe and Asia, anticipating the regional
  followers' market-clearing response.
- **Two regional followers**: EU and Asia welfare-maximising market clearers
  with inter-temporal storage subject to the EU 80% Nov-1 mandate
  (Regulation 2017/1938) and a seasonal convenience yield on inventory.
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

### Run the main model (two-leader EPEC)

```bash
python 11_epec_2leader.py
```

Runs the full Gauss–Seidel diagonalization. Total wall time ~60–90 minutes
on a 16-thread workstation. Output includes per-iteration profits and a
realised-path equilibrium table at the end.

### Run the deterministic baseline (single-period, four leaders)

```bash
python 08_epec_diagonalization.py
```

The original deterministic 4-leader EPEC (USA + Australia + Russia + Qatar),
no Bayesian tree, no convenience yield. Used as a comparison anchor.

---

## File organisation

| File | Role |
|---|---|
| `lng_data.py` | Calibration data: break-even prices, transport costs, liquefaction capacities, region-access shares, and event definitions. Sourced from Zwickl-Neumann (2024) Table 6 and Appendix A. |
| `scenario_tree.py` | Builds the 59-node Bayesian scenario tree. Maintains posterior Beta-Bernoulli beliefs over the two-state Markov chain transition rates. |
| `11_epec_2leader.py` | **Main model.** Two-leader stochastic Stackelberg EPEC with Gauss–Seidel diagonalization. |
| `08_epec_diagonalization.py` | Deterministic four-leader baseline (no scenario tree, single-period). |
| `ttf_history.csv` | Historical TTF spot prices for calibration / validation reference. |

---

## Model parameters (in `11_epec_2leader.py`)

| Parameter | Value | Source |
|---|---|---|
| EU demand block 1 | 22 bcm/mo at €120/MWh | Heating/petrochemical fuel-switching parity |
| EU demand block 2 | 10 bcm/mo at €60/MWh | Industrial coal-switch parity (EUA-inclusive) |
| EU demand block 3 | 7 bcm/mo at €35/MWh | Power-sector fuel switching |
| EU winter seasonality | 1.65 | Eurostat 2018–23 typical |
| EU storage cap | 100 bcm | GIE AGSI+ aggregate working gas |
| EU Nov-1 mandate | 80% | Reg. 2017/1938 as amended 2022, flex provision |
| Convenience yield (Apr–Oct) | −€5/MWh | Schwartz (1997) / Pirrong (2011) |
| Bayesian prior on closure rate | Beta(5, 100), mean 4.76%/mo | Post-2014 GPR-index regime |
| Bayesian prior on reopening rate | Beta(2, 12), mean 14.3%/mo | Historical chokepoint duration record |
| Damping for diagonalization | α = 0.4 | Cournot-type damping convention |

---

## Output interpretation

Each iteration prints leader expected profits and average realised-path
dispatch. After max_iter (or convergence), the final block shows:

- Per-leader expected profit over the entire scenario tree
- Realised-path equilibrium prices π_EU and π_Asia at each month
- Per-leader monthly dispatch q_USA_EU, q_USA_Asia, q_Qatar_EU, q_Qatar_Asia

The reported window is t = −5 (Sep 2025) to t = +12 (Feb 2027). Nodes
t = +13 to +24 are computed but not displayed — they exist only to push the
terminal storage condition far enough out that it does not contaminate the
reported window (standard horizon-extension fix for terminal-condition
artefacts in multi-stage stochastic optimisation).

---

## License

MIT — see `LICENSE` file.

## Citation

If you use this code for academic work, please cite the underlying paper
(in preparation) and the calibration source:

```
Zwickl-Bernhard, S. & Neumann, A. (2024). Modeling Europe's role in the
global LNG market 2040: Balancing decarbonization goals, energy security,
and geopolitical tensions. Energy, 301, 131612.
https://doi.org/10.1016/j.energy.2024.131612
```
