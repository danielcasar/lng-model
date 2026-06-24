"""
Step 13: COMPETITIVE market core -- rolling welfare-maximisation LP.

THE restructured market model (see paper, model-abstraction discussion).
Price formation in the short-run LNG market is driven by market TIGHTNESS
meeting a demand-response ladder, not by strategic withholding (Fulwood
2024, OIES NG 195): at ~98% liquefaction utilisation there is no slack to
withhold. Accordingly the market core is PERFECTLY COMPETITIVE:

  max  E[ consumer surplus - supply cost - storage holding cost ]

over the Bayesian scenario tree, subject to capacities, contract floors,
storage dynamics, deliverability limits and the Nov-1 mandates. All agents
(including USA and Gulf) are price-takers; market prices are the DUALS of
the nodal market-balance constraints. The model is a pure LP: no binaries,
no Big-M, no equilibrium selection, no convergence question -- one monthly
re-solve takes well under a second.

Rolling horizon: each month rebuilds the belief subtree (conjugate Bayes +
the duration-dependent escalation tail), re-solves, and implements only the
root decisions. Shared market structure (fringe, costs, capacities, demand
blocks, the closure/escalation/reroute derates, make_ctx) lives in market.py.

A two-leader strategic Stackelberg-EPEC variant was explored as a comparison
(does market power matter under the closure?); it is archived outside the repo
(code/archive/) and is not part of the tracked model.
"""

import csv
import os
import time

import pyomo.environ as pyo

from scenario_tree import (
    build_tree_from, realized_status, calendar_month, T_LAST,
    ALPHA_C_PRIOR, BETA_C_PRIOR, ALPHA_R_PRIOR, BETA_R_PRIOR,
)
from model_config import (
    ROLL_START, ROLL_END,
    HOLDING_COST, storage, STORAGE_FLOOR_FRAC,
    EU_NOV_TARGET_FRAC, EU_NOV_TARGET_FRAC_2026,
    NOV_2026_T, STORAGE_TARGETS_EU, STORAGE_OBS_EU,
    EU_MAX_INJECT_BCM, EU_MAX_WITHDRAW_BCM,
    LEADERS, LEADER_REGIONS, CONTRACT_FLOOR,
)

# Shared market structure (fringe, costs, capacities, demand blocks, derates).
import market as mkt
REGIONS          = mkt.REGIONS
FRINGE_BY_REGION = mkt.FRINGE_BY_REGION
BLOCKS_BY_REGION = mkt.BLOCKS_BY_REGION

MONTH_NAMES = ["", "Jan", "Feb", "Mar", "Apr", "May", "Jun",
               "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]

# Show model prices in the calibration report/CSV through this month even where
# no observation exists yet (t=+10 = Dec 2026). Observations currently run to
# t=+4 (Jun 2026); t=+5..+10 are model forecast (observed columns left blank).
REPORT_PRICE_END = +10


def build_welfare_lp(ctx):
    """Welfare-maximisation LP over one scenario (sub)tree."""
    NODES        = ctx["NODES"]
    NODE_IDS     = ctx["NODE_IDS"]
    NOV_NODES    = ctx["NOV_NODES"]
    TERMINAL_IDS = ctx["TERMINAL_IDS"]
    S_INIT       = ctx["S_INIT"]

    m = pyo.ConcreteModel("competitive_market")
    m.R  = pyo.Set(initialize=list(REGIONS))
    m.N  = pyo.Set(initialize=NODE_IDS)
    m.RS = pyo.Set(initialize=[(r, s) for r in REGIONS
                               for s in FRINGE_BY_REGION[r]], dimen=2)
    m.RK = pyo.Set(initialize=[(r, k) for r in REGIONS
                               for k in BLOCKS_BY_REGION[r]], dimen=2)
    m.LR = pyo.Set(initialize=[(L, r) for L in LEADERS
                               for r in LEADER_REGIONS[L]], dimen=2)

    m.fringe_supply = pyo.Var(m.RS, m.N, domain=pyo.NonNegativeReals)
    m.leader_supply = pyo.Var(m.LR, m.N, domain=pyo.NonNegativeReals)
    m.demand_served = pyo.Var(m.RK, m.N, domain=pyo.NonNegativeReals)
    m.storage_level = pyo.Var(m.R,  m.N, domain=pyo.NonNegativeReals)
    m.storage_flow  = pyo.Var(m.R,  m.N, domain=pyo.Reals)

    # Market balance per (region, node): demand + injection <= total supply.
    # Its dual, divided by the node probability, is the market price.
    def _balance(mdl, region, nid):
        return (sum(mdl.demand_served[region, k, nid] for k in BLOCKS_BY_REGION[region])
                + mdl.storage_flow[region, nid]
                <= sum(mdl.fringe_supply[region, s, nid] for s in FRINGE_BY_REGION[region])
                   + sum(mdl.leader_supply[L, region, nid] for L in LEADERS
                         if region in LEADER_REGIONS[L]))
    m.balance = pyo.Constraint(m.R, m.N, rule=_balance)

    m.fringe_cap = pyo.Constraint(m.RS, m.N,
        rule=lambda mdl, region, s, nid:
            mdl.fringe_supply[region, s, nid]
            <= mkt.fringe_capacity(region, s, NODES[nid]))
    m.block_cap = pyo.Constraint(m.RK, m.N,
        rule=lambda mdl, region, k, nid:
            mdl.demand_served[region, k, nid]
            <= mkt.block_size(region, k, NODES[nid].t))

    # Leader capacity (path-aware: closure blocking + restart ramp) and
    # contract floor (LT-contract share that must be dispatched).
    def _leader_cap(mdl, L, nid):
        return (sum(mdl.leader_supply[L, region, nid] for region in LEADER_REGIONS[L])
                <= mkt.leader_cap_at_node(L, NODES[nid]))
    m.leader_cap = pyo.Constraint(pyo.Set(initialize=LEADERS), m.N, rule=_leader_cap)

    def _leader_floor(mdl, L, nid):
        return (sum(mdl.leader_supply[L, region, nid] for region in LEADER_REGIONS[L])
                >= CONTRACT_FLOOR[L] * mkt.leader_cap_at_node(L, NODES[nid]))
    m.leader_floor = pyo.Constraint(pyo.Set(initialize=LEADERS), m.N, rule=_leader_floor)

    def _storage_balance(mdl, region, nid):
        node = NODES[nid]
        prev = S_INIT[region] if node.parent_id == "" \
               else mdl.storage_level[region, node.parent_id]
        return mdl.storage_level[region, nid] == prev + mdl.storage_flow[region, nid]
    m.storage_balance = pyo.Constraint(m.R, m.N, rule=_storage_balance)

    m.storage_cap = pyo.Constraint(m.R, m.N,
        rule=lambda mdl, region, nid:
            mdl.storage_level[region, nid] <= storage[region]["S_max"])

    # Minimum operational storage floor (precautionary cushion): stock may
    # not be drawn below STORAGE_FLOOR_FRAC of working capacity in any node.
    m.storage_floor = pyo.Constraint(m.R, m.N,
        rule=lambda mdl, region, nid:
            mdl.storage_level[region, nid]
            >= STORAGE_FLOOR_FRAC * storage[region]["S_max"])

    def _inject(mdl, region, nid):
        if region != "EU": return pyo.Constraint.Skip
        return mdl.storage_flow[region, nid] <= EU_MAX_INJECT_BCM
    m.inject_limit = pyo.Constraint(m.R, m.N, rule=_inject)

    def _withdraw(mdl, region, nid):
        if region != "EU": return pyo.Constraint.Skip
        return mdl.storage_flow[region, nid] >= -EU_MAX_WITHDRAW_BCM
    m.withdraw_limit = pyo.Constraint(m.R, m.N, rule=_withdraw)

    def _terminal(mdl, region, nid):
        if nid not in TERMINAL_IDS: return pyo.Constraint.Skip
        return mdl.storage_level[region, nid] == storage[region]["S_term"]
    m.terminal_storage = pyo.Constraint(m.R, m.N, rule=_terminal)

    def _nov(mdl, nid):
        if nid not in NOV_NODES: return pyo.Constraint.Skip
        frac = (EU_NOV_TARGET_FRAC_2026 if NODES[nid].t == NOV_2026_T
                else EU_NOV_TARGET_FRAC)
        return mdl.storage_level["EU", nid] >= frac * storage["EU"]["S_max"]
    m.nov_mandate = pyo.Constraint(m.N, rule=_nov)

    # Expected welfare: WTP of served demand minus supply costs minus
    # storage holding costs, probability-weighted over the tree.
    m.obj = pyo.Objective(
        sense=pyo.maximize,
        expr=sum(NODES[nid].cum_prob * (
                 sum(mkt.block_wtp(r, k) * m.demand_served[r, k, nid]
                     for (r, k) in m.RK)
                 - sum(mkt.fringe_cost(r, s) * m.fringe_supply[r, s, nid]
                       for (r, s) in m.RS)
                 - sum(mkt.leader_cost[L][r] * m.leader_supply[L, r, nid]
                       for (L, r) in m.LR)
                 - HOLDING_COST * sum(m.storage_level[r, nid] for r in REGIONS))
                 for nid in m.N))

    m.dual = pyo.Suffix(direction=pyo.Suffix.IMPORT)
    return m


def solve_competitive(ctx):
    """Solve the welfare LP; return (prices, leader_q, stocks, welfare)."""
    NODES, NODE_IDS = ctx["NODES"], ctx["NODE_IDS"]
    m = build_welfare_lp(ctx)
    solver = pyo.SolverFactory("gurobi")
    solver.options["OutputFlag"] = 0
    solver.solve(m, tee=False)

    prices = {}
    for region in REGIONS:
        prices[region] = {}
        for nid in NODE_IDS:
            prob = max(NODES[nid].cum_prob, 1e-12)
            prices[region][nid] = abs(m.dual[m.balance[region, nid]]) / prob
    leader_q = {L: {region: {nid: pyo.value(m.leader_supply[L, region, nid])
                             for nid in NODE_IDS}
                    for region in LEADER_REGIONS[L]} for L in LEADERS}
    stocks = {region: {nid: pyo.value(m.storage_level[region, nid])
                       for nid in NODE_IDS} for region in REGIONS}
    return prices, leader_q, stocks, pyo.value(m.obj)


def roll(verbose=True):
    t_script = time.time()
    counts  = (ALPHA_C_PRIOR, BETA_C_PRIOR, ALPHA_R_PRIOR, BETA_R_PRIOR)
    s_state = {region: storage[region]["S_init"] for region in REGIONS}
    trajectory = []
    closed_run = 0   # true consecutive closed months elapsed up to t0

    for t0 in range(ROLL_START, ROLL_END + 1):
        open0 = realized_status(t0)
        closed_run = 0 if open0 else closed_run + 1
        t_roll = time.time()
        # Carry the TRUE elapsed closure into the tree so the escalation hazard
        # and reroute derate see real duration (the Bayesian counts already
        # persist across re-solves; the duration must too).
        nodes, realized_ids = build_tree_from(t0, open0, *counts,
                                              closed_at_start=closed_run)
        ctx = mkt.make_ctx(nodes, realized_ids, s_state)

        prices, leader_q, stocks, welfare = solve_competitive(ctx)

        root = realized_ids[0]
        rec = {
            "t": t0, "month": MONTH_NAMES[calendar_month(t0)], "open": open0,
            "p_EU": prices["EU"][root], "p_AS": prices["Asia"][root],
            "qUSA_EU": leader_q["USA"]["EU"][root],
            "qUSA_AS": leader_q["USA"]["Asia"][root],
            "qGLF_EU": leader_q["Gulf"]["EU"][root],
            "qGLF_AS": leader_q["Gulf"]["Asia"][root],
            "S_EU": stocks["EU"][root], "S_AS": stocks["Asia"][root],
            "secs": time.time() - t_roll,
        }
        trajectory.append(rec)
        if verbose:
            print(f"ROLL t={t0:+d} ({rec['month']}) "
                  f"{'OPEN' if open0 else 'CLOSED':>6}  "
                  f"p_EU={rec['p_EU']:6.1f}  p_AS={rec['p_AS']:6.1f}  "
                  f"S_EU={rec['S_EU']:5.1f}  [{rec['secs']:.1f}s]", flush=True)

        s_state = {region: max(0.0, stocks[region][root]) for region in REGIONS}
        if t0 < ROLL_END:
            # Carry beliefs by conjugate Bayesian update on the ACTUALLY realized
            # transition t0 -> t0+1 (the tree no longer has a realized spine to
            # read counts off; the root is the only realized node).
            aC, bC, aR, bR = counts
            nxt_open = realized_status(t0 + 1)
            if open0:                       # observed OPEN ->
                if nxt_open: bC += 1.0      #   stayed open
                else:        aC += 1.0      #   closed
            else:                           # observed CLOSED ->
                if nxt_open: aR += 1.0      #   reopened
                else:        bR += 1.0      #   stayed closed
            counts = (aC, bC, aR, bR)

    report(trajectory, time.time() - t_script)
    return trajectory


def report(trajectory, total_secs):
    TARGETS = {  # t: (TTF_eur, JKM_eur) -- see calibration_targets.csv
        -5: (33.0, 36.0), -4: (34.0, 37.0), -3: (36.0, 38.0),
        -2: (38.0, 39.0), -1: (37.0, 38.0),  0: (37.0, 38.0),
         1: (57.0, 63.0),  2: (42.0, 55.5),  3: (46.0, 57.0),
         4: (49.6, 59.5),
    }
    print("\n" + "=" * 70)
    print("CALIBRATION REPORT (competitive rolling): model vs observed")
    print("=" * 70)
    print(f"{'t':>3} {'mo':>6}  {'EU_mod':>7} {'EU_obs':>7} {'dEU':>6}   "
          f"{'AS_mod':>7} {'AS_obs':>7} {'dAS':>6}")
    sq_err, n_obs = 0.0, 0
    for rec in trajectory:
        if rec["t"] > REPORT_PRICE_END:
            continue
        if rec["t"] in TARGETS:
            eu_obs, as_obs = TARGETS[rec["t"]]
            print(f"{rec['t']:>+3d} {rec['month']:>6}  "
                  f"{rec['p_EU']:>7.1f} {eu_obs:>7.1f} {rec['p_EU']-eu_obs:>+6.1f}   "
                  f"{rec['p_AS']:>7.1f} {as_obs:>7.1f} {rec['p_AS']-as_obs:>+6.1f}")
            sq_err += (rec["p_EU"] - eu_obs) ** 2 + (rec["p_AS"] - as_obs) ** 2
            n_obs  += 2
        else:   # not yet observed -- show model forecast only
            print(f"{rec['t']:>+3d} {rec['month']:>6}  "
                  f"{rec['p_EU']:>7.1f} {'--':>7} {'--':>6}   "
                  f"{rec['p_AS']:>7.1f} {'--':>7} {'--':>6}")
    rmse = (sq_err / max(n_obs, 1)) ** 0.5
    print("-" * 70)
    print(f"RMSE over {n_obs} observed values (through Jun 2026): {rmse:.2f} EUR/MWh")

    # EU storage: model realized-path trajectory vs observed (GIE AGSI+ /
    # Fulwood 2026). RMSE over the months for which an observation exists.
    print("\n" + "=" * 70)
    print("EU STORAGE: model vs observed (bcm, working-gas)")
    print("=" * 70)
    print(f"{'t':>3} {'mo':>6}  {'S_model':>8} {'S_obs':>7} {'d':>6}")
    s_sq, s_n = 0.0, 0
    for rec in trajectory:
        s_obs = STORAGE_OBS_EU.get(rec["t"])
        if s_obs is None:
            print(f"{rec['t']:>+3d} {rec['month']:>6}  {rec['S_EU']:>8.1f} "
                  f"{'--':>7} {'--':>6}")
        else:
            print(f"{rec['t']:>+3d} {rec['month']:>6}  {rec['S_EU']:>8.1f} "
                  f"{s_obs:>7.1f} {rec['S_EU']-s_obs:>+6.1f}")
            s_sq += (rec["S_EU"] - s_obs) ** 2
            s_n  += 1
    s_rmse = (s_sq / max(s_n, 1)) ** 0.5
    print("-" * 70)
    print(f"Storage RMSE over {s_n} observed months: {s_rmse:.2f} bcm")
    print(f"Total computing time: {total_secs:.1f} s "
          f"({len(trajectory)} monthly re-solves)", flush=True)

    save_results(trajectory, TARGETS, rmse, s_rmse)


def save_results(trajectory, targets, rmse, s_rmse=None):
    """Persist the rolled trajectory + calibration to results/ as CSV."""
    out_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
    os.makedirs(out_dir, exist_ok=True)

    # 1. Full realized-path trajectory (prices, leader dispatch, storage).
    #    S_EU_obs = observed end-of-month EU storage where known (else blank).
    traj_path = os.path.join(out_dir, "competitive_trajectory.csv")
    with open(traj_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["t", "month", "status", "p_EU", "p_Asia",
                    "qUSA_EU", "qUSA_Asia", "qGulf_EU", "qGulf_Asia",
                    "S_EU", "S_EU_obs", "S_Asia"])
        for r in trajectory:
            s_obs = STORAGE_OBS_EU.get(r["t"])
            w.writerow([r["t"], r["month"], "OPEN" if r["open"] else "CLOSED",
                        round(r["p_EU"], 2), round(r["p_AS"], 2),
                        round(r["qUSA_EU"], 2), round(r["qUSA_AS"], 2),
                        round(r["qGLF_EU"], 2), round(r["qGLF_AS"], 2),
                        round(r["S_EU"], 1),
                        "" if s_obs is None else round(s_obs, 1),
                        round(r["S_AS"], 1)])

    # 1b. Dedicated EU storage comparison (model vs observed, observed months).
    stor_path = os.path.join(out_dir, "competitive_storage.csv")
    with open(stor_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["t", "month", "S_EU_model", "S_EU_obs", "dS_EU", "source"])
        for r in trajectory:
            s_obs = STORAGE_OBS_EU.get(r["t"])
            if s_obs is None:
                continue
            if r["t"] == +1:
                src = "Fulwood2026/AGSI+ (firm)"
            elif r["t"] == -2:
                src = "AGSI+/CEEnergyNews (firm)"
            elif r["t"] == +8:
                src = "Fulwood2026 proj."
            elif r["t"] in (-1, 0):
                src = "AGSI+ interp. (est.)"
            else:
                src = "GIE AGSI+ (approx.)"
            w.writerow([r["t"], r["month"], round(r["S_EU"], 1), round(s_obs, 1),
                        round(r["S_EU"] - s_obs, 1), src])
        if s_rmse is not None:
            w.writerow([])
            w.writerow(["storage_RMSE_bcm", round(s_rmse, 2)])

    # 2. Calibration table: model vs observed over the target window.
    cal_path = os.path.join(out_dir, "competitive_calibration.csv")
    with open(cal_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["t", "month", "EU_model", "EU_obs", "dEU",
                    "Asia_model", "Asia_obs", "dAsia"])
        for r in trajectory:
            if r["t"] > REPORT_PRICE_END:
                continue
            if r["t"] in targets:
                eu_o, as_o = targets[r["t"]]
                w.writerow([r["t"], r["month"], round(r["p_EU"], 1), eu_o,
                            round(r["p_EU"] - eu_o, 1), round(r["p_AS"], 1), as_o,
                            round(r["p_AS"] - as_o, 1)])
            else:   # not yet observed -- model forecast only
                w.writerow([r["t"], r["month"], round(r["p_EU"], 1), "", "",
                            round(r["p_AS"], 1), "", ""])
        w.writerow([])
        w.writerow(["RMSE_EUR_per_MWh (through Jun 2026)", round(rmse, 2)])
    print(f"Results written to {os.path.relpath(out_dir)}/ "
          f"(competitive_trajectory.csv, competitive_calibration.csv, "
          f"competitive_storage.csv)", flush=True)


if __name__ == "__main__":
    print("=" * 78)
    print("COMPETITIVE rolling-horizon market core (welfare LP, prices = duals)")
    print(f"Roll window: t={ROLL_START:+d} .. {ROLL_END:+d}; "
          f"each re-solve plans to t={T_LAST:+d}")
    print("=" * 78, flush=True)
    roll()
