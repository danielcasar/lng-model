"""
Step 11: Two-leader stochastic Stackelberg EPEC -- USA + Gulf (Qatar+UAE).

Two strategic Stackelberg leaders: the USA and a composite "Gulf" leader
(Qatar + UAE, the Hormuz-transiting exporters, jointly ~20% of global LNG).
Australia and Russia remain in the fringe (Asia-locked, genuinely
degenerate strategic decision). The EPEC across the two leaders is solved
by Gauss-Seidel diagonalization with damped best-response.

Rationale: the Gulf composite is the only non-US actor whose strategic
decision is economically meaningful in this scenario because (i) it is the
supplier being blocked, (ii) it has discretionary allocation between EU
and Asia when the strait is open, and (iii) its pre-/post-crisis
positioning is exactly the cross-basin reallocation a multi-leader EPEC is
designed to capture. Qatar and the UAE face identical blocking exposure
(93% / 96% of their shipments transit Hormuz), so aggregating them is the
natural coalition. With two leaders, diagonalization converges much faster
than the 4-leader case while preserving the multi-leader structure.

Implementation:
  - Each leader L solves its own MPCC with the other leader's q^{-L}
    held as a fixed parameter (read from the previous iteration).
  - Per iteration: solve USA's MPCC, then Gulf's MPCC, then apply a
    damped update q^{(i+1)} = alpha * q_best + (1-alpha) * q^{(i)}.
  - Convergence declared when max |dq| < 0.5 bcm across leaders, regions
    and tree nodes. Damping alpha = 0.4.
  - Wall time approx. 2 min on a 24-core server, ~60-90 min on a laptop.
"""

import time
import pyomo.environ as pyo
import lng_data as ld
from scenario_tree import (
    build_tree, calendar_month,
    T_FIRST, T_PRE_END, T_CLOSURE_START, T_CLOSURE_END,
    T_POST_START, T_LAST,
)

# ALL configuration values (with source citations) live in model_config.py;
# this script contains only model logic.
from model_config import (
    EVENT_NAME,
    LEADERS, LEADER_REGIONS, GULF_MEMBERS, BLOCKED_LEADERS, CONTRACT_FLOOR,
    SPOT_TRADABLE, EU_ACCESS, ASIA_ACCESS, pipeline,
    demand_blocks_base,
    EU_MONTH_FACTOR, WINTER, SUMMER, ASIA_WINTER_FACTOR, ASIA_SUMMER_FACTOR,
    HOLDING_COST, storage, EU_NOV_TARGET_FRAC, EU_MAX_FILL,
    EU_MAX_INJECT_BCM, EU_MAX_WITHDRAW_BCM,
    M_X, M_D, M_PI, M_DUE, M_STOCK,
)

# =============================================================================
# DERIVED MARKET STRUCTURE (from model_config + lng_data)
# =============================================================================

EVENT = ld.EVENTS[EVENT_NAME]

EU_FRINGE_share   = {e: SPOT_TRADABLE * s for e, s in EU_ACCESS.items()}
ASIA_FRINGE_share = {e: SPOT_TRADABLE * s for e, s in ASIA_ACCESS.items()}

lng_EU_fringe   = ld.regional_supply("EU",   list(EU_FRINGE_share),   EU_FRINGE_share,
                                     blocked_suppliers=EVENT["blocked_suppliers"])
lng_Asia_fringe = ld.regional_supply("Asia", list(ASIA_FRINGE_share), ASIA_FRINGE_share,
                                     blocked_suppliers=EVENT["blocked_suppliers"])
fringe = {
    "EU":   {**pipeline["EU"],   **lng_EU_fringe},
    "Asia": {**pipeline["Asia"], **lng_Asia_fringe},
}

# =============================================================================
# LEADER COSTS AND CAPACITIES
# =============================================================================

# USA: costs and capacity straight from the data. Gulf composite: capacity
# is Qatar + UAE (Other_Middle_East); delivered cost uses Qatar's BEP and
# transport costs -- the capacity-weighted BEP differs by < $0.03/MMBtu
# (UAE share ~4% of the composite), so Qatar's cost is used for both members.
leader_cost = {
    "USA":  {r: ld.delivered_cost_eur_mwh("USA",   r) for r in LEADER_REGIONS["USA"]},
    "Gulf": {r: ld.delivered_cost_eur_mwh("Qatar", r) for r in LEADER_REGIONS["Gulf"]},
}
_LEADER_CAP_BASE = {
    "USA":  ld.annual_bn_mmbtu_to_monthly_bcm(ld.LIQ_CAP_BN_MMBTU_YR["USA"]),
    "Gulf": sum(ld.annual_bn_mmbtu_to_monthly_bcm(ld.LIQ_CAP_BN_MMBTU_YR[m])
                for m in GULF_MEMBERS),
}

def leader_cap_at_node(L, node):
    """Monthly capacity for leader L at scenario-tree node n.
    Zero if the Strait is closed at the node and L is Hormuz-stranded."""
    if (not node.closure_open) and L in BLOCKED_LEADERS:
        return 0.0
    return _LEADER_CAP_BASE[L]

# =============================================================================
# DEMAND / SEASONALITY HELPERS (values in model_config.py)
# =============================================================================

def season_factor(r, t):
    m = calendar_month(t)
    if r == "EU":
        return EU_MONTH_FACTOR[m]
    else:
        if m in WINTER: return ASIA_WINTER_FACTOR
        if m in SUMMER: return ASIA_SUMMER_FACTOR
        return 1.00

def cost(r, s):     return fringe[r][s]["cost"]
def Xcap(r, s, node):
    return fringe[r][s]["cap_closed" if not node.closure_open else "cap_open"]
def Vcap(r, k, t):  return demand_blocks_base[r][k][0] * season_factor(r, t)
def wtp(r, k):      return demand_blocks_base[r][k][1]

# =============================================================================
# BUILD SCENARIO TREE (shared across both leaders)
# =============================================================================

REGIONS  = ("EU", "Asia")
S_by_r   = {r: list(fringe[r].keys()) for r in REGIONS}
K_by_r   = {r: list(range(len(demand_blocks_base[r]))) for r in REGIONS}
EU_NOV_MIN = EU_NOV_TARGET_FRAC * storage["EU"]["S_max"]    # Reg 2017/1938: 90% Nov-1 target

def make_ctx(nodes, realized_ids, s_init):
    """Bundle a scenario tree + initial storage state into the context dict
    consumed by the MPCC builder. The rolling-horizon driver (file 12)
    creates a fresh ctx for every monthly re-solve; the one-shot model uses
    DEFAULT_CTX below."""
    return {
        "NODES":        nodes,
        "NODE_IDS":     list(nodes.keys()),
        "REALIZED_IDS": realized_ids,
        "NOV_NODES":    [nid for nid in realized_ids
                         if calendar_month(nodes[nid].t) == 10],
        "TERMINAL_IDS": [nid for nid, n in nodes.items()
                         if not n.children and n.t == T_LAST],
        "S_INIT":       dict(s_init),
    }

_NODES_FULL, _REALIZED_FULL = build_tree()
DEFAULT_CTX = make_ctx(_NODES_FULL, _REALIZED_FULL,
                       {r: storage[r]["S_init"] for r in REGIONS})

# Backwards-compatible module-level aliases (used by __main__ prints)
NODES        = DEFAULT_CTX["NODES"]
NODE_IDS     = DEFAULT_CTX["NODE_IDS"]
REALIZED_IDS = DEFAULT_CTX["REALIZED_IDS"]

# =============================================================================
# PER-LEADER MPCC BUILDER (one of two)
# =============================================================================

def build_leader_mpcc(L, others_q, ctx=None):
    """Build leader L's MPCC with the other leader's q held fixed.

    others_q[r][nid] = the OTHER leader's supply to region r at node nid.
    ctx = tree context from make_ctx(); defaults to the full-horizon tree.
    """
    ctx = ctx if ctx is not None else DEFAULT_CTX
    NODES                = ctx["NODES"]
    NODE_IDS             = ctx["NODE_IDS"]
    NOV_TARGETS_EU_NODES = ctx["NOV_NODES"]
    TERMINAL_NODE_IDS    = ctx["TERMINAL_IDS"]
    S_INIT               = ctx["S_INIT"]

    accessible = LEADER_REGIONS[L]

    m = pyo.ConcreteModel(f"MPCC_{L}_stochastic")

    m.R  = pyo.Set(initialize=list(REGIONS))
    m.N  = pyo.Set(initialize=NODE_IDS)
    m.RS = pyo.Set(initialize=[(r, s) for r in REGIONS for s in S_by_r[r]], dimen=2)
    m.RK = pyo.Set(initialize=[(r, k) for r in REGIONS for k in K_by_r[r]], dimen=2)

    # Leader's decision: q[r, n] (bounded to 0 if region not accessible)
    def _q_bounds(mdl, r, nid):
        if r in accessible: return (0, None)
        return (0, 0)
    m.q = pyo.Var(m.R, m.N, domain=pyo.NonNegativeReals, bounds=_q_bounds)

    # Followers' primal vars (per node)
    m.x     = pyo.Var(m.RS, m.N, domain=pyo.NonNegativeReals)
    m.d     = pyo.Var(m.RK, m.N, domain=pyo.NonNegativeReals)
    m.stock = pyo.Var(m.R,  m.N, domain=pyo.NonNegativeReals)
    m.flow  = pyo.Var(m.R,  m.N, domain=pyo.Reals)

    # Followers' duals
    m.pi  = pyo.Var(m.R,  m.N, domain=pyo.NonNegativeReals)
    m.lam = pyo.Var(m.RS, m.N, domain=pyo.NonNegativeReals)
    m.mu  = pyo.Var(m.RK, m.N, domain=pyo.NonNegativeReals)
    m.eta = pyo.Var(m.R,  m.N, domain=pyo.NonNegativeReals)

    # Binaries for Big-M complementarity
    m.b_x     = pyo.Var(m.RS, m.N, domain=pyo.Binary)
    m.b_d     = pyo.Var(m.RK, m.N, domain=pyo.Binary)
    m.b_xcap  = pyo.Var(m.RS, m.N, domain=pyo.Binary)
    m.b_dcap  = pyo.Var(m.RK, m.N, domain=pyo.Binary)
    m.b_pi    = pyo.Var(m.R,  m.N, domain=pyo.Binary)
    m.b_stock = pyo.Var(m.R,  m.N, domain=pyo.Binary)
    m.b_eta   = pyo.Var(m.R,  m.N, domain=pyo.Binary)

    # Leader capacity per node
    def _leader_cap(mdl, nid):
        return sum(mdl.q[r, nid] for r in accessible) <= leader_cap_at_node(L, NODES[nid])
    m.lcap = pyo.Constraint(m.N, rule=_leader_cap)

    # Per-leader delivery floor: share of capacity that is NOT strategically
    # withholdable (see CONTRACT_FLOOR in model_config.py). Binds total
    # dispatch, so cross-basin arbitrage stays strategic; zero when blocked.
    def _leader_floor(mdl, nid):
        return (sum(mdl.q[r, nid] for r in accessible)
                >= CONTRACT_FLOOR[L] * leader_cap_at_node(L, NODES[nid]))
    m.lcap_floor = pyo.Constraint(m.N, rule=_leader_floor)

    # Market balance per node (leader's own q + OTHER leader's q + fringe x = demand + storage flow)
    def _balance(mdl, r, nid):
        return (sum(mdl.d[r, k, nid] for k in K_by_r[r]) + mdl.flow[r, nid]
                <= mdl.q[r, nid] + others_q[r].get(nid, 0.0)
                   + sum(mdl.x[r, s, nid] for s in S_by_r[r]))
    m.balance = pyo.Constraint(m.R, m.N, rule=_balance)

    m.xcap = pyo.Constraint(m.RS, m.N,
        rule=lambda mdl, r, s, nid: mdl.x[r, s, nid] <= Xcap(r, s, NODES[nid]))
    m.dcap = pyo.Constraint(m.RK, m.N,
        rule=lambda mdl, r, k, nid: mdl.d[r, k, nid] <= Vcap(r, k, NODES[nid].t))

    def _storage_balance(mdl, r, nid):
        node = NODES[nid]
        if node.parent_id == "":
            prev = S_INIT[r]
        else:
            prev = mdl.stock[r, node.parent_id]
        return mdl.stock[r, nid] == prev + mdl.flow[r, nid]
    m.storage_balance = pyo.Constraint(m.R, m.N, rule=_storage_balance)

    m.storage_cap = pyo.Constraint(m.R, m.N,
        rule=lambda mdl, r, nid: mdl.stock[r, nid] <= storage[r]["S_max"])

    # Observed cycling envelope (EU only): end-of-month fill may not exceed
    # the historical maximum for that calendar month (AGSI+, see above).
    def _cycling_envelope(mdl, r, nid):
        if r != "EU": return pyo.Constraint.Skip
        frac = EU_MAX_FILL[calendar_month(NODES[nid].t)]
        if frac >= 1.0: return pyo.Constraint.Skip
        return mdl.stock[r, nid] <= frac * storage[r]["S_max"]
    m.cycling_env = pyo.Constraint(m.R, m.N, rule=_cycling_envelope)

    # Physical deliverability limits (EU only): injection and withdrawal
    # rates bounded by GIE aggregate technical capacity.
    def _flow_limits_up(mdl, r, nid):
        if r != "EU": return pyo.Constraint.Skip
        return mdl.flow[r, nid] <= EU_MAX_INJECT_BCM
    m.flow_up = pyo.Constraint(m.R, m.N, rule=_flow_limits_up)
    def _flow_limits_dn(mdl, r, nid):
        if r != "EU": return pyo.Constraint.Skip
        return mdl.flow[r, nid] >= -EU_MAX_WITHDRAW_BCM
    m.flow_dn = pyo.Constraint(m.R, m.N, rule=_flow_limits_dn)

    def _terminal(mdl, r, nid):
        if nid not in TERMINAL_NODE_IDS: return pyo.Constraint.Skip
        return mdl.stock[r, nid] == storage[r]["S_term"]
    m.terminal = pyo.Constraint(m.R, m.N, rule=_terminal)

    def _nov(mdl, nid):
        if nid not in NOV_TARGETS_EU_NODES: return pyo.Constraint.Skip
        return mdl.stock["EU", nid] >= EU_NOV_MIN
    m.eu_nov = pyo.Constraint(m.N, rule=_nov)

    # KKT stationarity
    m.stat_x = pyo.Constraint(m.RS, m.N,
        rule=lambda mdl, r, s, nid:
            cost(r, s) + mdl.lam[r, s, nid] - mdl.pi[r, nid] >= 0)
    m.stat_d = pyo.Constraint(m.RK, m.N,
        rule=lambda mdl, r, k, nid:
            mdl.pi[r, nid] + mdl.mu[r, k, nid] - wtp(r, k) >= 0)

    def _stat_stock(mdl, r, nid):
        node = NODES[nid]
        if not node.children: return pyo.Constraint.Skip
        total_child_prob = sum(NODES[c].cum_prob / node.cum_prob for c in node.children) \
                           if node.cum_prob > 0 else 1.0
        expected_pi_next = sum((NODES[c].cum_prob / node.cum_prob if node.cum_prob > 0 else 0)
                               * mdl.pi[r, c] for c in node.children) / max(total_child_prob, 1e-9)
        return mdl.pi[r, nid] - expected_pi_next + mdl.eta[r, nid] + HOLDING_COST >= 0
    m.stat_stock = pyo.Constraint(m.R, m.N, rule=_stat_stock)

    # Fortuny-Amat Big-M complementarity
    m.compl_x_a = pyo.Constraint(m.RS, m.N,
        rule=lambda mdl, r, s, nid: mdl.x[r, s, nid] <= M_X * mdl.b_x[r, s, nid])
    m.compl_x_b = pyo.Constraint(m.RS, m.N,
        rule=lambda mdl, r, s, nid:
            cost(r, s) + mdl.lam[r, s, nid] - mdl.pi[r, nid]
            <= M_DUE * (1 - mdl.b_x[r, s, nid]))

    m.compl_d_a = pyo.Constraint(m.RK, m.N,
        rule=lambda mdl, r, k, nid: mdl.d[r, k, nid] <= M_D * mdl.b_d[r, k, nid])
    m.compl_d_b = pyo.Constraint(m.RK, m.N,
        rule=lambda mdl, r, k, nid:
            mdl.pi[r, nid] + mdl.mu[r, k, nid] - wtp(r, k)
            <= M_DUE * (1 - mdl.b_d[r, k, nid]))

    m.compl_xcap_a = pyo.Constraint(m.RS, m.N,
        rule=lambda mdl, r, s, nid: mdl.lam[r, s, nid] <= M_PI * mdl.b_xcap[r, s, nid])
    m.compl_xcap_b = pyo.Constraint(m.RS, m.N,
        rule=lambda mdl, r, s, nid:
            Xcap(r, s, NODES[nid]) - mdl.x[r, s, nid]
            <= M_X * (1 - mdl.b_xcap[r, s, nid]))

    m.compl_dcap_a = pyo.Constraint(m.RK, m.N,
        rule=lambda mdl, r, k, nid: mdl.mu[r, k, nid] <= M_PI * mdl.b_dcap[r, k, nid])
    m.compl_dcap_b = pyo.Constraint(m.RK, m.N,
        rule=lambda mdl, r, k, nid:
            Vcap(r, k, NODES[nid].t) - mdl.d[r, k, nid]
            <= M_D * (1 - mdl.b_dcap[r, k, nid]))

    def _compl_pi_b(mdl, r, nid):
        node = NODES[nid]
        slack = (mdl.q[r, nid] + others_q[r].get(nid, 0.0)
                 + sum(mdl.x[r, s, nid] for s in S_by_r[r])
                 - sum(mdl.d[r, k, nid] for k in K_by_r[r]) - mdl.flow[r, nid])
        big = (leader_cap_at_node(L, node) + others_q[r].get(nid, 0.0)
               + sum(Xcap(r, s, node) for s in S_by_r[r]) + M_STOCK + 10.0)
        return slack <= big * (1 - mdl.b_pi[r, nid])
    m.compl_pi_a = pyo.Constraint(m.R, m.N,
        rule=lambda mdl, r, nid: mdl.pi[r, nid] <= M_PI * mdl.b_pi[r, nid])
    m.compl_pi_b = pyo.Constraint(m.R, m.N, rule=_compl_pi_b)

    def _compl_stock_a(mdl, r, nid):
        if not NODES[nid].children: return pyo.Constraint.Skip
        return mdl.stock[r, nid] <= M_STOCK * mdl.b_stock[r, nid]
    m.compl_stock_a = pyo.Constraint(m.R, m.N, rule=_compl_stock_a)

    def _compl_stock_b(mdl, r, nid):
        node = NODES[nid]
        if not node.children: return pyo.Constraint.Skip
        expected_pi_next = sum((NODES[c].cum_prob / node.cum_prob if node.cum_prob > 0 else 0)
                               * mdl.pi[r, c] for c in node.children) \
                           / max(sum(NODES[c].cum_prob / node.cum_prob for c in node.children)
                                 if node.cum_prob > 0 else 1.0, 1e-9)
        return (mdl.pi[r, nid] - expected_pi_next + mdl.eta[r, nid] + HOLDING_COST) \
               <= M_DUE * (1 - mdl.b_stock[r, nid])
    m.compl_stock_b = pyo.Constraint(m.R, m.N, rule=_compl_stock_b)

    m.compl_eta_a = pyo.Constraint(m.R, m.N,
        rule=lambda mdl, r, nid: mdl.eta[r, nid] <= M_PI * mdl.b_eta[r, nid])
    m.compl_eta_b = pyo.Constraint(m.R, m.N,
        rule=lambda mdl, r, nid:
            storage[r]["S_max"] - mdl.stock[r, nid]
            <= M_STOCK * (1 - mdl.b_eta[r, nid]))

    # Leader's objective: expected probability-weighted profit
    m.obj = pyo.Objective(
        sense=pyo.maximize,
        expr=sum(NODES[nid].cum_prob * (m.pi[r, nid] - leader_cost[L][r]) * m.q[r, nid]
                 for r in accessible for nid in NODE_IDS))

    return m

# =============================================================================
# DIAGONALIZATION
# =============================================================================

def init_quantities(ctx=None):
    ctx = ctx if ctx is not None else DEFAULT_CTX
    NODES, NODE_IDS = ctx["NODES"], ctx["NODE_IDS"]
    q = {}
    for L in LEADERS:
        regs = LEADER_REGIONS[L]
        q[L] = {r: {nid: 0.0 for nid in NODE_IDS} for r in REGIONS}
        for nid in NODE_IDS:
            node = NODES[nid]
            cap = leader_cap_at_node(L, node)
            share = 0.5 * cap / max(1, len(regs))
            for r in regs:
                q[L][r][nid] = share
    return q

def solve_leader(L, others_q, ctx=None, time_limit=180, mip_gap=3e-2):
    # During diagonalization iterations a loose 3% gap is sufficient: the
    # damped update only uses the best response directionally, and the
    # equilibrium is refined across iterations anyway. The final storage-
    # extraction solve uses a tighter gap (see __main__). With the 17-block
    # calibrated demand staircase the MIQCP has ~2,000 demand-side binaries
    # (3x the coarse grid), so per-solve effort is materially higher.
    ctx = ctx if ctx is not None else DEFAULT_CTX
    NODE_IDS = ctx["NODE_IDS"]
    m = build_leader_mpcc(L, others_q, ctx=ctx)
    solver = pyo.SolverFactory("gurobi")
    solver.options["NonConvex"]  = 2
    solver.options["MIPGap"]     = mip_gap
    solver.options["TimeLimit"]  = time_limit
    solver.options["OutputFlag"] = 0
    results = solver.solve(m, tee=False, load_solutions=False)
    try:
        m.solutions.load_from(results)
    except Exception:
        return None, None, None, None
    try:
        q_new  = {r: {nid: max(0.0, pyo.value(m.q[r, nid])) for nid in NODE_IDS} for r in REGIONS}
        prices = {r: {nid: pyo.value(m.pi[r, nid]) for nid in NODE_IDS} for r in REGIONS}
        stocks = {r: {nid: pyo.value(m.stock[r, nid]) for nid in NODE_IDS} for r in REGIONS}
        profit = pyo.value(m.obj)
    except Exception:
        return None, None, None, None
    return q_new, prices, profit, stocks

def max_change(q_old, q_new, ctx=None):
    ctx = ctx if ctx is not None else DEFAULT_CTX
    NODE_IDS = ctx["NODE_IDS"]
    return max(abs(q_old[L][r][nid] - q_new[L][r][nid])
               for L in LEADERS for r in REGIONS for nid in NODE_IDS)

def diagonalize(ctx=None, max_iter=8, tol=0.5, alpha=0.4,
                time_limit=180, verbose=True):
    ctx = ctx if ctx is not None else DEFAULT_CTX
    NODE_IDS, REALIZED_IDS = ctx["NODE_IDS"], ctx["REALIZED_IDS"]
    q = init_quantities(ctx)
    last_prices  = None
    last_profits = {}
    last_stocks  = None
    t_start = time.time()

    for it in range(max_iter):
        t_iter = time.time()
        if verbose:
            print(f"\n--- Iteration {it+1} ---", flush=True)
        q_prev = {L: {r: dict(q[L][r]) for r in REGIONS} for L in LEADERS}

        for L in LEADERS:
            t_solve = time.time()
            other_L = [Lp for Lp in LEADERS if Lp != L][0]
            others = {r: q[other_L][r] for r in REGIONS}
            q_br, prices, profit, stocks = solve_leader(L, others, ctx=ctx,
                                                        time_limit=time_limit)
            solve_secs = time.time() - t_solve
            if q_br is None:
                print(f"  {L:10s}  SOLVER FAILED, keeping previous q "
                      f"[{solve_secs:.0f}s]", flush=True)
                continue
            for r in REGIONS:
                for nid in NODE_IDS:
                    q[L][r][nid] = alpha * q_br[r][nid] + (1 - alpha) * q[L][r][nid]
            last_prices    = prices
            last_profits[L] = profit
            last_stocks    = stocks
            if verbose:
                avg_eu = sum(q[L]["EU"][nid] for nid in REALIZED_IDS) / len(REALIZED_IDS)
                avg_as = sum(q[L]["Asia"][nid] for nid in REALIZED_IDS) / len(REALIZED_IDS)
                print(f"  {L:10s}  E[profit]={profit:10.1f}  "
                      f"q_EU_realized={avg_eu:5.2f}  q_AS_realized={avg_as:5.2f}  "
                      f"[solve {solve_secs:.0f}s]", flush=True)

        delta = max_change(q_prev, q, ctx)
        iter_secs  = time.time() - t_iter
        total_secs = time.time() - t_start
        if verbose:
            print(f"  max |dq| = {delta:.3f}   "
                  f"[iteration {iter_secs:.0f}s, total {total_secs/60:.1f}min]", flush=True)
        if delta < tol:
            if verbose:
                print(f"\n*** Converged after {it+1} iterations (tol={tol}). "
                      f"Total wall time: {(time.time()-t_start)/60:.1f} min ***", flush=True)
            return q, last_prices, last_profits, it + 1, last_stocks

    if verbose:
        print(f"\n!!! No convergence after {max_iter} iterations (last dq={delta:.3f}). "
              f"Total wall time: {(time.time()-t_start)/60:.1f} min", flush=True)
    return q, last_prices, last_profits, max_iter, last_stocks

# =============================================================================
# RUN
# =============================================================================

if __name__ == "__main__":
    t_script = time.time()
    print("=" * 90)
    print("Two-leader stochastic Stackelberg EPEC (USA + Gulf[Qatar+UAE])")
    print(f"Event: {EVENT['name']}")
    print(f"Tree nodes: {len(NODE_IDS)}; realized path: {len(REALIZED_IDS)}")
    print(f"Strategic leaders: {LEADERS} (Gulf = Qatar + UAE composite)")
    print(f"Fringe (price-taking): Australia, Russia (Asia) + Norway/Algeria/Sakhalin pipelines")
    print("=" * 90, flush=True)

    q_eq, prices, profits, iters, _ = diagonalize()

    MONTH_NAMES = ["", "Jan","Feb","Mar","Apr","May","Jun",
                   "Jul","Aug","Sep","Oct","Nov","Dec"]

    print("\n" + "=" * 90)
    print(f"Final equilibrium (after {iters} iterations)")
    print("=" * 90)
    print("\nLeader expected profits over scenario tree:")
    for L in LEADERS:
        print(f"  {L:10s}  {profits.get(L, float('nan')):10.1f}")

    # Re-solve one final pass of each leader's MPCC to extract storage levels
    # along the realized path (these are follower-side decisions inside the
    # leader's MPCC; we read them from the last USA solve).
    print("\nExtracting storage trajectory from final MPCC solve...", flush=True)
    final_others = {r: q_eq["Gulf"][r] for r in REGIONS}
    final_m = build_leader_mpcc("USA", final_others)
    final_solver = pyo.SolverFactory("gurobi")
    final_solver.options["NonConvex"]  = 2
    final_solver.options["MIPGap"]     = 1e-2
    final_solver.options["TimeLimit"]  = 300
    final_solver.options["OutputFlag"] = 0
    final_solver.solve(final_m, tee=False)
    s_eu = {nid: pyo.value(final_m.stock["EU", nid]) for nid in REALIZED_IDS}

    print("\nRealized-path equilibrium prices, dispatches, and EU storage:")
    print("(Gulf = Qatar + UAE composite leader)")
    hdr = (f"{'t':>3} {'mo':>4} {'status':>8}  {'p_EU':>7} {'p_AS':>7}   "
           f"{'USA_EU':>7} {'USA_AS':>7} {'GLF_EU':>7} {'GLF_AS':>7}  {'S_EU':>6}")
    print(hdr)
    print("-" * len(hdr))
    for nid in REALIZED_IDS:
        n = NODES[nid]
        mo = MONTH_NAMES[calendar_month(n.t)]
        status = "OPEN" if n.closure_open else "CLOSED"
        pe = prices["EU"][nid] if prices else float('nan')
        pa = prices["Asia"][nid] if prices else float('nan')
        print(f"{n.t:>+3d} {mo:>4} {status:>8}  {pe:>7.2f} {pa:>7.2f}   "
              f"{q_eq['USA']['EU'][nid]:>7.2f} {q_eq['USA']['Asia'][nid]:>7.2f} "
              f"{q_eq['Gulf']['EU'][nid]:>7.2f} {q_eq['Gulf']['Asia'][nid]:>7.2f}  "
              f"{s_eu[nid]:>6.1f}")

    # ------------------------------------------------------------------
    # CALIBRATION REPORT: model vs observed prices (Sep 2025 - Jun 2026)
    # Targets from calibration_targets.csv (TTF / JKM monthly averages,
    # converted at USD/MMBtu x 3.17 = EUR/MWh).
    # ------------------------------------------------------------------
    TARGETS = {  # t: (TTF_eur, JKM_eur)
        -5: (33.0, 36.0), -4: (34.0, 37.0), -3: (36.0, 38.0),
        -2: (38.0, 39.0), -1: (37.0, 38.0),  0: (37.0, 38.0),
         1: (57.0, 63.0),  2: (42.0, 55.5),  3: (46.0, 57.0),
         4: (49.6, 59.5),
    }
    print("\n" + "=" * 70)
    print("CALIBRATION REPORT: model vs observed (Sep 2025 - Jun 2026)")
    print("=" * 70)
    print(f"{'t':>3} {'mo':>6}  {'EU_mod':>7} {'EU_obs':>7} {'dEU':>6}   "
          f"{'AS_mod':>7} {'AS_obs':>7} {'dAS':>6}")
    sq_err, n_obs = 0.0, 0
    for nid in REALIZED_IDS:
        n = NODES[nid]
        if n.t not in TARGETS:
            continue
        eu_obs, as_obs = TARGETS[n.t]
        eu_mod = prices["EU"][nid] if prices else float('nan')
        as_mod = prices["Asia"][nid] if prices else float('nan')
        mo = MONTH_NAMES[calendar_month(n.t)]
        print(f"{n.t:>+3d} {mo:>6}  {eu_mod:>7.1f} {eu_obs:>7.1f} {eu_mod-eu_obs:>+6.1f}   "
              f"{as_mod:>7.1f} {as_obs:>7.1f} {as_mod-as_obs:>+6.1f}")
        sq_err += (eu_mod - eu_obs) ** 2 + (as_mod - as_obs) ** 2
        n_obs  += 2
    rmse = (sq_err / max(n_obs, 1)) ** 0.5
    print("-" * 70)
    print(f"RMSE over {n_obs} observations: {rmse:.2f} EUR/MWh")

    total_min = (time.time() - t_script) / 60
    print(f"\nTotal computing time: {total_min:.1f} min "
          f"({iters} diagonalization iterations + final storage extraction)", flush=True)
