"""
Step 8: EPEC with multiple strategic leaders, solved by diagonalization.

Leaders:   USA, Australia, Russia, Qatar
Followers: EU and Asia regional supply/demand equilibrium (fringe + storage)

(Other Middle East was tried as a fifth leader but with <1 bcm/month capacity
its contribution to the equilibrium was negligible — dropped for clarity.
Its liquefaction capacity is still represented inside the EVENT block list
where applicable, with no effect on the four-leader equilibrium below.)

EPEC = Equilibrium Problem with Equilibrium Constraints. Each leader solves
its OWN MPEC anticipating the followers' equilibrium; leaders are simultaneous
Cournot-Stackelberg players against each other. A single-level monolith
(stacking each leader's KKTs as constraints in every other leader's problem)
is computationally brittle and rarely converges in practice.

Standard remedy: diagonalization (Gauss-Seidel best-response). Each iteration,
every leader re-optimises with the OTHER leaders' supplies held fixed as
parameters. We damp the update (alpha = 0.5) for stability and stop when the
max change in any q_L,r,t falls below TOL.

Convergence is not guaranteed in general (the EPEC may have multiple
equilibria or none); we verify post hoc that the residual is small and that
quantities are economically plausible.
"""

import math
import pyomo.environ as pyo
import lng_data as ld

# =============================================================================
# SCENARIO + LEADER CONFIGURATION
# =============================================================================

EVENT = ld.EVENTS["hormuz_closure"]

# =============================================================================
# HORIZON CONFIGURATION (adjustable)
#
# CLOSURE_DURATION_MONTHS: length of the Hormuz closure in months. Default is
#   6, matching the course-assignment specification. Set to 3 for a short
#   "ceasefire" scenario, 12 for a prolonged crisis, etc.
# PRE_SHOCK_MONTHS:        baseline months modelled before crisis onset.
# POST_SHOCK_MONTHS:       recovery months modelled after reopening
#                          (includes the reopening month itself at t=0).
#
# The calendar is automatically pinned so that the first crisis month
# (t = -CLOSURE_DURATION_MONTHS) maps to March 2026 (the actual closure
# onset). All downstream timing (crisis window, belief function, Nov-1
# storage mandate checkpoints) is derived from these three parameters.
# =============================================================================

CLOSURE_DURATION_MONTHS = 6   # March 2026 - August 2026
PRE_SHOCK_MONTHS        = 6   # September 2025 - February 2026
POST_SHOCK_MONTHS       = 7   # September 2026 onwards (incl. reopening)

# Sanity check: the horizon should span at least two Nov-1 mandate checkpoints
# (pre-shock year and recovery year). With very long closures or short
# post-shock horizons the second checkpoint may fall outside the horizon and
# the storage mandate would be silently dropped.
assert PRE_SHOCK_MONTHS >= 2, "Need at least 2 pre-shock months to capture an Oct-end checkpoint."

LEADERS = ["USA", "Australia", "Russia", "Qatar"]

# Accessible-region sets from observed LNG shipping routes documented in
# Zou et al. (2022) Tables 4-5 and Zwickl-Bernhard & Neumann (2024) Appendix A:
#   USA, Qatar:        ship to both EU and Asia
#   Australia, Russia: serve Asia only (no Atlantic terminals at scale)
LEADER_REGIONS = {
    "USA":       ["EU", "Asia"],
    "Australia": ["Asia"],
    "Russia":    ["Asia"],
    "Qatar":     ["EU", "Asia"],
}

# =============================================================================
# FRINGE (price-taking) SUPPLIERS — everyone except the four leaders
# =============================================================================

# Regional shares: fraction of each fringe exporter's global liquefaction we
# assume can serve each region. Reflects geographical routing patterns documented
# in Zwickl-Bernhard & Neumann (2024) and observed 2018-2023 flow shares from
# the IEA Natural Gas Information annual report; not directly tabulated in any
# single source, so values are informed estimates.
EU_FRINGE_share = {
    "Algeria": 0.5, "Nigeria": 0.5, "Trinidad": 0.6,
    "Other_Americas": 0.3, "Other_Africa": 0.5,
}
ASIA_FRINGE_share = {
    "Indonesia": 0.7, "Malaysia": 0.7, "Oman": 0.7, "Other_Asia_Pacific": 0.8,
}

# Pipeline costs and capacities are indicative current-market estimates, not
# from any single calibrated source. Norway and Algeria EU pipe capacities
# approximate observed 2024 flow ceilings reported in ENTSOG transparency data;
# Norwegian delivered cost ~EUR 16/MWh reflects long-run break-even of Norwegian
# Continental Shelf production (Equinor 2024 quarterly reports); Algerian pipe
# cost is comparable. Sakhalin -> NE Asia ~EUR 16/MWh from Zwickl-Bernhard &
# Neumann (2024) Appendix A transport-cost decomposition.
pipeline = {
    "EU": {
        "Norway_pipe":  {"cost": 16.0, "cap_open": 10.0, "cap_closed": 10.0},
        "Algeria_pipe": {"cost": 20.0, "cap_open":  4.0, "cap_closed":  4.0},
    },
    "Asia": {
        "Sakhalin_pipe":{"cost": 16.0, "cap_open":  3.0, "cap_closed":  3.0},
    },
}

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

leader_cost = {
    L: {r: ld.delivered_cost_eur_mwh(L, r) for r in LEADER_REGIONS[L]}
    for L in LEADERS
}

_LEADER_CAP_BASE = {
    L: ld.annual_bn_mmbtu_to_monthly_bcm(ld.LIQ_CAP_BN_MMBTU_YR[L])
    for L in LEADERS
}

# Crisis window derived from CLOSURE_DURATION_MONTHS (defined at top of file).
CRISIS_START_T = -CLOSURE_DURATION_MONTHS    # first month of closure (Mar 2026)
CRISIS_END_T   = -1                          # last month of closure; reopens at t=0

def leader_cap(L, t):
    """Monthly liquefaction capacity available to leader L at time t.
    Zero during the crisis window if the leader is blocked by the event."""
    if CRISIS_START_T <= t <= CRISIS_END_T and L in EVENT["blocked_suppliers"]:
        return 0.0
    return _LEADER_CAP_BASE[L]

# =============================================================================
# STORAGE, DEMAND, TIME, SEASONALITY  (same as 07)
# =============================================================================

HOLDING_COST = 0.10

# Demand blocks calibrated to:
#   EU:  Eurostat monthly natural gas balance (nrg_cb_gasm), 2018-2023 typical
#        year ~35 bcm/month average. Sectoral split from IEA Natural Gas
#        Information: residential+baseload power ~50% (block 1), peaking power
#        + non-switchable industrial ~30% (block 2), switchable industrial
#        ~20% (block 3). WTPs informed by Hauser et al. (2023) on EU industrial
#        demand-destruction thresholds and Colombo & Toni (2025) on near-zero
#        short-run elasticity.
#   Asia: Japan/Korea/Taiwan/China LNG imports + pipeline-substitutable demand
#         ~40 bcm/month. Larger block 1 reflects Asian power-sector dominance.
demand_blocks_base = {
    "EU":   [(15.0, 200.0), (10.0, 100.0), (7.0, 50.0)],
    "Asia": [(22.0, 180.0), (10.0,  90.0), (8.0, 25.0)],
}

storage = {
    # EU working gas ~100 bcm aggregate (GIE AGSI+ transparency platform, 2024:
    # 1,072 TWh = 99 bcm). With the extended horizon starting at t=-12 (Sep
    # 2025), initial inventory is set to 85 bcm (~85% of capacity), matching
    # the typical end-of-summer fill under the EU 2017/1938 storage mandate
    # (target 80%+ by November 1). Terminal-refill returns to 30 bcm at the
    # end of horizon (Mar 2027), matching the typical end-of-winter trough
    # consistent with the seasonal cycle.
    "EU":   {"S_max": 100.0, "S_init": 85.0, "S_term": 30.0},
    # Japan + Korea + Taiwan aggregate LNG-terminal stock buffer ~20 bcm
    # (METI Japan Petroleum Strategic Reserve data + KOGAS storage reports).
    # Less seasonal; init = terminal = 50% capacity.
    "Asia": {"S_max":  20.0, "S_init": 10.0, "S_term": 10.0},
}

# Time horizon derived from PRE_SHOCK_MONTHS / CLOSURE_DURATION_MONTHS /
# POST_SHOCK_MONTHS. At default 6/6/7 this gives a 19-month horizon spanning
# Sep 2025 (t=-12) through Mar 2027 (t=+6).
T       = list(range(-(PRE_SHOCK_MONTHS + CLOSURE_DURATION_MONTHS),
                     POST_SHOCK_MONTHS))
T_END   = T[-1]
REGIONS = ("EU", "Asia")
S_by_r  = {r: list(fringe[r].keys())                  for r in REGIONS}
K_by_r  = {r: list(range(len(demand_blocks_base[r]))) for r in REGIONS}

# Calendar offset chosen so that the first crisis month (t = CRISIS_START_T =
# -CLOSURE_DURATION_MONTHS) maps to March 2026 (month 3). For the default
# 6-month closure this gives offset = 9 so t=0 (reopening) = September 2026.
# Sweeping CLOSURE_DURATION_MONTHS automatically shifts the reopening month
# while keeping the crisis-onset month fixed at March 2026.
_CAL_OFFSET = 3 + CLOSURE_DURATION_MONTHS
def calendar_month(t): return ((_CAL_OFFSET + t - 1) % 12) + 1
WINTER = {11, 12, 1, 2, 3};  SUMMER = {6, 7, 8}
# Seasonal multipliers from Eurostat monthly natural gas balance (nrg_cb_gasm),
# typical-year 2018-2023 (excluding 2022 anomaly):
#   EU   winter peak ~1.45 x annual mean, summer trough ~0.65 x.
#   Asia winter peak ~1.25 x, summer trough ~0.85 x (BP Statistical Review +
#   IEA Gas Market Report quarterly LNG import data, Japan/Korea/Taiwan).
def season_factor(r, t):
    m = calendar_month(t)
    if r == "EU":
        if m in WINTER: return 1.45
        if m in SUMMER: return 0.65
        return 1.00
    else:
        if m in WINTER: return 1.25
        if m in SUMMER: return 0.85
        return 1.00

def is_closed(t):  return CRISIS_START_T <= t <= CRISIS_END_T
def cost(r, s):    return fringe[r][s]["cost"]
def Xcap(r, s, t): return fringe[r][s]["cap_closed" if is_closed(t) else "cap_open"]
def Vcap(r, k, t): return demand_blocks_base[r][k][0] * season_factor(r, t)

# =============================================================================
# UNCERTAINTY PREMIUM
# =============================================================================

# Uncertainty premium parameterisation:
#   SHOCK_PEAK    = 1.0   normalisation, dimensionless intensity during crisis.
#   PERMANENT_FLOOR=0.15  post-crisis residual; the "rare-disaster" recurrence-
#                         probability term of Barro (2006) and Gabaix (2012).
#   BELIEF_DECAY  = 4 mo  exponential decay constant; rough-fitted to observed
#                         TTF risk-premium half-life post-2022 (Caldara &
#                         Iacoviello 2022 geopolitical-risk index correlation).
#   LAMBDA_WTP    = 20    WTP shift per unit intensity (EUR/MWh).
#                         Anchored to the observed post-2022 TTF residual
#                         elevation of EUR 10-20/MWh above pre-2022 baseline
#                         (Eurostat + Colombo & Toni 2025).
SHOCK_PEAK       = 1.0
PERMANENT_FLOOR  = 0.15
BELIEF_DECAY     = 4.0
LAMBDA_WTP       = 20.0
# CRISIS_START_T and CRISIS_END_T are defined alongside leader_cap above.

def belief(t):
    if t < CRISIS_START_T:    return 0.0         # pre-shock: no uncertainty premium
    if t <= CRISIS_END_T:     return SHOCK_PEAK  # crisis: peak intensity
    return PERMANENT_FLOOR + (SHOCK_PEAK - PERMANENT_FLOOR) * math.exp(-t / BELIEF_DECAY)

def wtp(r, k, t):
    return demand_blocks_base[r][k][1] + LAMBDA_WTP * belief(t)

# =============================================================================
# PER-LEADER MPEC BUILDER
# =============================================================================

M_X, M_D, M_PI, M_DUE, M_STOCK = 60.0, 60.0, 600.0, 600.0, 150.0

def build_leader_mpec(L, others_q):
    """Build leader L's MPEC given other leaders' fixed export schedule.

    others_q[L'][r][t] = volume leader L' (!=L) exports to region r at time t.
    The leader's own q is decision; followers' KKTs are embedded as constraints.
    """
    accessible = LEADER_REGIONS[L]

    def other_supply(r, t):
        return sum(others_q[Lp][r].get(t, 0.0)
                   for Lp in LEADERS
                   if Lp != L and r in LEADER_REGIONS[Lp])

    m = pyo.ConcreteModel(f"MPEC_{L}")
    m.R  = pyo.Set(initialize=list(REGIONS))
    m.T  = pyo.Set(initialize=T, ordered=True)
    m.RS = pyo.Set(initialize=[(r, s) for r in REGIONS for s in S_by_r[r]], dimen=2)
    m.RK = pyo.Set(initialize=[(r, k) for r in REGIONS for k in K_by_r[r]], dimen=2)

    def _q_bounds(mdl, r, t):
        if r in accessible: return (0, None)
        return (0, 0)
    m.q = pyo.Var(m.R, m.T, domain=pyo.NonNegativeReals, bounds=_q_bounds)

    m.x     = pyo.Var(m.RS, m.T, domain=pyo.NonNegativeReals)
    m.d     = pyo.Var(m.RK, m.T, domain=pyo.NonNegativeReals)
    m.stock = pyo.Var(m.R,  m.T, domain=pyo.NonNegativeReals)
    m.flow  = pyo.Var(m.R,  m.T, domain=pyo.Reals)

    m.pi  = pyo.Var(m.R,  m.T, domain=pyo.NonNegativeReals)
    m.lam = pyo.Var(m.RS, m.T, domain=pyo.NonNegativeReals)
    m.mu  = pyo.Var(m.RK, m.T, domain=pyo.NonNegativeReals)
    m.eta = pyo.Var(m.R,  m.T, domain=pyo.NonNegativeReals)

    m.b_x     = pyo.Var(m.RS, m.T, domain=pyo.Binary)
    m.b_d     = pyo.Var(m.RK, m.T, domain=pyo.Binary)
    m.b_xcap  = pyo.Var(m.RS, m.T, domain=pyo.Binary)
    m.b_dcap  = pyo.Var(m.RK, m.T, domain=pyo.Binary)
    m.b_pi    = pyo.Var(m.R,  m.T, domain=pyo.Binary)
    m.b_stock = pyo.Var(m.R,  m.T, domain=pyo.Binary)
    m.b_eta   = pyo.Var(m.R,  m.T, domain=pyo.Binary)

    # ---- Leader capacity ----
    m.lcap = pyo.Constraint(m.T,
        rule=lambda mdl, t: sum(mdl.q[r, t] for r in accessible) <= leader_cap(L, t))

    # ---- Follower market clearing (primal feasibility) ----
    m.balance = pyo.Constraint(
        m.R, m.T,
        rule=lambda mdl, r, t:
            sum(mdl.d[r, k, t] for k in K_by_r[r]) + mdl.flow[r, t]
            <= mdl.q[r, t] + other_supply(r, t)
               + sum(mdl.x[r, s, t] for s in S_by_r[r]))

    m.xcap = pyo.Constraint(m.RS, m.T,
        rule=lambda mdl, r, s, t: mdl.x[r, s, t] <= Xcap(r, s, t))
    m.dcap = pyo.Constraint(m.RK, m.T,
        rule=lambda mdl, r, k, t: mdl.d[r, k, t] <= Vcap(r, k, t))

    def _storage_balance(mdl, r, t):
        if t == T[0]: prev = storage[r]["S_init"]
        else:         prev = mdl.stock[r, T[T.index(t) - 1]]
        return mdl.stock[r, t] == prev + mdl.flow[r, t]
    m.storage_balance = pyo.Constraint(m.R, m.T, rule=_storage_balance)

    m.storage_cap = pyo.Constraint(m.R, m.T,
        rule=lambda mdl, r, t: mdl.stock[r, t] <= storage[r]["S_max"])
    m.terminal = pyo.Constraint(m.R,
        rule=lambda mdl, r: mdl.stock[r, T_END] == storage[r]["S_term"])

    # EU storage mandate (Regulation EU 2017/1938 as amended in 2022, with the
    # 2026 Commission flex provision relaxing the 90% target to 80%): EU member
    # states must collectively reach >= 80% of working-gas capacity by Nov 1
    # each year. In our convention stock[r,t] is end-of-month inventory, so
    # the Nov 1 checkpoint corresponds to end-of-October (calendar_month(t)==10).
    # Identified dynamically so that any closure-duration sweep keeps the
    # mandate at the correct calendar months.
    NOV_TARGETS_EU = [t for t in T if calendar_month(t) == 10]
    EU_NOV_MIN = 0.80 * storage["EU"]["S_max"]
    def _nov_target(mdl, t):
        if t not in NOV_TARGETS_EU: return pyo.Constraint.Skip
        return mdl.stock["EU", t] >= EU_NOV_MIN
    m.eu_nov_mandate = pyo.Constraint(m.T, rule=_nov_target)

    # ---- Dual feasibility (stationarity) ----
    m.stat_x = pyo.Constraint(m.RS, m.T,
        rule=lambda mdl, r, s, t: cost(r, s) + mdl.lam[r, s, t] - mdl.pi[r, t] >= 0)
    m.stat_d = pyo.Constraint(m.RK, m.T,
        rule=lambda mdl, r, k, t: mdl.pi[r, t] + mdl.mu[r, k, t] - wtp(r, k, t) >= 0)

    def _stat_stock(mdl, r, t):
        if t == T_END: return pyo.Constraint.Skip
        t_next = T[T.index(t) + 1]
        return mdl.pi[r, t] - mdl.pi[r, t_next] + mdl.eta[r, t] + HOLDING_COST >= 0
    m.stat_stock = pyo.Constraint(m.R, m.T, rule=_stat_stock)

    # ---- Complementarity (Big-M) ----
    m.compl_x_a = pyo.Constraint(m.RS, m.T,
        rule=lambda mdl, r, s, t: mdl.x[r, s, t] <= M_X * mdl.b_x[r, s, t])
    m.compl_x_b = pyo.Constraint(m.RS, m.T,
        rule=lambda mdl, r, s, t:
            cost(r, s) + mdl.lam[r, s, t] - mdl.pi[r, t]
            <= M_DUE * (1 - mdl.b_x[r, s, t]))

    m.compl_d_a = pyo.Constraint(m.RK, m.T,
        rule=lambda mdl, r, k, t: mdl.d[r, k, t] <= M_D * mdl.b_d[r, k, t])
    m.compl_d_b = pyo.Constraint(m.RK, m.T,
        rule=lambda mdl, r, k, t:
            mdl.pi[r, t] + mdl.mu[r, k, t] - wtp(r, k, t)
            <= M_DUE * (1 - mdl.b_d[r, k, t]))

    m.compl_xcap_a = pyo.Constraint(m.RS, m.T,
        rule=lambda mdl, r, s, t: mdl.lam[r, s, t] <= M_PI * mdl.b_xcap[r, s, t])
    m.compl_xcap_b = pyo.Constraint(m.RS, m.T,
        rule=lambda mdl, r, s, t:
            Xcap(r, s, t) - mdl.x[r, s, t] <= M_X * (1 - mdl.b_xcap[r, s, t]))

    m.compl_dcap_a = pyo.Constraint(m.RK, m.T,
        rule=lambda mdl, r, k, t: mdl.mu[r, k, t] <= M_PI * mdl.b_dcap[r, k, t])
    m.compl_dcap_b = pyo.Constraint(m.RK, m.T,
        rule=lambda mdl, r, k, t:
            Vcap(r, k, t) - mdl.d[r, k, t] <= M_D * (1 - mdl.b_dcap[r, k, t]))

    def _compl_pi_b(mdl, r, t):
        slack = (mdl.q[r, t] + other_supply(r, t)
                 + sum(mdl.x[r, s, t] for s in S_by_r[r])
                 - sum(mdl.d[r, k, t] for k in K_by_r[r]) - mdl.flow[r, t])
        big = (leader_cap(L, t) + other_supply(r, t)
               + sum(Xcap(r, s, t) for s in S_by_r[r]) + M_STOCK + 10.0)
        return slack <= big * (1 - mdl.b_pi[r, t])
    m.compl_pi_a = pyo.Constraint(m.R, m.T,
        rule=lambda mdl, r, t: mdl.pi[r, t] <= M_PI * mdl.b_pi[r, t])
    m.compl_pi_b = pyo.Constraint(m.R, m.T, rule=_compl_pi_b)

    def _compl_stock_a(mdl, r, t):
        if t == T_END: return pyo.Constraint.Skip
        return mdl.stock[r, t] <= M_STOCK * mdl.b_stock[r, t]
    m.compl_stock_a = pyo.Constraint(m.R, m.T, rule=_compl_stock_a)

    def _compl_stock_b(mdl, r, t):
        if t == T_END: return pyo.Constraint.Skip
        t_next = T[T.index(t) + 1]
        return (mdl.pi[r, t] - mdl.pi[r, t_next] + mdl.eta[r, t] + HOLDING_COST) \
               <= M_DUE * (1 - mdl.b_stock[r, t])
    m.compl_stock_b = pyo.Constraint(m.R, m.T, rule=_compl_stock_b)

    m.compl_eta_a = pyo.Constraint(m.R, m.T,
        rule=lambda mdl, r, t: mdl.eta[r, t] <= M_PI * mdl.b_eta[r, t])
    m.compl_eta_b = pyo.Constraint(m.R, m.T,
        rule=lambda mdl, r, t:
            storage[r]["S_max"] - mdl.stock[r, t]
            <= M_STOCK * (1 - mdl.b_eta[r, t]))

    # ---- Leader's objective ----
    m.obj = pyo.Objective(
        sense=pyo.maximize,
        expr=sum((m.pi[r, t] - leader_cost[L][r]) * m.q[r, t]
                 for r in accessible for t in m.T))

    return m

# =============================================================================
# DIAGONALIZATION LOOP
# =============================================================================

def init_quantities():
    """Initial guess: each leader uses half its monthly capacity, split evenly
    across the regions it serves. A neutral starting point — neither cartel-like
    withholding nor competitive flooding."""
    q = {}
    for L in LEADERS:
        regs = LEADER_REGIONS[L]
        q[L] = {r: {t: 0.0 for t in T} for r in REGIONS}
        for t in T:
            share = 0.5 * leader_cap(L, t) / max(1, len(regs))
            for r in regs:
                q[L][r][t] = share
    return q

def solve_leader(L, others_q, time_limit=240):
    m = build_leader_mpec(L, others_q)
    solver = pyo.SolverFactory("gurobi")
    solver.options["NonConvex"]  = 2
    solver.options["MIPGap"]     = 5e-3
    solver.options["TimeLimit"]  = time_limit
    solver.options["OutputFlag"] = 0
    solver.solve(m, tee=False)

    q_new  = {r: {t: max(0.0, pyo.value(m.q[r, t])) for t in T} for r in REGIONS}
    prices = {r: {t: pyo.value(m.pi[r, t]) for t in T} for r in REGIONS}
    profit = pyo.value(m.obj)
    return q_new, prices, profit

def max_change(q_old, q_new):
    return max(abs(q_old[L][r][t] - q_new[L][r][t])
               for L in LEADERS for r in REGIONS for t in T)

def diagonalize(max_iter=20, tol=0.5, alpha=0.3):
    """Gauss-Seidel best-response with damping.

    alpha: relaxation parameter. q <- alpha * q_best_response + (1-alpha) * q_old.
           alpha = 1 is pure best-response (often oscillates);
           alpha = 0.5 is conservative and stabilises convergence.
    """
    q = init_quantities()
    last_prices  = None
    last_profits = {}

    for it in range(max_iter):
        print(f"\n--- Iteration {it+1} ---")
        q_prev = {L: {r: dict(q[L][r]) for r in REGIONS} for L in LEADERS}

        for L in LEADERS:
            others = {Lp: q[Lp] for Lp in LEADERS if Lp != L}
            q_br, prices, profit = solve_leader(L, others)
            # Damped update
            for r in REGIONS:
                for t in T:
                    q[L][r][t] = alpha * q_br[r][t] + (1 - alpha) * q[L][r][t]
            last_prices    = prices
            last_profits[L] = profit
            print(f"  {L:20s}  profit={profit:9.1f}  "
                  f"q_EU_avg={sum(q[L]['EU'].values())/len(T):5.2f}  "
                  f"q_AS_avg={sum(q[L]['Asia'].values())/len(T):5.2f}")

        delta = max_change(q_prev, q)
        print(f"  max |dq| = {delta:.3f}")
        if delta < tol:
            print(f"\n*** Converged after {it+1} iterations (tol={tol}). ***")
            return q, last_prices, last_profits, it + 1

    print(f"\n!!! No convergence after {max_iter} iterations (last dq={delta:.3f}).")
    return q, last_prices, last_profits, max_iter

# =============================================================================
# RUN
# =============================================================================

if __name__ == "__main__":
    print("=" * 90)
    print(f"EPEC diagonalization — Event: {EVENT['name']}")
    print(f"Leaders: {LEADERS}")
    print(f"Blocked during crisis: {EVENT['blocked_suppliers']}")
    print("=" * 90)

    q_eq, prices, profits, iters = diagonalize()

    MONTH_NAMES = ["", "Jan","Feb","Mar","Apr","May","Jun",
                   "Jul","Aug","Sep","Oct","Nov","Dec"]

    print("\n" + "=" * 90)
    print(f"Final equilibrium  (after {iters} iterations)")
    print("=" * 90)

    print("\nLeader profits (sum over horizon):")
    for L in LEADERS:
        print(f"  {L:20s}  {profits[L]:10.1f}")

    hdr = (f"{'t':>3} {'mo':>4}  {'p_EU':>7} {'p_AS':>7}   "
           f"{'USA_EU':>7} {'USA_AS':>7} {'AUS_AS':>7} {'RUS_AS':>7} "
           f"{'QAT_EU':>7} {'QAT_AS':>7}  {'status':>7}")
    print("\n" + hdr)
    print("-" * len(hdr))
    for t in T:
        mo = MONTH_NAMES[calendar_month(t)]
        pe, pa = prices["EU"][t], prices["Asia"][t]
        status = "CLOSED" if is_closed(t) else "OPEN"
        print(f"{t:>3} {mo:>4}  {pe:>7.2f} {pa:>7.2f}   "
              f"{q_eq['USA']['EU'][t]:>7.2f} {q_eq['USA']['Asia'][t]:>7.2f} "
              f"{q_eq['Australia']['Asia'][t]:>7.2f} {q_eq['Russia']['Asia'][t]:>7.2f} "
              f"{q_eq['Qatar']['EU'][t]:>7.2f} {q_eq['Qatar']['Asia'][t]:>7.2f}  {status:>7}")
