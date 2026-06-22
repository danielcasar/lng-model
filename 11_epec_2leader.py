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
  - Each leader solves its own MPCC with the other leader's supply
    held as a fixed parameter (read from the previous iteration).
  - Per iteration: solve USA's MPCC, then Gulf's MPCC, then apply a
    damped update: new supply = alpha * best response + (1-alpha) * old.
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
    GULF_RESTART_RAMP, GULF_DAMAGE_FACTOR,
    SPOT_TRADABLE, EU_ACCESS, ASIA_ACCESS, pipeline,
    demand_blocks_base,
    EU_MONTH_FACTOR, WINTER, SUMMER, ASIA_WINTER_FACTOR, ASIA_SUMMER_FACTOR,
    HOLDING_COST, storage, STORAGE_FLOOR_FRAC,
    EU_NOV_TARGET_FRAC, EU_NOV_TARGET_FRAC_2026,
    NOV_2026_T, STORAGE_TARGETS_EU, LNG_AVAILABILITY,
    ESCALATION_LOSS_FRAC,
    EU_MAX_INJECT_BCM, EU_MAX_WITHDRAW_BCM,
    M_FRINGE, M_DEMAND, M_PRICE, M_KKT, M_STORAGE,
)

# =============================================================================
# DERIVED MARKET STRUCTURE (from model_config + lng_data)
# =============================================================================

EVENT = ld.EVENTS[EVENT_NAME]

# Access share x spot-tradability x availability (maintenance/feedgas
# derating of LNG plants; pipelines and non-LNG aggregates unaffected)
EU_FRINGE_share   = {e: SPOT_TRADABLE * LNG_AVAILABILITY * supplier
                     for e, supplier in EU_ACCESS.items()}
ASIA_FRINGE_share = {e: SPOT_TRADABLE * LNG_AVAILABILITY * supplier
                     for e, supplier in ASIA_ACCESS.items()}

lng_EU_fringe   = ld.regional_supply("EU",   list(EU_FRINGE_share),   EU_FRINGE_share,
                                     blocked_suppliers=EVENT["blocked_suppliers"])
lng_Asia_fringe = ld.regional_supply("Asia", list(ASIA_FRINGE_share), ASIA_FRINGE_share,
                                     blocked_suppliers=EVENT["blocked_suppliers"])
fringe = {
    "EU":   {**pipeline["EU"],   **lng_EU_fringe},
    "Asia": {**pipeline["Asia"], **lng_Asia_fringe},
}

# Pipeline / non-LNG-aggregate fringe keys per region: these are NOT affected
# by an escalation (which removes seaborne LNG, not pipeline or domestic gas).
PIPELINE_KEYS = {r: set(pipeline[r].keys()) for r in ("EU", "Asia")}

def _escalation_factor(node):
    """Extra LNG-supply derating applied at ESCALATED nodes (1.0 elsewhere)."""
    return (1.0 - ESCALATION_LOSS_FRAC) if getattr(node, "escalated", False) else 1.0

# =============================================================================
# LEADER COSTS AND CAPACITIES
# =============================================================================

# USA: costs and capacity straight from the data. Gulf composite: capacity
# is Qatar + UAE (Other_Middle_East); delivered cost uses Qatar's BEP and
# transport costs -- the capacity-weighted BEP differs by < $0.03/MMBtu
# (UAE share ~4% of the composite), so Qatar's cost is used for both members.
leader_cost = {
    "USA":  {region: ld.delivered_cost_eur_mwh("USA",   region) for region in LEADER_REGIONS["USA"]},
    "Gulf": {region: ld.delivered_cost_eur_mwh("Qatar", region) for region in LEADER_REGIONS["Gulf"]},
}
_LEADER_CAP_BASE = {
    "USA":  LNG_AVAILABILITY
            * ld.annual_bn_mmbtu_to_monthly_bcm(ld.LIQ_CAP_BN_MMBTU_YR["USA"]),
    "Gulf": LNG_AVAILABILITY
            * sum(ld.annual_bn_mmbtu_to_monthly_bcm(ld.LIQ_CAP_BN_MMBTU_YR[member])
                  for member in GULF_MEMBERS),
}

def leader_cap_at_node(leader, node):
    """Monthly capacity of a leader at a scenario-tree node.

    Blocked (Gulf) leader: zero while the strait is closed; after a
    reopening the restart is gradual (Fulwood 2026, OIES): ~50% in the
    first open month, ~91% thereafter (two damaged Ras Laffan trains
    offline beyond the horizon). Branches on which no closure ever
    happened keep full capacity. An escalation removes a further
    fraction of every non-blocked leader's LNG supply."""
    if leader not in BLOCKED_LEADERS:
        return _escalation_factor(node) * _LEADER_CAP_BASE[leader]
    if not node.closure_open:
        return 0.0

    statuses = [open_ for (_, open_) in node.history]
    if all(statuses):
        # No closure observed on this branch. For rolling-horizon subtrees
        # rooted after the realized closure, the closure lies before the
        # subtree root: recover it from the realized calendar.
        if node.t > T_CLOSURE_END and node.history[0][0] > T_CLOSURE_START:
            months_open = node.t - T_CLOSURE_END
        else:
            return _LEADER_CAP_BASE[leader]      # genuinely never closed
    else:
        months_open = 0                          # consecutive open months
        for open_ in reversed(statuses):         # since the last closure
            if not open_:
                break
            months_open += 1

    factor = GULF_RESTART_RAMP if months_open == 1 else GULF_DAMAGE_FACTOR
    return factor * _LEADER_CAP_BASE[leader]

# =============================================================================
# DEMAND / SEASONALITY HELPERS (values in model_config.py)
# =============================================================================

def season_factor(region, t):
    month = calendar_month(t)
    if region == "EU":
        return EU_MONTH_FACTOR[month]
    else:
        if month in WINTER: return ASIA_WINTER_FACTOR
        if month in SUMMER: return ASIA_SUMMER_FACTOR
        return 1.00

def fringe_cost(region, supplier):
    """Delivered cost of a fringe supplier into a region (EUR/MWh)."""
    return fringe[region][supplier]["cost"]

def fringe_capacity(region, supplier, node):
    """Monthly capacity of a fringe supplier into a region at a tree node
    (zero for Hormuz-blocked suppliers while the strait is closed; LNG
    suppliers further derated at escalated nodes, pipelines unaffected)."""
    cap = fringe[region][supplier]["cap_closed" if not node.closure_open else "cap_open"]
    if supplier not in PIPELINE_KEYS[region]:
        cap *= _escalation_factor(node)
    return cap

def block_size(region, block, t):
    """Size of a demand block in a region at month t (bcm).

    Seasonality applies ONLY to the essential (heating) block; the
    price-response rungs (industry, power fuel-switching, storage-refill
    competition) are flat across the year (calibration v6.2). Scaling the
    whole staircase made the crisis summer both small AND cheap, which
    collapsed expected prices and triggered storage dumping into the
    closure months. With heating-only scaling the staircase reproduces
    the observed totals exactly: EU winter 24*1.41+16 = 49.8 bcm (obs Jan
    49), crisis summer 24*0.63+16 = 31.1 bcm (obs Apr-May 30-31)."""
    size = demand_blocks_base[region][block][0]
    if block == 0:
        return size * season_factor(region, t)
    return size

def block_wtp(region, block):
    """Willingness to pay of a demand block in a region (EUR/MWh)."""
    return demand_blocks_base[region][block][1]

# =============================================================================
# BUILD SCENARIO TREE (shared across both leaders)
# =============================================================================

REGIONS           = ("EU", "Asia")
FRINGE_BY_REGION  = {region: list(fringe[region].keys()) for region in REGIONS}
BLOCKS_BY_REGION  = {region: list(range(len(demand_blocks_base[region]))) for region in REGIONS}
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
        "NOV_NODES":    [node_id for node_id in realized_ids
                         if calendar_month(nodes[node_id].t) == 10],
        "TERMINAL_IDS": [node_id for node_id, node in nodes.items()
                         if not node.children and node.t == T_LAST],
        "S_INIT":       dict(s_init),
    }

_NODES_FULL, _REALIZED_FULL = build_tree()
DEFAULT_CTX = make_ctx(_NODES_FULL, _REALIZED_FULL,
                       {region: storage[region]["S_init"] for region in REGIONS})

# Backwards-compatible module-level aliases (used by __main__ prints)
NODES        = DEFAULT_CTX["NODES"]
NODE_IDS     = DEFAULT_CTX["NODE_IDS"]
REALIZED_IDS = DEFAULT_CTX["REALIZED_IDS"]

# =============================================================================
# PER-LEADER MPCC BUILDER (one of two)
# =============================================================================

def build_leader_mpcc(leader, others_q, ctx=None):
    """Build one leader's MPCC with the other leader's quantities held fixed.

    others_q[region][node_id] = the OTHER leader's supply per region and node.
    ctx = tree context from make_ctx(); defaults to the full-horizon tree.
    """
    ctx = ctx if ctx is not None else DEFAULT_CTX
    NODES                = ctx["NODES"]
    NODE_IDS             = ctx["NODE_IDS"]
    NOV_TARGETS_EU_NODES = ctx["NOV_NODES"]
    TERMINAL_NODE_IDS    = ctx["TERMINAL_IDS"]
    S_INIT               = ctx["S_INIT"]

    accessible = LEADER_REGIONS[leader]

    model = pyo.ConcreteModel(f"MPCC_{leader}_stochastic")

    model.R  = pyo.Set(initialize=list(REGIONS))                # regions
    model.N  = pyo.Set(initialize=NODE_IDS)                     # scenario-tree nodes
    model.RS = pyo.Set(initialize=[(region, supplier) for region in REGIONS      # (region, fringe supplier)
                               for supplier in FRINGE_BY_REGION[region]], dimen=2)
    model.RK = pyo.Set(initialize=[(region, block) for region in REGIONS      # (region, demand block)
                               for block in BLOCKS_BY_REGION[region]], dimen=2)

    # Leader's decision: supply per region and node (0 if region not accessible)
    def _supply_bounds(model, region, node_id):
        if region in accessible: return (0, None)
        return (0, 0)
    model.leader_supply = pyo.Var(model.R, model.N, domain=pyo.NonNegativeReals,
                              bounds=_supply_bounds)

    # Followers' primal variables (per node)
    model.fringe_supply = pyo.Var(model.RS, model.N, domain=pyo.NonNegativeReals)  # bcm from each price-taking supplier
    model.demand_served = pyo.Var(model.RK, model.N, domain=pyo.NonNegativeReals)  # bcm delivered to each demand block
    model.storage_level = pyo.Var(model.R,  model.N, domain=pyo.NonNegativeReals)  # end-of-month stock (bcm)
    model.storage_flow  = pyo.Var(model.R,  model.N, domain=pyo.Reals)             # injection (+) / withdrawal (-)

    # Followers' dual variables
    model.price           = pyo.Var(model.R,  model.N, domain=pyo.NonNegativeReals)  # market-clearing price (EUR/MWh)
    model.fringe_cap_rent = pyo.Var(model.RS, model.N, domain=pyo.NonNegativeReals)  # scarcity rent on fringe capacity
    model.block_cap_rent  = pyo.Var(model.RK, model.N, domain=pyo.NonNegativeReals)  # rent on demand-block saturation
    model.storage_cap_rent= pyo.Var(model.R,  model.N, domain=pyo.NonNegativeReals)  # rent on full storage

    # Binary switches for the Fortuny-Amat Big-M complementarity pairs
    model.is_fringe_active = pyo.Var(model.RS, model.N, domain=pyo.Binary)  # fringe supply > 0
    model.is_block_served  = pyo.Var(model.RK, model.N, domain=pyo.Binary)  # block demand > 0
    model.is_fringe_at_cap = pyo.Var(model.RS, model.N, domain=pyo.Binary)  # fringe capacity binding
    model.is_block_at_cap  = pyo.Var(model.RK, model.N, domain=pyo.Binary)  # block fully served
    model.is_market_tight  = pyo.Var(model.R,  model.N, domain=pyo.Binary)  # market balance binding (price > 0)
    model.is_storage_held  = pyo.Var(model.R,  model.N, domain=pyo.Binary)  # stock > 0 (Euler condition binding)
    model.is_storage_full  = pyo.Var(model.R,  model.N, domain=pyo.Binary)  # storage at S_max

    # Leader capacity per node
    def _leader_capacity(model, node_id):
        return (sum(model.leader_supply[region, node_id] for region in accessible)
                <= leader_cap_at_node(leader, NODES[node_id]))
    model.leader_capacity = pyo.Constraint(model.N, rule=_leader_capacity)

    # Per-leader delivery floor: share of capacity that is NOT strategically
    # withholdable (see CONTRACT_FLOOR in model_config.py). Binds total
    # dispatch, so cross-basin arbitrage stays strategic; zero when blocked.
    def _leader_floor(model, node_id):
        return (sum(model.leader_supply[region, node_id] for region in accessible)
                >= CONTRACT_FLOOR[leader] * leader_cap_at_node(leader, NODES[node_id]))
    model.leader_floor = pyo.Constraint(model.N, rule=_leader_floor)

    # Market balance per node:
    #   demand served + storage injection <= own supply + other leader's
    #   supply + fringe supply
    def _market_balance(model, region, node_id):
        return (sum(model.demand_served[region, block, node_id] for block in BLOCKS_BY_REGION[region])
                + model.storage_flow[region, node_id]
                <= model.leader_supply[region, node_id] + others_q[region].get(node_id, 0.0)
                   + sum(model.fringe_supply[region, supplier, node_id] for supplier in FRINGE_BY_REGION[region]))
    model.market_balance = pyo.Constraint(model.R, model.N, rule=_market_balance)

    model.fringe_cap = pyo.Constraint(model.RS, model.N,
        rule=lambda model, region, supplier, node_id:
            model.fringe_supply[region, supplier, node_id] <= fringe_capacity(region, supplier, NODES[node_id]))
    model.block_cap = pyo.Constraint(model.RK, model.N,
        rule=lambda model, region, block, node_id:
            model.demand_served[region, block, node_id] <= block_size(region, block, NODES[node_id].t))

    def _storage_balance(model, region, node_id):
        node = NODES[node_id]
        if node.parent_id == "":
            prev = S_INIT[region]
        else:
            prev = model.storage_level[region, node.parent_id]
        return model.storage_level[region, node_id] == prev + model.storage_flow[region, node_id]
    model.storage_balance = pyo.Constraint(model.R, model.N, rule=_storage_balance)

    model.storage_cap = pyo.Constraint(model.R, model.N,
        rule=lambda model, region, node_id:
            model.storage_level[region, node_id] <= storage[region]["S_max"])

    # Minimum operational storage floor (precautionary cushion, see config)
    model.storage_floor = pyo.Constraint(model.R, model.N,
        rule=lambda model, region, node_id:
            model.storage_level[region, node_id]
            >= STORAGE_FLOOR_FRAC * storage[region]["S_max"])

    # Physical deliverability limits (EU only): injection and withdrawal
    # rates bounded by GIE aggregate technical capacity.
    def _inject_limit(model, region, node_id):
        if region != "EU": return pyo.Constraint.Skip
        return model.storage_flow[region, node_id] <= EU_MAX_INJECT_BCM
    model.inject_limit = pyo.Constraint(model.R, model.N, rule=_inject_limit)
    def _withdraw_limit(model, region, node_id):
        if region != "EU": return pyo.Constraint.Skip
        return model.storage_flow[region, node_id] >= -EU_MAX_WITHDRAW_BCM
    model.withdraw_limit = pyo.Constraint(model.R, model.N, rule=_withdraw_limit)

    def _terminal_storage(model, region, node_id):
        if node_id not in TERMINAL_NODE_IDS: return pyo.Constraint.Skip
        return model.storage_level[region, node_id] == storage[region]["S_term"]
    model.terminal_storage = pyo.Constraint(model.R, model.N, rule=_terminal_storage)

    # 1-Nov filling target: 90% normally; 80% for the crisis-year November
    # (EU flexibility mechanism; Fulwood 2026 projects 76-81 bcm on
    # 1 Nov 2026 -- the 90% target is unattainable that year).
    def _nov_mandate(model, node_id):
        if node_id not in NOV_TARGETS_EU_NODES: return pyo.Constraint.Skip
        frac = (EU_NOV_TARGET_FRAC_2026 if NODES[node_id].t == NOV_2026_T
                else EU_NOV_TARGET_FRAC)
        return model.storage_level["EU", node_id] >= frac * storage["EU"]["S_max"]
    model.nov_mandate = pyo.Constraint(model.N, rule=_nov_mandate)

    # KKT stationarity conditions of the followers' market-clearing problem:
    #   fringe:  delivered cost + capacity rent >= price
    #            (equality whenever the supplier dispatches)
    #   demand:  price + saturation rent >= willingness to pay
    #            (equality whenever the block consumes)
    model.kkt_fringe = pyo.Constraint(model.RS, model.N,
        rule=lambda model, region, supplier, node_id:
            fringe_cost(region, supplier) + model.fringe_cap_rent[region, supplier, node_id]
            - model.price[region, node_id] >= 0)
    model.kkt_demand = pyo.Constraint(model.RK, model.N,
        rule=lambda model, region, block, node_id:
            model.price[region, node_id] + model.block_cap_rent[region, block, node_id]
            - block_wtp(region, block) >= 0)

    # Storage Euler condition on tree edges:
    #   price today >= E[price next month] - storage-cap rent - holding cost
    #   (equality whenever stock is held, i.e. intertemporal arbitrage)
    def _expected_next_price(model, region, node):
        total_child_prob = sum(NODES[child].cum_prob / node.cum_prob for child in node.children) \
                           if node.cum_prob > 0 else 1.0
        return sum((NODES[child].cum_prob / node.cum_prob if node.cum_prob > 0 else 0)
                   * model.price[region, child] for child in node.children) / max(total_child_prob, 1e-9)

    def _kkt_storage(model, region, node_id):
        node = NODES[node_id]
        if not node.children: return pyo.Constraint.Skip
        return (model.price[region, node_id] - _expected_next_price(model, region, node)
                + model.storage_cap_rent[region, node_id] + HOLDING_COST >= 0)
    model.kkt_storage = pyo.Constraint(model.R, model.N, rule=_kkt_storage)

    # Fortuny-Amat Big-M complementarity: each KKT inequality pairs with its
    # primal slack -- at most one of the two may be strictly positive, which
    # the binary switch enforces.

    # Fringe dispatches only if price covers cost (and vice versa)
    model.compl_fringe_a = pyo.Constraint(model.RS, model.N,
        rule=lambda model, region, supplier, node_id:
            model.fringe_supply[region, supplier, node_id]
            <= M_FRINGE * model.is_fringe_active[region, supplier, node_id])
    model.compl_fringe_b = pyo.Constraint(model.RS, model.N,
        rule=lambda model, region, supplier, node_id:
            fringe_cost(region, supplier) + model.fringe_cap_rent[region, supplier, node_id] - model.price[region, node_id]
            <= M_KKT * (1 - model.is_fringe_active[region, supplier, node_id]))

    # A block consumes only if its WTP covers the price (and vice versa)
    model.compl_demand_a = pyo.Constraint(model.RK, model.N,
        rule=lambda model, region, block, node_id:
            model.demand_served[region, block, node_id]
            <= M_DEMAND * model.is_block_served[region, block, node_id])
    model.compl_demand_b = pyo.Constraint(model.RK, model.N,
        rule=lambda model, region, block, node_id:
            model.price[region, node_id] + model.block_cap_rent[region, block, node_id] - block_wtp(region, block)
            <= M_KKT * (1 - model.is_block_served[region, block, node_id]))

    # Fringe capacity rent flows only when capacity is exhausted
    model.compl_fringe_cap_a = pyo.Constraint(model.RS, model.N,
        rule=lambda model, region, supplier, node_id:
            model.fringe_cap_rent[region, supplier, node_id]
            <= M_PRICE * model.is_fringe_at_cap[region, supplier, node_id])
    model.compl_fringe_cap_b = pyo.Constraint(model.RS, model.N,
        rule=lambda model, region, supplier, node_id:
            fringe_capacity(region, supplier, NODES[node_id]) - model.fringe_supply[region, supplier, node_id]
            <= M_FRINGE * (1 - model.is_fringe_at_cap[region, supplier, node_id]))

    # Block saturation rent flows only when the block is fully served
    model.compl_block_cap_a = pyo.Constraint(model.RK, model.N,
        rule=lambda model, region, block, node_id:
            model.block_cap_rent[region, block, node_id]
            <= M_PRICE * model.is_block_at_cap[region, block, node_id])
    model.compl_block_cap_b = pyo.Constraint(model.RK, model.N,
        rule=lambda model, region, block, node_id:
            block_size(region, block, NODES[node_id].t) - model.demand_served[region, block, node_id]
            <= M_DEMAND * (1 - model.is_block_at_cap[region, block, node_id]))

    # Positive price only when the market balance is binding (no spare supply)
    def _compl_price_b(model, region, node_id):
        node = NODES[node_id]
        spare_supply = (model.leader_supply[region, node_id] + others_q[region].get(node_id, 0.0)
                        + sum(model.fringe_supply[region, supplier, node_id] for supplier in FRINGE_BY_REGION[region])
                        - sum(model.demand_served[region, block, node_id] for block in BLOCKS_BY_REGION[region])
                        - model.storage_flow[region, node_id])
        big = (leader_cap_at_node(leader, node) + others_q[region].get(node_id, 0.0)
               + sum(fringe_capacity(region, supplier, node) for supplier in FRINGE_BY_REGION[region])
               + M_STORAGE + 10.0)
        return spare_supply <= big * (1 - model.is_market_tight[region, node_id])
    model.compl_price_a = pyo.Constraint(model.R, model.N,
        rule=lambda model, region, node_id:
            model.price[region, node_id] <= M_PRICE * model.is_market_tight[region, node_id])
    model.compl_price_b = pyo.Constraint(model.R, model.N, rule=_compl_price_b)

    # Stock is held only when the Euler condition holds with equality
    def _compl_storage_a(model, region, node_id):
        if not NODES[node_id].children: return pyo.Constraint.Skip
        return model.storage_level[region, node_id] <= M_STORAGE * model.is_storage_held[region, node_id]
    model.compl_storage_a = pyo.Constraint(model.R, model.N, rule=_compl_storage_a)

    def _compl_storage_b(model, region, node_id):
        node = NODES[node_id]
        if not node.children: return pyo.Constraint.Skip
        return (model.price[region, node_id] - _expected_next_price(model, region, node)
                + model.storage_cap_rent[region, node_id] + HOLDING_COST
                <= M_KKT * (1 - model.is_storage_held[region, node_id]))
    model.compl_storage_b = pyo.Constraint(model.R, model.N, rule=_compl_storage_b)

    # Storage-cap rent flows only when storage is full
    model.compl_full_a = pyo.Constraint(model.R, model.N,
        rule=lambda model, region, node_id:
            model.storage_cap_rent[region, node_id] <= M_PRICE * model.is_storage_full[region, node_id])
    model.compl_full_b = pyo.Constraint(model.R, model.N,
        rule=lambda model, region, node_id:
            storage[region]["S_max"] - model.storage_level[region, node_id]
            <= M_STORAGE * (1 - model.is_storage_full[region, node_id]))

    # Leader's objective: expected probability-weighted profit
    #   sum over nodes of P(node) x (price - delivered cost) x own supply
    model.obj = pyo.Objective(
        sense=pyo.maximize,
        expr=sum(NODES[node_id].cum_prob
                 * (model.price[region, node_id] - leader_cost[leader][region]) * model.leader_supply[region, node_id]
                 for region in accessible for node_id in NODE_IDS))

    return model

# =============================================================================
# DIAGONALIZATION
# =============================================================================

def init_quantities(ctx=None):
    ctx = ctx if ctx is not None else DEFAULT_CTX
    NODES, NODE_IDS = ctx["NODES"], ctx["NODE_IDS"]
    quantities = {}
    for leader in LEADERS:
        regs = LEADER_REGIONS[leader]
        quantities[leader] = {region: {node_id: 0.0 for node_id in NODE_IDS} for region in REGIONS}
        for node_id in NODE_IDS:
            node = NODES[node_id]
            cap = leader_cap_at_node(leader, node)
            share = 0.5 * cap / max(1, len(regs))
            for region in regs:
                quantities[leader][region][node_id] = share
    return quantities

def solve_leader(leader, others_q, ctx=None, time_limit=180, mip_gap=3e-2):
    # During diagonalization iterations a loose 3% gap is sufficient: the
    # damped update only uses the best response directionally, and the
    # equilibrium is refined across iterations anyway. The final storage-
    # extraction solve uses a tighter gap (see __main__). With the 17-block
    # calibrated demand staircase the MIQCP has ~2,000 demand-side binaries
    # (3x the coarse grid), so per-solve effort is materially higher.
    ctx = ctx if ctx is not None else DEFAULT_CTX
    NODE_IDS = ctx["NODE_IDS"]
    model = build_leader_mpcc(leader, others_q, ctx=ctx)
    solver = pyo.SolverFactory("gurobi")
    solver.options["NonConvex"]  = 2
    solver.options["MIPGap"]     = mip_gap
    solver.options["TimeLimit"]  = time_limit
    solver.options["OutputFlag"] = 0
    # Prioritise FINDING feasible incumbents over proving optimality
    # (Gurobi MIPFocus=1): the diagonalization only needs a good incumbent
    # per iteration, and with the tightened Big-Ms (calibration v5) the
    # time-limited solves otherwise sometimes terminate with no incumbent
    # at all ("SOLVER FAILED" in the rolling log).
    solver.options["MIPFocus"]   = 1
    results = solver.solve(model, tee=False, load_solutions=False)
    try:
        model.solutions.load_from(results)
    except Exception:
        return None, None, None, None
    try:
        q_new  = {region: {node_id: max(0.0, pyo.value(model.leader_supply[region, node_id]))
                      for node_id in NODE_IDS} for region in REGIONS}
        prices = {region: {node_id: pyo.value(model.price[region, node_id])
                      for node_id in NODE_IDS} for region in REGIONS}
        stocks = {region: {node_id: pyo.value(model.storage_level[region, node_id])
                      for node_id in NODE_IDS} for region in REGIONS}
        profit = pyo.value(model.obj)
    except Exception:
        return None, None, None, None
    return q_new, prices, profit, stocks

def max_change(q_old, q_new, ctx=None):
    ctx = ctx if ctx is not None else DEFAULT_CTX
    NODE_IDS = ctx["NODE_IDS"]
    return max(abs(q_old[leader][region][node_id] - q_new[leader][region][node_id])
               for leader in LEADERS for region in REGIONS for node_id in NODE_IDS)

def diagonalize(ctx=None, max_iter=8, tol=0.5, alpha=0.4,
                time_limit=180, verbose=True, q_init=None):
    ctx = ctx if ctx is not None else DEFAULT_CTX
    NODE_IDS, REALIZED_IDS = ctx["NODE_IDS"], ctx["REALIZED_IDS"]
    # Warm start: the rolling driver passes the previous month's equilibrium
    # (mapped onto the new subtree) -- the equilibrium changes only
    # incrementally between months, so this starts the Gauss-Seidel loop
    # near the fixed point instead of at the crude capacity-share split.
    quantities = q_init if q_init is not None else init_quantities(ctx)
    last_prices  = None
    last_profits = {}
    last_stocks  = None
    t_start = time.time()

    for it in range(max_iter):
        t_iter = time.time()
        if verbose:
            print(f"\n--- Iteration {it+1} ---", flush=True)
        q_prev = {leader: {region: dict(quantities[leader][region]) for region in REGIONS} for leader in LEADERS}

        for leader in LEADERS:
            t_solve = time.time()
            other_leader = [cand for cand in LEADERS if cand != leader][0]
            others = {region: quantities[other_leader][region] for region in REGIONS}
            q_br, prices, profit, stocks = solve_leader(leader, others, ctx=ctx,
                                                        time_limit=time_limit)
            solve_secs = time.time() - t_solve
            if q_br is None:
                print(f"  {leader:10s}  SOLVER FAILED, keeping previous quantities "
                      f"[{solve_secs:.0f}s]", flush=True)
                continue
            for region in REGIONS:
                for node_id in NODE_IDS:
                    quantities[leader][region][node_id] = alpha * q_br[region][node_id] + (1 - alpha) * quantities[leader][region][node_id]
            last_prices    = prices
            last_profits[leader] = profit
            last_stocks    = stocks
            if verbose:
                avg_eu = sum(quantities[leader]["EU"][node_id] for node_id in REALIZED_IDS) / len(REALIZED_IDS)
                avg_as = sum(quantities[leader]["Asia"][node_id] for node_id in REALIZED_IDS) / len(REALIZED_IDS)
                print(f"  {leader:10s}  E[profit]={profit:10.1f}  "
                      f"q_EU_realized={avg_eu:5.2f}  q_AS_realized={avg_as:5.2f}  "
                      f"[solve {solve_secs:.0f}s]", flush=True)

        delta = max_change(q_prev, quantities, ctx)
        iter_secs  = time.time() - t_iter
        total_secs = time.time() - t_start
        if verbose:
            print(f"  max |dq| = {delta:.3f}   "
                  f"[iteration {iter_secs:.0f}s, total {total_secs/60:.1f}min]", flush=True)
        if delta < tol:
            if verbose:
                print(f"\n*** Converged after {it+1} iterations (tol={tol}). "
                      f"Total wall time: {(time.time()-t_start)/60:.1f} min ***", flush=True)
            return quantities, last_prices, last_profits, it + 1, last_stocks

    if verbose:
        print(f"\n!!! No convergence after {max_iter} iterations (last dq={delta:.3f}). "
              f"Total wall time: {(time.time()-t_start)/60:.1f} min", flush=True)
    return quantities, last_prices, last_profits, max_iter, last_stocks

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
    for leader in LEADERS:
        print(f"  {leader:10s}  {profits.get(leader, float('nan')):10.1f}")

    # Re-solve one final pass of each leader's MPCC to extract storage levels
    # along the realized path (these are follower-side decisions inside the
    # leader's MPCC; we read them from the last USA solve).
    print("\nExtracting storage trajectory from final MPCC solve...", flush=True)
    final_others = {region: q_eq["Gulf"][region] for region in REGIONS}
    final_model = build_leader_mpcc("USA", final_others)
    final_solver = pyo.SolverFactory("gurobi")
    final_solver.options["NonConvex"]  = 2
    final_solver.options["MIPGap"]     = 1e-2
    final_solver.options["TimeLimit"]  = 300
    final_solver.options["OutputFlag"] = 0
    final_solver.solve(final_model, tee=False)
    s_eu = {node_id: pyo.value(final_model.storage_level["EU", node_id]) for node_id in REALIZED_IDS}

    print("\nRealized-path equilibrium prices, dispatches, and EU storage:")
    print("(Gulf = Qatar + UAE composite leader)")
    hdr = (f"{'t':>3} {'mo':>4} {'status':>8}  {'p_EU':>7} {'p_AS':>7}   "
           f"{'USA_EU':>7} {'USA_AS':>7} {'GLF_EU':>7} {'GLF_AS':>7}  {'S_EU':>6}")
    print(hdr)
    print("-" * len(hdr))
    for node_id in REALIZED_IDS:
        node = NODES[node_id]
        mo = MONTH_NAMES[calendar_month(node.t)]
        status = "OPEN" if node.closure_open else "CLOSED"
        pe = prices["EU"][node_id] if prices else float('nan')
        pa = prices["Asia"][node_id] if prices else float('nan')
        print(f"{node.t:>+3d} {mo:>4} {status:>8}  {pe:>7.2f} {pa:>7.2f}   "
              f"{q_eq['USA']['EU'][node_id]:>7.2f} {q_eq['USA']['Asia'][node_id]:>7.2f} "
              f"{q_eq['Gulf']['EU'][node_id]:>7.2f} {q_eq['Gulf']['Asia'][node_id]:>7.2f}  "
              f"{s_eu[node_id]:>6.1f}")

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
    for node_id in REALIZED_IDS:
        node = NODES[node_id]
        if node.t not in TARGETS:
            continue
        eu_obs, as_obs = TARGETS[node.t]
        eu_mod = prices["EU"][node_id] if prices else float('nan')
        as_mod = prices["Asia"][node_id] if prices else float('nan')
        mo = MONTH_NAMES[calendar_month(node.t)]
        print(f"{node.t:>+3d} {mo:>6}  {eu_mod:>7.1f} {eu_obs:>7.1f} {eu_mod-eu_obs:>+6.1f}   "
              f"{as_mod:>7.1f} {as_obs:>7.1f} {as_mod-as_obs:>+6.1f}")
        sq_err += (eu_mod - eu_obs) ** 2 + (as_mod - as_obs) ** 2
        n_obs  += 2
    rmse = (sq_err / max(n_obs, 1)) ** 0.5
    print("-" * 70)
    print(f"RMSE over {n_obs} observations: {rmse:.2f} EUR/MWh")

    for t_obs, s_obs in sorted(STORAGE_TARGETS_EU.items()):
        node_obs = next((x for x in REALIZED_IDS if NODES[x].t == t_obs), None)
        if node_obs is not None:
            print(f"Storage check t={t_obs:+d}: model S_EU={s_eu[node_obs]:.1f} bcm "
                  f"vs observed {s_obs:.1f} bcm (GIE AGSI+ / Fulwood 2026)")

    total_min = (time.time() - t_script) / 60
    print(f"\nTotal computing time: {total_min:.1f} min "
          f"({iters} diagonalization iterations + final storage extraction)", flush=True)
