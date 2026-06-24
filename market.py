"""
Shared LNG market structure: regions, fringe suppliers, delivered costs,
liquefaction capacities, the closure / escalation / reroute capacity derates,
the demand-response staircase, seasonality, and the scenario-tree context
builder (make_ctx).

This module is the single source of market structure for BOTH the competitive
welfare-LP core (13_competitive_rolling.py) and the strategic Stackelberg-EPEC
comparison (archived). All tunable values live in model_config.py and the
supply-side data in lng_data.py; this module contains only derived structure
and helper functions.
"""

import lng_data as ld
from scenario_tree import (
    calendar_month, T_CLOSURE_START, T_CLOSURE_END, T_LAST,
)

# ALL configuration values (with source citations) live in model_config.py.
from model_config import (
    EVENT_NAME,
    LEADERS, LEADER_REGIONS, GULF_MEMBERS, BLOCKED_LEADERS,
    GULF_RESTART_RAMP, GULF_DAMAGE_FACTOR,
    SPOT_TRADABLE, EU_ACCESS, ASIA_ACCESS, pipeline,
    demand_blocks_base,
    EU_MONTH_FACTOR, WINTER, SUMMER, ASIA_WINTER_FACTOR, ASIA_SUMMER_FACTOR,
    storage, EU_NOV_TARGET_FRAC, LNG_AVAILABILITY,
    ESCALATION_LOSS_FRAC, REROUTE_RATE_PER_MONTH, REROUTE_CAP,
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

def _months_closed(node):
    """TRUE consecutive closed months up to and including this node, as stamped
    on the node at tree-build time (scenario_tree). Carries the pre-root elapsed
    closure in rolling re-solves; falls back to the in-tree trailing count."""
    mc = getattr(node, "months_closed", None)
    if mc is not None:
        return mc
    n = 0
    for (_, open_) in reversed(node.history):
        if open_:
            break
        n += 1
    return n

def _reroute_factor(node):
    """Realized crisis-deepening derate: effective seaborne-LNG deliverability
    falls the longer the strait stays shut -- realized re-escalation
    (calibration_targets.csv) + shipping/rerouting fleet tie-up (Fulwood 2026,
    OIES). Open nodes unaffected; capped. (Disabled when rate = 0.)"""
    if node.closure_open:
        return 1.0
    return 1.0 - min(REROUTE_CAP, REROUTE_RATE_PER_MONTH * _months_closed(node))

def _lng_supply_factor(node):
    """Combined seaborne-LNG derate at a node: counterfactual escalation tail
    x realized duration-dependent reroute drag."""
    return _escalation_factor(node) * _reroute_factor(node)

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

    Blocked (Gulf) leader: zero while the strait is closed; after a reopening
    the restart is gradual (Fulwood 2026, OIES): ~50% in the first open month,
    ~91% thereafter (two damaged Ras Laffan trains offline beyond the horizon).
    Branches on which no closure ever happened keep full capacity. An escalation
    removes a further fraction of every non-blocked leader's LNG supply."""
    if leader not in BLOCKED_LEADERS:
        return _lng_supply_factor(node) * _LEADER_CAP_BASE[leader]
    if not node.closure_open:
        return 0.0

    statuses = [open_ for (_, open_) in node.history]
    if all(statuses):
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
        cap *= _lng_supply_factor(node)
    return cap

def block_size(region, block, t):
    """Size of a demand block in a region at month t (bcm). Seasonality applies
    ONLY to the essential (heating) block; the price-response rungs are flat
    across the year (calibration v6.2)."""
    size = demand_blocks_base[region][block][0]
    if block == 0:
        return size * season_factor(region, t)
    return size

def block_wtp(region, block):
    """Willingness to pay of a demand block in a region (EUR/MWh)."""
    return demand_blocks_base[region][block][1]

# =============================================================================
# REGION / SET DEFINITIONS + SCENARIO-TREE CONTEXT BUILDER
# =============================================================================

REGIONS           = ("EU", "Asia")
FRINGE_BY_REGION  = {region: list(fringe[region].keys()) for region in REGIONS}
BLOCKS_BY_REGION  = {region: list(range(len(demand_blocks_base[region]))) for region in REGIONS}
EU_NOV_MIN = EU_NOV_TARGET_FRAC * storage["EU"]["S_max"]

def make_ctx(nodes, realized_ids, s_init):
    """Bundle a scenario tree + initial storage state into the context dict
    consumed by the LP / MPCC builders. Each monthly rolling re-solve creates a
    fresh ctx."""
    return {
        "NODES":        nodes,
        "NODE_IDS":     list(nodes.keys()),
        "REALIZED_IDS": realized_ids,
        # Nov-1 storage mandate applies at every OPEN November-snapshot node
        # (calendar month 10 = end-Oct). NOT applied on closed/escalated
        # branches: during a closure the EU flexibility mechanism suspends the
        # target, and a still-closed branch physically cannot refill (mandating
        # it would make the LP infeasible).
        "NOV_NODES":    [node_id for node_id, node in nodes.items()
                         if calendar_month(node.t) == 10 and node.closure_open],
        # Terminal storage anchored at every leaf at the horizon.
        "TERMINAL_IDS": [node_id for node_id, node in nodes.items()
                         if not node.children and node.t == T_LAST],
        "S_INIT":       dict(s_init),
    }
