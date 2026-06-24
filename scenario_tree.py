"""
Scenario tree for the multi-stage stochastic EPEC model.

The single source of uncertainty is the closure-state of the Strait of Hormuz
at each month. We model this as a two-state Markov chain with state-dependent
monthly transition probabilities (p_C = monthly P(open -> closed),
p_R = monthly P(closed -> open)).

The TRANSITION RATES THEMSELVES ARE UNKNOWN to the agents. Each agent maintains
a Beta(alpha, beta) prior over each transition rate and updates it via
conjugate Bayes' rule each month:
  - In an OPEN state, observe (open -> open):     beta_C  += 1
  - In an OPEN state, observe (open -> closed):   alpha_C += 1
  - In a CLOSED state, observe (closed -> closed): beta_R += 1
  - In a CLOSED state, observe (closed -> open):   alpha_R += 1

The posterior MEAN of each rate is the branching probability used to construct
the next layer of the tree. Conditional probabilities at each decision node
sum to 1 by construction (Bayes), so the tree is a valid stochastic process.

This implementation follows the regime-switching / Bayesian-belief-updating
framework of:
  - Grenadier & Malenko (2010, JF) "A Bayesian Approach to Real Options"
  - Wachter & Zhu (2025, QE) "Learning with Rare Disasters"
  - Bouri et al. (2020, Energy Economics) for the commodity-market application
  - Hou & Nguyen (2018, Energy Economics) for the gas-market Markov-switching
    precedent.

The premium emerges endogenously from the multi-stage stochastic equilibrium
on this tree -- no Lambda*beta WTP overlay is imposed.
"""

import math
from dataclasses import dataclass, field
from scipy import stats

# Time horizon and Bayesian priors (with source citations and calibration
# rationale) live in model_config.py -- the single configuration source.
# They are re-exported here so downstream imports keep working.
from model_config import (
    T_FIRST, T_PRE_END, T_CLOSURE_START, T_CLOSURE_END,
    T_POST_START, T_LAST, CAL_OFFSET,
    ALPHA_C_PRIOR, BETA_C_PRIOR, ALPHA_R_PRIOR, BETA_R_PRIOR,
    ESCALATION_RATE_BASE, ESCALATION_RATE_SLOPE, ESCALATION_RATE_CAP,
    ESCALATION_HAZARD_BY_K, ESCALATION_PERSIST,
)

def _trailing_closed(history):
    """Number of consecutive CLOSED months at the end of a node's history
    (escalation hazard rises with this duration)."""
    n = 0
    for (_, open_) in reversed(history):
        if open_:
            break
        n += 1
    return n

def escalation_hazard(months_closed):
    """Duration-dependent structural escalation hazard p_esc(k) in the number
    of months k the strait has stayed shut. Exogenous -- NOT a Bayesian-updated
    belief (see model_config). If ESCALATION_HAZARD_BY_K is non-empty, the
    per-month calibrated values are used (linearly extrapolated beyond the
    largest specified k); otherwise the parametric min(CAP, BASE+SLOPE*(k-1))."""
    k = max(1, months_closed)
    d = ESCALATION_HAZARD_BY_K
    if d:
        if k in d:
            return d[k]
        ks = sorted(d)
        if k < ks[0]:
            return d[ks[0]]
        k1, k2 = ks[-2], ks[-1]                       # extrapolate from last two
        slope = (d[k2] - d[k1]) / (k2 - k1)
        return min(0.95, max(0.0, d[k2] + slope * (k - k2)))
    return min(ESCALATION_RATE_CAP,
               ESCALATION_RATE_BASE + ESCALATION_RATE_SLOPE * (k - 1))

def calendar_month(t):
    return ((CAL_OFFSET + t - 1) % 12) + 1

def beta_mean(alpha, beta):
    """Mean of Beta(alpha, beta)."""
    return alpha / (alpha + beta)

# =============================================================================
# SCENARIO TREE NODE
# =============================================================================

@dataclass
class TreeNode:
    """A single node in the scenario tree.

    Each node stores the posterior beliefs (alpha_C, beta_C, alpha_R, beta_R)
    that result from Bayesian-updating the prior with the observation history
    leading from root to this node. The branching probabilities to children
    are the posterior means of the relevant transition rate.
    """
    node_id:      str
    t:            int
    closure_open: bool          # True if OPEN at this t, False if CLOSED/ESCALATED
    history:      tuple         # tuple of (t, status) pairs from root to here
    parent_id:    str
    cum_prob:     float         # unconditional probability of reaching this node
    # Bayesian posterior counts AT THIS NODE (already updated with this node's obs)
    alpha_C: float
    beta_C:  float
    alpha_R: float
    beta_R:  float
    children: list = field(default_factory=list)
    escalated: bool = False     # True if this is a (deeper-disruption) ESCALATED node
    months_closed: int = 0      # TRUE consecutive closed months up to & incl. this
                                # node (carries the pre-root duration in rolling
                                # re-solves, so escalation/reroute see real elapsed
                                # closure -- not a per-solve reset)

# =============================================================================
# TREE CONSTRUCTION
# =============================================================================

def realized_status(t):
    """True if the strait is OPEN at month t on the realized path
    (closed exactly during T_CLOSURE_START..T_CLOSURE_END)."""
    return not (T_CLOSURE_START <= t <= T_CLOSURE_END)


def build_tree_from(t_start, open_start, alpha_C, beta_C, alpha_R, beta_R,
                    closed_at_start=0, verbose=False):
    """Build the scenario tree from an arbitrary starting month and belief
    state -- the workhorse for ROLLING-HORIZON re-solves.

    The root carries the posterior counts as of t_start (already updated
    with the observation at t_start). From there the tree walks the
    realized status path to T_LAST, instantiating at every step the
    counterfactual sibling (the branch where the closure state would have
    flipped) so agents' conditional expectations are well-defined.

    closed_at_start = the number of consecutive months the strait has ALREADY
    been closed up to and including t_start (>=1 if the root is closed, 0 if
    open). It seeds the per-node months_closed so the duration-dependent
    escalation hazard and the realized reroute derate see the TRUE elapsed
    closure in rolling re-solves -- without it the duration would reset to 1
    every month (the Bayesian counts persist; the duration must too).

    build_tree() (below) is the special case t_start = T_FIRST with prior
    beliefs -- the full-horizon tree used by the one-shot model.
    """
    nodes = {}
    realized_ids = []

    def _mc(history, closure_open):
        """True consecutive closed months up to & incl. this node: trailing
        closed run in-tree, plus the pre-root elapsed closure when the run
        reaches all the way back to the (closed) root."""
        if closure_open:
            return 0
        tc = _trailing_closed(history)
        return (closed_at_start + tc - 1) if tc == len(history) else tc

    def add_node(t, closure_open, parent_id, history, edge_prob,
                 a_C, b_C, a_R, b_R, escalated=False, node_id=None):
        if node_id is None:
            h_signature = "".join("O" if h[1] else "C" for h in history)
            node_id = f"t{t:+d}_{h_signature}" + ("E" if escalated else "")
        if node_id in nodes:
            return nodes[node_id]
        parent_cum = nodes[parent_id].cum_prob if parent_id else 1.0
        n = TreeNode(
            node_id=node_id, t=t, closure_open=closure_open,
            history=history, parent_id=parent_id,
            cum_prob=parent_cum * edge_prob,
            alpha_C=a_C, beta_C=b_C, alpha_R=a_R, beta_R=b_R,
            children=[], escalated=escalated,
            months_closed=_mc(history, closure_open),
        )
        nodes[node_id] = n
        if parent_id and node_id not in nodes[parent_id].children:
            nodes[parent_id].children.append(node_id)
        return n

    root_history = ((t_start, open_start),)
    root = TreeNode(
        node_id="root", t=t_start, closure_open=open_start,
        history=root_history, parent_id="", cum_prob=1.0,
        alpha_C=alpha_C, beta_C=beta_C, alpha_R=alpha_R, beta_R=beta_R,
        children=[], months_closed=_mc(root_history, open_start),
    )
    nodes["root"] = root
    realized_ids.append("root")

    prev_id = "root"
    for t in range(t_start + 1, T_LAST + 1):
        parent = nodes[prev_id]
        if parent.closure_open:
            # From OPEN: event = closure (rate p_C)
            p_event = beta_mean(parent.alpha_C, parent.beta_C)
            open_prob,  close_prob  = 1.0 - p_event, p_event
            open_counts  = (parent.alpha_C,       parent.beta_C + 1.0,
                            parent.alpha_R,       parent.beta_R)
            close_counts = (parent.alpha_C + 1.0, parent.beta_C,
                            parent.alpha_R,       parent.beta_R)
        else:
            # From CLOSED: reopening (learned rate p_R) OR escalation (exogenous
            # duration-dependent hazard, rising with months-shut). The closed
            # state retains probability 1 - p_R - p_esc.
            p_event = beta_mean(parent.alpha_R, parent.beta_R)
            p_esc   = min(escalation_hazard(parent.months_closed),
                          max(0.0, 1.0 - p_event - 1e-6))
            open_prob,  close_prob  = p_event, 1.0 - p_event - p_esc
            open_counts  = (parent.alpha_C, parent.beta_C,
                            parent.alpha_R + 1.0, parent.beta_R)
            close_counts = (parent.alpha_C, parent.beta_C,
                            parent.alpha_R,       parent.beta_R + 1.0)

        n_open  = add_node(t, True,  prev_id, parent.history + ((t, True),),
                           open_prob,  *open_counts)
        n_close = add_node(t, False, prev_id, parent.history + ((t, False),),
                           close_prob, *close_counts)
        # Escalation branch: a counterfactual deeper-disruption hung off every
        # CLOSED node (the downside the closed state otherwise lacks). With
        # ESCALATION_PERSIST it is an ABSORBING multi-month state -- the
        # disruption strikes at month t and persists (supply depressed) to the
        # horizon, so gas carried into the crisis pays off repeatedly in the
        # bad branch (precautionary-refill incentive). Otherwise a 1-month leaf.
        if not parent.closure_open and p_esc > 0.0:
            if ESCALATION_PERSIST:
                chain_parent, chain_hist, edge = prev_id, parent.history, p_esc
                for tt in range(t, T_LAST + 1):
                    chain_hist = chain_hist + ((tt, False),)
                    n_esc = add_node(tt, False, chain_parent, chain_hist,
                                     edge, *close_counts, escalated=True,
                                     node_id=f"esc{t:+d}_t{tt:+d}")
                    chain_parent, edge = n_esc.node_id, 1.0   # absorbing
            else:
                add_node(t, False, prev_id, parent.history + ((t, False),),
                         p_esc, *close_counts, escalated=True)

        nxt = n_open if realized_status(t) else n_close
        prev_id = nxt.node_id
        realized_ids.append(prev_id)

    if verbose:
        print(f"Built tree from t={t_start:+d} with {len(nodes)} nodes, "
              f"{len(realized_ids)} on realized path")
        print(f"Realized cumulative probability: {nodes[realized_ids[-1]].cum_prob:.6g}")

    return nodes, realized_ids


def build_tree(verbose=False):
    """Full-horizon tree from T_FIRST with prior beliefs (one-shot model)."""
    return build_tree_from(T_FIRST, True,
                           ALPHA_C_PRIOR, BETA_C_PRIOR,
                           ALPHA_R_PRIOR, BETA_R_PRIOR, verbose=verbose)

# =============================================================================
# DIAGNOSTICS
# =============================================================================

def report_tree(nodes, realized_ids):
    """Print a human-readable summary of the realized path with posterior means."""
    months = ['','Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec']
    print(f"{'='*90}")
    print("Scenario tree -- realized path with Beta-Bernoulli posteriors")
    print(f"{'='*90}")
    print(f"{'t':>3} {'mo':>4} {'status':>8} "
          f"{'E[p_C]':>9} {'E[p_R]':>9} {'P(reach)':>11}")
    print('-' * 90)
    for nid in realized_ids:
        n = nodes[nid]
        ep_c = beta_mean(n.alpha_C, n.beta_C)
        ep_r = beta_mean(n.alpha_R, n.beta_R)
        status = "OPEN" if n.closure_open else "CLOSED"
        print(f"{n.t:>+3d} {months[calendar_month(n.t)]:>4} {status:>8} "
              f"{ep_c:>9.5f} {ep_r:>9.4f} {n.cum_prob:>11.4g}")

    # Compare pre-shock vs post-closure posterior means
    pre_node  = nodes[realized_ids[T_PRE_END - T_FIRST]]      # t = 0
    post_node = nodes[realized_ids[-1]]                       # t = +12
    print()
    print(f"Pre-shock posterior E[p_C] at t=0:       {beta_mean(pre_node.alpha_C, pre_node.beta_C):.6f}")
    print(f"Post-closure posterior E[p_C] at t=+12:  {beta_mean(post_node.alpha_C, post_node.beta_C):.6f}")
    print(f"Ratio (post / pre):                       {beta_mean(post_node.alpha_C, post_node.beta_C) / beta_mean(pre_node.alpha_C, pre_node.beta_C):.2f}x")

# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    nodes, realized_ids = build_tree(verbose=True)
    report_tree(nodes, realized_ids)
