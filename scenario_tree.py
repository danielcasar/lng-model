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
    ESCALATION_HAZARD_BY_K, ESCALATION_PERSIST, BRANCH_DEPTH,
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
                    closed_at_start=0, branch_depth=BRANCH_DEPTH, verbose=False):
    """Build a PROPER multi-stage BRANCHING scenario tree from t_start.

    From every node each reachable next state is instantiated, weighted by the
    agent's belief:
        OPEN      -> {OPEN (1-p_C),  CLOSED (p_C)}
        CLOSED    -> {OPEN (p_R), CLOSED (1-p_R-p_esc), ESCALATED (p_esc)}
        ESCALATED -> {ESCALATED (1.0)}            (absorbing deeper-disruption)
    p_C, p_R are Beta-Bernoulli posterior means (conjugate-updated per branch);
    p_esc is the exogenous duration-dependent structural hazard.

    The full belief tree is built for the first `branch_depth` months from the
    root (the decision-relevant window -- enough to span the realized closure
    from any crisis re-solve); beyond that each node continues along its single
    MODAL (most-likely) successor to T_LAST, a deterministic tail that anchors
    the terminal-storage condition. This keeps the path-dependent-belief tree
    tractable while removing the old caterpillar's foresight: every reopening
    date inside the window is weighed at its true belief probability, none
    privileged (the agent cannot "know" the realized reopening month).

    closed_at_start seeds months_closed (consecutive closed months already
    elapsed up to & incl. t_start) so the duration-dependent escalation hazard
    and the realized reroute derate see the TRUE elapsed closure.
    """
    nodes = {}
    seq = [0]
    def _new_id():
        seq[0] += 1
        return f"n{seq[0]:05d}"

    root = TreeNode(
        node_id="root", t=t_start, closure_open=open_start,
        history=((t_start, open_start),), parent_id="", cum_prob=1.0,
        alpha_C=alpha_C, beta_C=beta_C, alpha_R=alpha_R, beta_R=beta_R,
        children=[], escalated=False,
        months_closed=(0 if open_start else max(1, closed_at_start)),
    )
    nodes["root"] = root

    def add_child(parent, closure_open, escalated, edge_prob, counts):
        a_C, b_C, a_R, b_R = counts
        t = parent.t + 1
        nid = _new_id()
        child = TreeNode(
            node_id=nid, t=t, closure_open=closure_open,
            history=parent.history + ((t, closure_open),),
            parent_id=parent.node_id, cum_prob=parent.cum_prob * edge_prob,
            alpha_C=a_C, beta_C=b_C, alpha_R=a_R, beta_R=b_R,
            children=[], escalated=escalated,
            months_closed=(0 if closure_open else parent.months_closed + 1),
        )
        nodes[nid] = child
        parent.children.append(nid)
        return child

    def successors(node):
        """(closure_open, escalated, prob, counts) for each child of `node`."""
        if node.escalated:                                    # absorbing
            return [(False, True, 1.0,
                     (node.alpha_C, node.beta_C, node.alpha_R, node.beta_R))]
        if node.closure_open:
            p_c = beta_mean(node.alpha_C, node.beta_C)
            return [
                (True,  False, 1.0 - p_c,
                 (node.alpha_C, node.beta_C + 1.0, node.alpha_R, node.beta_R)),
                (False, False, p_c,
                 (node.alpha_C + 1.0, node.beta_C, node.alpha_R, node.beta_R)),
            ]
        p_r = beta_mean(node.alpha_R, node.beta_R)
        p_e = min(escalation_hazard(node.months_closed), max(0.0, 1.0 - p_r - 1e-6))
        return [
            (True,  False, p_r,
             (node.alpha_C, node.beta_C, node.alpha_R + 1.0, node.beta_R)),
            (False, False, 1.0 - p_r - p_e,
             (node.alpha_C, node.beta_C, node.alpha_R, node.beta_R + 1.0)),
            (False, True,  p_e,
             (node.alpha_C, node.beta_C, node.alpha_R, node.beta_R + 1.0)),
        ]

    def expand(node, depth):
        if node.t >= T_LAST:
            return
        opts = successors(node)
        if depth >= branch_depth and not node.escalated:
            # beyond the branching window: single MODAL deterministic successor
            best = max(opts, key=lambda o: o[2])
            opts = [(best[0], best[1], 1.0, best[3])]
        for (is_open, is_esc, prob, counts) in opts:
            if prob <= 1e-9:
                continue
            expand(add_child(node, is_open, is_esc, prob, counts), depth + 1)

    expand(root, 0)
    realized_ids = ["root"]      # no spine: the root IS the realized decision node
    if verbose:
        print(f"Built BRANCHING tree from t={t_start:+d}: {len(nodes)} nodes "
              f"(branch_depth={branch_depth})")
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
