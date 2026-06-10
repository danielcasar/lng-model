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

# =============================================================================
# TIME / CALENDAR
# =============================================================================

# Time indexing convention:
#   t = -5 to 0  : pre-closure baseline (6 months, Sep 2025 - Feb 2026)
#   t = +1 to +6 : closure (Mar 2026 - Aug 2026), realized 6-month duration
#   t = +7 to +24: post-closure recovery (18 months, Sep 2026 - Feb 2028)
# The post-closure horizon is extended to t = +24 to push the terminal
# storage constraint (stock at T_LAST = S_term) far enough out that it does
# not contaminate the reported results at t = +7 to +12. Headline results
# are reported only for t <= +12; nodes t = +13 to +24 exist solely to give
# the dynamic-programming structure a credible "tail" so that storage and
# leader decisions in the early post-closure period reflect genuine
# forward-looking optimisation rather than the dual of an arbitrary
# terminal equality at t = +12. This is the standard horizon-extension
# fix for terminal-condition artefacts in finite-horizon stochastic
# optimisation (Conejo, Carrion & Morales 2010, ch. 3).
T_FIRST           = -5
T_PRE_END         =  0
T_CLOSURE_START   = +1
T_CLOSURE_END     = +6
T_POST_START      = +7
T_LAST            = +24

# t = +1 -> March 2026 (month 3) -> CAL_OFFSET = 2
CAL_OFFSET = 2

def calendar_month(t):
    return ((CAL_OFFSET + t - 1) % 12) + 1

# =============================================================================
# BETA-BERNOULLI BAYESIAN PRIORS ON TRANSITION RATES
# =============================================================================
# Prior on p_C (monthly closure-arrival rate when open):
#   Calibrated to the elevated geopolitical-risk regime of 2024-26 rather
#   than the long historical chokepoint baseline. We use Beta(5, 100) with
#   prior mean 5/105 = 0.0476/month (~5% monthly probability of a major
#   closure event under the agents' prior beliefs). Effective sample size
#   ~100 months reflects the ~10-year post-2014 window of elevated
#   geopolitical risk during which agents have updated their beliefs about
#   the current regime. Consistent with Caldara & Iacoviello (2022) GPR
#   index findings of persistently elevated Middle East risk through 2025-26.
# Prior on the closure-arrival rate: Beta(2, 40), mean 2/42 = 4.8%/month.
# CALIBRATED TO THE OBSERVED PRE-CRISIS PREMIUM: TTF was flat at ~37
# EUR/MWh through Jan-Feb 2026 -- the market priced essentially no
# imminent-closure premium and was genuinely surprised on 2 March (TTF
# +55% in two trading days, Kpler). A higher prior (9%/month was tested)
# generates a winter precautionary premium far above the observed level.
# Small effective sample size (42 months) keeps the posterior responsive
# to each new observation, which matters for the crisis-period dynamics
# (ceasefire dip, re-escalation) carried by the reopening-rate posterior.
ALPHA_C_PRIOR = 2.0
BETA_C_PRIOR  = 40.0

# Prior on p_R (monthly reopening rate when closed):
#   Historical closure durations of major energy-supply chokepoint events
#   (~5-9 months for most events) suggest a typical duration of ~7 months,
#   hence prior mean reopening rate of ~0.143/month. We use Beta(2, 12)
#   with mean 2/14 = 0.143. Smaller effective sample size ~14 months
#   reflects the limited historical record of individual closure durations.
ALPHA_R_PRIOR = 2.0
BETA_R_PRIOR  = 12.0

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
    closure_open: bool          # True if OPEN at this t, False if CLOSED
    history:      tuple         # tuple of (t, status) pairs from root to here
    parent_id:    str
    cum_prob:     float         # unconditional probability of reaching this node
    # Bayesian posterior counts AT THIS NODE (already updated with this node's obs)
    alpha_C: float
    beta_C:  float
    alpha_R: float
    beta_R:  float
    children: list = field(default_factory=list)

# =============================================================================
# TREE CONSTRUCTION
# =============================================================================

def build_tree(verbose=False):
    """Build the scenario tree along the realized path.

    The realized path traverses: OPEN through pre-closure, CLOSED at t=+1
    through closure phase, OPEN at t=+7 through post-closure phase. At each
    realized-path node we also instantiate the (lower-probability) counterfactual
    child where the state would have flipped, to support the agents' expected-
    value computations.
    """
    nodes = {}
    realized_ids = []

    def add_node(t, closure_open, parent_id, history, edge_prob,
                 alpha_C, beta_C, alpha_R, beta_R):
        h_signature = "".join("O" if h[1] else "C" for h in history)
        node_id = f"t{t:+d}_{h_signature}"
        if node_id in nodes:
            return nodes[node_id]
        parent_cum = nodes[parent_id].cum_prob if parent_id else 1.0
        n = TreeNode(
            node_id=node_id, t=t, closure_open=closure_open,
            history=history, parent_id=parent_id,
            cum_prob=parent_cum * edge_prob,
            alpha_C=alpha_C, beta_C=beta_C, alpha_R=alpha_R, beta_R=beta_R,
            children=[],
        )
        nodes[node_id] = n
        if parent_id and node_id not in nodes[parent_id].children:
            nodes[parent_id].children.append(node_id)
        return n

    # -------------------------------------------------------------------------
    # Root at t = T_FIRST = -5, status OPEN, prior beliefs only.
    # -------------------------------------------------------------------------
    root_history = ((T_FIRST, True),)
    root = TreeNode(
        node_id="root", t=T_FIRST, closure_open=True,
        history=root_history, parent_id="", cum_prob=1.0,
        alpha_C=ALPHA_C_PRIOR, beta_C=BETA_C_PRIOR,
        alpha_R=ALPHA_R_PRIOR, beta_R=BETA_R_PRIOR,
        children=[],
    )
    nodes["root"] = root
    realized_ids.append("root")

    # -------------------------------------------------------------------------
    # Walk the realized "pre-closure: stays OPEN" branch month by month.
    # At each open-node, posterior mean of p_C gives the branching probability.
    # -------------------------------------------------------------------------
    prev_id = "root"
    for t in range(T_FIRST + 1, T_PRE_END + 1):
        parent = nodes[prev_id]
        p_close = beta_mean(parent.alpha_C, parent.beta_C)
        # Realized: stays OPEN -> beta_C posterior count += 1
        new_hist_open = parent.history + ((t, True),)
        n_open = add_node(t, True, prev_id, new_hist_open, 1.0 - p_close,
                          parent.alpha_C, parent.beta_C + 1.0,
                          parent.alpha_R, parent.beta_R)
        # Counterfactual: closes -> alpha_C += 1
        new_hist_closed = parent.history + ((t, False),)
        add_node(t, False, prev_id, new_hist_closed, p_close,
                 parent.alpha_C + 1.0, parent.beta_C,
                 parent.alpha_R, parent.beta_R)
        prev_id = n_open.node_id
        realized_ids.append(prev_id)

    # -------------------------------------------------------------------------
    # t = +1: closure observed. From the last pre-closure OPEN node, the
    # realized branch is the (lower-probability) "closes" branch.
    # -------------------------------------------------------------------------
    parent = nodes[realized_ids[-1]]  # t=0 OPEN node
    p_close = beta_mean(parent.alpha_C, parent.beta_C)
    closure_history = parent.history + ((T_CLOSURE_START, False),)
    closure_node = add_node(T_CLOSURE_START, False, parent.node_id,
                            closure_history, p_close,
                            parent.alpha_C + 1.0, parent.beta_C,
                            parent.alpha_R, parent.beta_R)
    # Counterfactual at t=+1: closure did NOT happen (in fact it did)
    no_close_hist = parent.history + ((T_CLOSURE_START, True),)
    add_node(T_CLOSURE_START, True, parent.node_id, no_close_hist, 1.0 - p_close,
             parent.alpha_C, parent.beta_C + 1.0,
             parent.alpha_R, parent.beta_R)
    realized_ids.append(closure_node.node_id)

    # -------------------------------------------------------------------------
    # Walk the realized "closure: stays CLOSED" branch through t = +6.
    # At each closed-node, posterior mean of p_R gives the branching prob.
    # -------------------------------------------------------------------------
    prev_id = closure_node.node_id
    for t in range(T_CLOSURE_START + 1, T_CLOSURE_END + 1):
        parent = nodes[prev_id]
        p_reopen = beta_mean(parent.alpha_R, parent.beta_R)
        # Realized: stays CLOSED -> beta_R += 1
        new_hist_closed = parent.history + ((t, False),)
        n_closed = add_node(t, False, prev_id, new_hist_closed, 1.0 - p_reopen,
                            parent.alpha_C, parent.beta_C,
                            parent.alpha_R, parent.beta_R + 1.0)
        # Counterfactual: reopens
        new_hist_open = parent.history + ((t, True),)
        add_node(t, True, prev_id, new_hist_open, p_reopen,
                 parent.alpha_C, parent.beta_C,
                 parent.alpha_R + 1.0, parent.beta_R)
        prev_id = n_closed.node_id
        realized_ids.append(prev_id)

    # -------------------------------------------------------------------------
    # t = +7: realized REOPENING. From the last CLOSED node (t=+6), realized
    # branch is the "reopens" outcome.
    # -------------------------------------------------------------------------
    parent = nodes[realized_ids[-1]]  # t=+6 CLOSED node
    p_reopen = beta_mean(parent.alpha_R, parent.beta_R)
    reopen_hist = parent.history + ((T_POST_START, True),)
    reopen_node = add_node(T_POST_START, True, parent.node_id, reopen_hist,
                           p_reopen,
                           parent.alpha_C, parent.beta_C,
                           parent.alpha_R + 1.0, parent.beta_R)
    # Counterfactual: closure extends past +6
    stays_closed_hist = parent.history + ((T_POST_START, False),)
    add_node(T_POST_START, False, parent.node_id, stays_closed_hist, 1.0 - p_reopen,
             parent.alpha_C, parent.beta_C,
             parent.alpha_R, parent.beta_R + 1.0)
    realized_ids.append(reopen_node.node_id)

    # -------------------------------------------------------------------------
    # Walk realized "post-closure: stays OPEN" branch through t = +12.
    # -------------------------------------------------------------------------
    prev_id = reopen_node.node_id
    for t in range(T_POST_START + 1, T_LAST + 1):
        parent = nodes[prev_id]
        p_close = beta_mean(parent.alpha_C, parent.beta_C)
        new_hist_open = parent.history + ((t, True),)
        n_open = add_node(t, True, prev_id, new_hist_open, 1.0 - p_close,
                          parent.alpha_C, parent.beta_C + 1.0,
                          parent.alpha_R, parent.beta_R)
        new_hist_closed = parent.history + ((t, False),)
        add_node(t, False, prev_id, new_hist_closed, p_close,
                 parent.alpha_C + 1.0, parent.beta_C,
                 parent.alpha_R, parent.beta_R)
        prev_id = n_open.node_id
        realized_ids.append(prev_id)

    if verbose:
        print(f"Built tree with {len(nodes)} nodes, {len(realized_ids)} on realized path")
        print(f"Realized cumulative probability: {nodes[realized_ids[-1]].cum_prob:.6g}")

    return nodes, realized_ids

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
