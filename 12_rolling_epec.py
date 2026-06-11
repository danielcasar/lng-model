"""
Step 12: ROLLING-HORIZON two-leader EPEC -- the main model.

Each month t on the realized path (Sep 2025 ... Feb 2027), the model:

  1. builds the belief tree from t onward (Bayesian posteriors given all
     closure-state observations up to t),
  2. solves the two-leader EPEC (USA + Gulf) on that subtree by Gauss-
     Seidel diagonalization -- agents plan over the full remaining horizon,
  3. IMPLEMENTS ONLY month t's decisions (leader dispatch, prices, storage
     flow at the subtree root),
  4. rolls forward: the realized status of t+1 arrives, beliefs update by
     conjugate Bayes, and the implemented end-of-month storage level
     becomes the next solve's initial condition.

Rationale (vs. the one-shot tree solve in 11_epec_2leader.py):
  - The one-shot solve computes an OPEN-LOOP equilibrium: leaders commit
    at t = -5 to a full contingent plan. Rolling re-optimization is the
    standard approximation of the more realistic FEEDBACK equilibrium --
    nobody irrevocably commits in September 2025 to their August 2026
    cargo allocations.
  - Re-planning every month with updated probabilities also removes the
    intertemporal hoarding equilibria that the one-shot MPCC's optimistic
    solution selection can produce: plans whose payoff lives in a
    never-reached future are re-audited every month.

Reuses the entire calibrated market model (demand staircase, fringe,
contract floors, mandates) from 11_epec_2leader.py via
its ctx-parameterized builders.
"""

import importlib
import time

from scenario_tree import (
    build_tree_from, realized_status, calendar_month,
    T_FIRST, T_LAST,
    ALPHA_C_PRIOR, BETA_C_PRIOR, ALPHA_R_PRIOR, BETA_R_PRIOR,
)
# Rolling-horizon settings (with rationale) live in model_config.py
from model_config import ROLL_START, ROLL_END, ROLL_MAX_ITER, ROLL_TIME_LIMIT

# module name starts with a digit -> import via importlib
m11 = importlib.import_module("11_epec_2leader")

REGIONS  = m11.REGIONS
LEADERS  = m11.LEADERS
storage  = m11.storage

MONTH_NAMES = ["", "Jan", "Feb", "Mar", "Apr", "May", "Jun",
               "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]


def warm_start_from(prev_nodes, prev_realized, prev_q, new_nodes):
    """Map the previous month's equilibrium onto the new subtree as a
    Gauss-Seidel warm start: every new node inherits the previous solution
    at the realized node of the same calendar time (the closest available
    proxy -- counterfactual branches did not exist in the old tree under
    the same ids). Blocked leaders are zeroed at closed nodes."""
    by_t = {prev_nodes[node_id].t: node_id for node_id in prev_realized}
    q0 = {}
    for leader in m11.LEADERS:
        q0[leader] = {}
        for region in REGIONS:
            col = prev_q[leader][region]
            q0[leader][region] = {}
            for node_id, node in new_nodes.items():
                prev_node_id = by_t.get(node.t)
                v = col.get(prev_node_id, 0.0) if prev_node_id is not None else 0.0
                if (not node.closure_open) and leader in m11.BLOCKED_LEADERS:
                    v = 0.0
                q0[leader][region][node_id] = v
    return q0


def roll():
    t_script = time.time()

    # Initial state: prior beliefs (root observation t=-5 OPEN is already
    # reflected in build_tree_from semantics: counts passed are those AT
    # the root) and the calibrated initial storage levels.
    counts  = (ALPHA_C_PRIOR, BETA_C_PRIOR, ALPHA_R_PRIOR, BETA_R_PRIOR)
    s_state = {region: storage[region]["S_init"] for region in REGIONS}

    trajectory = []   # one record per implemented month
    prev_sol   = None   # (nodes, realized_ids, q_eq) of the previous roll

    for t0 in range(ROLL_START, ROLL_END + 1):
        open0 = realized_status(t0)
        t_roll = time.time()

        nodes, realized_ids = build_tree_from(t0, open0, *counts)
        ctx = m11.make_ctx(nodes, realized_ids, s_state)
        q_init = (warm_start_from(*prev_sol, nodes)
                  if prev_sol is not None else None)

        print(f"\n{'='*78}")
        print(f"ROLL t={t0:+d} ({MONTH_NAMES[calendar_month(t0)]}) "
              f"status={'OPEN' if open0 else 'CLOSED'}  "
              f"tree={len(nodes)} nodes  S_EU={s_state['EU']:.1f}  "
              f"beliefs aC/bC={counts[0]:.0f}/{counts[1]:.0f} "
              f"aR/bR={counts[2]:.0f}/{counts[3]:.0f}", flush=True)

        q_eq, prices, profits, iters, stocks = m11.diagonalize(
            ctx=ctx, max_iter=ROLL_MAX_ITER, tol=0.5, alpha=0.5,
            time_limit=ROLL_TIME_LIMIT, verbose=False, q_init=q_init)
        prev_sol = (nodes, realized_ids, q_eq)

        root_id = realized_ids[0]
        rec = {
            "t": t0,
            "month": MONTH_NAMES[calendar_month(t0)],
            "open": open0,
            "p_EU":   prices["EU"][root_id]   if prices else float("nan"),
            "p_AS":   prices["Asia"][root_id] if prices else float("nan"),
            "qUSA_EU": q_eq["USA"]["EU"][root_id],
            "qUSA_AS": q_eq["USA"]["Asia"][root_id],
            "qGLF_EU": q_eq["Gulf"]["EU"][root_id],
            "qGLF_AS": q_eq["Gulf"]["Asia"][root_id],
            "S_EU":   stocks["EU"][root_id]   if stocks else float("nan"),
            "S_AS":   stocks["Asia"][root_id] if stocks else float("nan"),
            "secs":   time.time() - t_roll,
        }
        trajectory.append(rec)
        print(f"  implemented: p_EU={rec['p_EU']:.1f} p_AS={rec['p_AS']:.1f} "
              f"S_EU(end)={rec['S_EU']:.1f}  [{rec['secs']:.0f}s]", flush=True)

        # Roll the state forward: implemented storage becomes next initial
        # condition; beliefs update with the realized transition to t0+1.
        if stocks:
            s_state = {region: max(0.0, stocks[region][root_id]) for region in REGIONS}
        if t0 < ROLL_END:
            child_realized = nodes[realized_ids[1]]
            counts = (child_realized.alpha_C, child_realized.beta_C,
                      child_realized.alpha_R, child_realized.beta_R)

    # ----------------------------------------------------------------------
    # Rolled trajectory + calibration report
    # ----------------------------------------------------------------------
    print("\n" + "=" * 78)
    print("ROLLING-HORIZON trajectory (each month re-solved with updated beliefs)")
    print("(Gulf = Qatar + UAE composite leader)")
    print("=" * 78)
    hdr = (f"{'t':>3} {'mo':>4} {'status':>8}  {'p_EU':>7} {'p_AS':>7}   "
           f"{'USA_EU':>7} {'USA_AS':>7} {'GLF_EU':>7} {'GLF_AS':>7}  {'S_EU':>6}")
    print(hdr)
    print("-" * len(hdr))
    for rec in trajectory:
        status = "OPEN" if rec["open"] else "CLOSED"
        print(f"{rec['t']:>+3d} {rec['month']:>4} {status:>8}  "
              f"{rec['p_EU']:>7.2f} {rec['p_AS']:>7.2f}   "
              f"{rec['qUSA_EU']:>7.2f} {rec['qUSA_AS']:>7.2f} "
              f"{rec['qGLF_EU']:>7.2f} {rec['qGLF_AS']:>7.2f}  "
              f"{rec['S_EU']:>6.1f}")

    TARGETS = {  # t: (TTF_eur, JKM_eur) -- see calibration_targets.csv
        -5: (33.0, 36.0), -4: (34.0, 37.0), -3: (36.0, 38.0),
        -2: (38.0, 39.0), -1: (37.0, 38.0),  0: (37.0, 38.0),
         1: (57.0, 63.0),  2: (42.0, 55.5),  3: (46.0, 57.0),
         4: (49.6, 59.5),
    }
    print("\n" + "=" * 70)
    print("CALIBRATION REPORT (rolling): model vs observed (Sep 25 - Jun 26)")
    print("=" * 70)
    print(f"{'t':>3} {'mo':>6}  {'EU_mod':>7} {'EU_obs':>7} {'dEU':>6}   "
          f"{'AS_mod':>7} {'AS_obs':>7} {'dAS':>6}")
    sq_err, n_obs = 0.0, 0
    for rec in trajectory:
        if rec["t"] not in TARGETS:
            continue
        eu_obs, as_obs = TARGETS[rec["t"]]
        print(f"{rec['t']:>+3d} {rec['month']:>6}  "
              f"{rec['p_EU']:>7.1f} {eu_obs:>7.1f} {rec['p_EU']-eu_obs:>+6.1f}   "
              f"{rec['p_AS']:>7.1f} {as_obs:>7.1f} {rec['p_AS']-as_obs:>+6.1f}")
        sq_err += (rec["p_EU"] - eu_obs) ** 2 + (rec["p_AS"] - as_obs) ** 2
        n_obs  += 2
    rmse = (sq_err / max(n_obs, 1)) ** 0.5
    print("-" * 70)
    print(f"RMSE over {n_obs} observations: {rmse:.2f} EUR/MWh")

    for t_obs, s_obs in sorted(m11.STORAGE_TARGETS_EU.items()):
        rec_obs = next((x for x in trajectory if x["t"] == t_obs), None)
        if rec_obs is not None:
            print(f"Storage check t={t_obs:+d}: model S_EU={rec_obs['S_EU']:.1f} bcm "
                  f"vs observed {s_obs:.1f} bcm (GIE AGSI+ / Fulwood 2026)")

    total_min = (time.time() - t_script) / 60
    print(f"\nTotal computing time: {total_min:.1f} min "
          f"({len(trajectory)} monthly re-solves x {ROLL_MAX_ITER} iterations)",
          flush=True)
    return trajectory


if __name__ == "__main__":
    print("=" * 78)
    print("ROLLING-HORIZON two-leader stochastic Stackelberg EPEC (USA + Gulf)")
    print(f"Roll window: t={ROLL_START:+d} .. {ROLL_END:+d}; "
          f"each re-solve plans to t={T_LAST:+d}")
    print("=" * 78, flush=True)
    roll()
