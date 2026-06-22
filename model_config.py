"""
SINGLE SOURCE OF ALL MODEL CONFIGURATION.

Every tunable value of the model lives in this file: time horizon, Bayesian
priors, strategic-leader definitions, fringe access, demand staircases,
seasonality, storage limits, Big-M constants and algorithm settings.
The supply-side DATA (break-even prices, transport costs, liquefaction
capacities) lives in lng_data.py, sourced from Zwickl-Bernhard & Neumann
(2024). The model scripts (11_epec_2leader.py, 12_rolling_epec.py,
scenario_tree.py) contain only model logic and import everything from here.

Each value carries its source as an inline comment. The companion sheet
parameters.csv documents the same values row-by-row with units, a type flag
(Data / Derived / Calibrated / Assumption / Numerical) and full citations —
update BOTH files together when changing a value.
"""

# =============================================================================
# EVENT
# =============================================================================

EVENT_NAME = "hormuz_closure"    # key into lng_data.EVENTS

# =============================================================================
# TIME HORIZON
#
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
# =============================================================================

T_FIRST           = -5
T_PRE_END         =  0
T_CLOSURE_START   = +1     # March 2026 -- realized closure start (Kpler, 2 Mar 2026)
T_CLOSURE_END     = +6     # Aug 2026 -- scenario assumption: 6-month closure
T_POST_START      = +7
T_LAST            = +24

# t = +1 -> March 2026 (month 3) -> CAL_OFFSET = 2
CAL_OFFSET = 2

# =============================================================================
# BETA-BERNOULLI BAYESIAN PRIORS ON TRANSITION RATES
# =============================================================================

# The belief-driven premium this tree generates is the formalisation of
# what Fulwood (2024, OIES NG 195) calls the "fear premium": in 2022 TTF
# peaked on the FEAR of losing Russian gas and collapsed once the loss
# became certain ("the sky hadn't fallen in") -- prices move on beliefs
# about supply states, not only on realised supply. Our conjugate
# updating reproduces exactly that decay of the premium as uncertainty
# resolves.
#
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

# =============================================================================
# ESCALATION (downside) STATE -- two-sided closure uncertainty (v8)
# =============================================================================
# The two-state open/closed chain is one-sidedly OPTIMISTIC: from the closed
# state the only exit is reopening (supply returns, price falls), so the
# expected future price during a closure can only decline and no scarcity
# premium or upward price drift can form. We add an abstract ESCALATION
# state reachable only from the closed state, in which the disruption DEEPENS
# -- an additional fraction of global LNG supply is removed (a second
# chokepoint, a wider conflict, damaged trains). It is NEVER realized on the
# observed path; it exists purely as the downside branch agents price in.
# Its presence raises the expected future price at closed nodes, and because
# the reopening belief decays month-to-month (beta_R grows) the escalation
# tail gains relative weight as the closure persists -- producing a premium
# that RISES the longer the strait stays shut. The escalation hazard is a
# FIXED parameter, not Bayesian-updated: escalation is never observed on the
# path, so there is nothing to learn; it represents a persistent structural
# tail risk. Abstract first cut -- both knobs to be anchored to Fulwood's
# structural ("12-month") scenario once the mechanism is validated.
ESCALATION_RATE      = 0.10   # fixed monthly P(closed -> escalated)
ESCALATION_LOSS_FRAC = 0.25   # extra fraction of LNG supply removed if escalated

# PERSISTENCE (v9.1): a one-period escalation leaf is too myopic to reward
# precautionary storage -- the agent would have to value stock for a single
# month only, so it never refills against the tail. Making escalation an
# ABSORBING multi-month state (it persists from the month it strikes to the
# end of the horizon, supply staying depressed) gives gas carried INTO the
# crisis a repeated, high-WTP payoff in the bad branch, which is what makes
# pre-/early-crisis storage REFILL economic -- matching the observed EU
# behaviour (storage rose 30->46 bcm Mar-Jun 2026 despite the closure).
# Economically this is the standard precautionary-inventory motive against a
# persistent disaster state (Wachter & Zhu 2025, QE, "rare disasters").
ESCALATION_PERSIST = True     # escalated state absorbs to T_LAST (vs 1-month leaf)

# =============================================================================
# REALIZED CRISIS-DEEPENING DERATE -- duration-dependent seaborne-LNG loss (v9)
# =============================================================================
# The escalation state above is COUNTERFACTUAL (a priced-in tail that is never
# realized). The realized closure path, by contrast, treated every closed month
# as equally severe -- so the model's crisis prices SAGGED into the low-demand
# summer while the OBSERVED crisis prices ROSE (Apr ceasefire dip, then May-Jun
# re-escalation). We add a realized derate that grows with the number of months
# the strait has been shut: a closed node loses an extra REROUTE_RATE_PER_MONTH
# x (months closed so far) fraction of seaborne LNG (leaders + LNG fringe; NOT
# pipelines/domestic), capped at REROUTE_CAP. Combined rationale:
#   - REALIZED RE-ESCALATION: the crisis actually deepened over the spring
#     (calibration_targets.csv: "moderation after ceasefire" -> "re-escalation
#     creep"); a flat-severity closure cannot reproduce the rising profile.
#   - SHIPPING / REROUTING DRAG: a prolonged closure ties up the global LNG
#     fleet on longer (Cape-route) voyages and depletes floating inventory, so
#     EFFECTIVE deliverable volume keeps falling the longer it lasts
#     (Fulwood 2026, OIES, HormuzClosureGasFlows).
# Like the escalation knobs, the EXISTENCE/direction is sourced; the MAGNITUDE
# (rate, cap) is the calibration lever. No sourced volume or price is altered.
# Rate cut 0.10 -> 0.03 (v9.1) once PERSISTENT escalation was added: the
# precautionary-refill it triggers raises crisis prices on its own (injection
# competes with demand for scarce supply), so the realized derate now only adds
# a small residual drag. Combined (persistent escalation + reroute 0.03) the
# realized-path price RMSE falls to ~3.9 EUR/MWh and storage REFILLS through the
# crisis (Mar-Jun) as observed, instead of hugging the 30% floor.
REROUTE_RATE_PER_MONTH = 0.03   # extra seaborne-LNG loss per month-closed
REROUTE_CAP            = 0.30   # maximum cumulative derate

# =============================================================================
# STRATEGIC LEADERS
#
# The second strategic leader is a COMPOSITE of the Hormuz-transiting Gulf
# exporters: Qatar (93% of LNG shipments via Hormuz) and the UAE (96%),
# jointly ~20% of global LNG exports (Zwickl-Bernhard et al. 2026). Both are
# stranded identically under a closure, so they face the same strategic
# situation; aggregating them closes the gap between the event definition
# (which blocks all Hormuz-transiting capacity) and the strategic-actor set.
# UAE capacity is represented by the "Other_Middle_East" entry in lng_data.
# =============================================================================

LEADERS        = ["USA", "Gulf"]
LEADER_REGIONS = {"USA":  ["EU", "Asia"],
                  "Gulf": ["EU", "Asia"]}
GULF_MEMBERS   = ["Qatar", "Other_Middle_East"]
BLOCKED_LEADERS = {"Gulf"}    # stranded whenever the strait is closed

# Gulf restart is NOT instantaneous after a reopening (Fulwood 2026, OIES):
# Ras Laffan needs 2-4 weeks to restart plus 1-2 weeks to ramp, Das Island
# ~3 weeks -- so the first open month delivers only about half of capacity.
# Two damaged Ras Laffan trains stay offline until ~2031 (~1.3 bcm/month of
# the 15.1 bcm/month composite), so capacity after any closure is
# permanently ~9% lower within our horizon.
GULF_RESTART_RAMP  = 0.50   # capacity factor in the first month after reopening
GULF_DAMAGE_FACTOR = 0.91   # capacity factor thereafter (damaged trains)

# AVAILABLE vs nameplate LNG capacity (calibration v7): scheduled and
# unscheduled maintenance, technical outages and feedgas constraints keep
# available capacity below nameplate -- Fulwood (2024, OIES NG 195,
# footnote 3 and Fig. 4): the ~98% pre-crisis utilisation is of AVAILABLE
# capacity. Applied to all LNG suppliers (leaders + fringe), not to
# pipelines or the non-LNG aggregates.
LNG_AVAILABILITY = 0.92

# Per-leader delivery floor (calibration v3). The floor represents the
# share of capacity that is NOT strategically withholdable:
#   USA  0.90 -- US liquefaction ran at ~full utilisation through
#                2025-26 (EIA LNG export data); "USA" aggregates many
#                competing private exporters whose individual incentive is
#                to dispatch, so the unified-actor withholding power is small.
#   Gulf 0.85 -- QatarEnergy's portfolio is ~85% long-term contracted
#                (GIIGNL Annual Report); only the residual is discretionary.
# The floor binds total dispatch (not per-destination), so cross-basin
# arbitrage of the contracted volume remains a strategic choice.
# During a closure the blocked leader's capacity is zero => floor zero.
CONTRACT_FLOOR = {"USA": 0.90, "Gulf": 0.85}

# =============================================================================
# FRINGE SUPPLIERS (price-taking)
#
# Small price-taking LNG fringe RESTORED (calibration v2): the first
# calibration run showed that without a competitive fringe, the two Cournot
# leaders facing the steep (inelastic) demand curve withhold supply until
# prices hit the essential-block ceiling permanently (EU model 119 vs
# observed 33-38). The real market is disciplined by many small price-taking
# exporters (~17 bcm/month jointly); restoring them caps the strategic
# markup at realistic levels. Access shares are stylised destination-market
# fractions reflecting historical trade patterns (IEA gas-market reports).
# =============================================================================

SPOT_TRADABLE = 1.00    # full nameplate spot-tradable (former Hypothesis H1 removed)

EU_ACCESS = {
    "Algeria": 0.5, "Nigeria": 0.5, "Trinidad": 0.6,
    "Other_Americas": 0.3, "Other_Africa": 0.5,
}
ASIA_ACCESS = {
    "Indonesia": 0.7, "Malaysia": 0.7, "Oman": 0.7, "Other_Asia_Pacific": 0.8,
    "Australia": 1.0,    # demoted leader, kept as Asian price-taking fringe
    "Russia":    1.0,    # demoted leader, kept as Asian price-taking fringe
}

# Pipeline supply: Norwegian exports ~120 bcm/yr = ~10 bcm/month (IEA);
# Algeria (Transmed/Medgaz) ~4 bcm/month utilised; Sakhalin -> NE Asia
# ~3 bcm/month. Costs are round-number estimates.
#
# NON-LNG SUPPLY AGGREGATES (calibration v6.1): the demand staircases are
# calibrated to TOTAL observed gas demand (eu_demand_monthly.csv), which
# is served not only by LNG and the three big pipelines but also by
# domestic production and smaller pipe routes. The oversized 2040 LNG
# capacities used before v6 accidentally covered this missing supply;
# with the realistic 2026 LNG fleet it must be modelled explicitly,
# otherwise the market is artificially scarce in every month (v6 run:
# prices pinned at the rationing ceilings even pre-crisis).
#   EU_other_supply  ~9 bcm/month: EU domestic production ~3 (DE/IT/RO/
#     DK/NL residual), UK net flows ~2, Azerbaijan TAP ~1, Turkstream
#     ~1.3, Libya ~0.2, biomethane ~1.5  (IEA Gas Market Report 2026)
#   Asia_other_supply ~8 bcm/month (v7, was 12): Chinese pipeline
#     imports ONLY (Central Asia ~4 + Power of Siberia ~3.5 + Myanmar
#     ~0.3, IEA). Domestic production cannot surge into the seaborne
#     spot market that the JKM staircase represents.
pipeline = {
    "EU":   {"Norway_pipe":  {"cost": 16.0, "cap_open": 10.0, "cap_closed": 10.0},
             "Algeria_pipe": {"cost": 20.0, "cap_open":  4.0, "cap_closed":  4.0},
             "EU_other_supply": {"cost": 13.0, "cap_open": 9.0, "cap_closed": 9.0}},
    "Asia": {"Sakhalin_pipe":{"cost": 16.0, "cap_open":  3.0, "cap_closed":  3.0},
             "Asia_other_supply": {"cost": 12.0, "cap_open": 8.0, "cap_closed": 8.0}},
}

# =============================================================================
# DEMAND STAIRCASES -- CALIBRATED TO OBSERVED DATA
# (see calibration_targets.csv and eu_demand_monthly.csv)
#
# The block-WTP ladder follows the price-formation hypothesis of Fulwood
# (2024, OIES NG 195): in a supply-short market the price is set not by
# supply costs but by successive DEMAND RESPONSES -- coal/oil switching,
# efficiency, behavioural change, industrial closures, and finally
# rationing ("Panic!"). Each WTP block is one rung of that ladder; the
# crisis moves the marginal block up the ladder exactly as his
# multi-dimensional framework describes.
#
# EU: the observed (quantity, price) pairs straddling the closure imply a
# steep inverse demand: (Jan26: 49 bcm at EUR 37) vs (Mar26: 35 bcm at
# EUR 57) gives slope ~ -5.3 EUR/MWh per bcm of base demand, i.e. the
# inelastic short-run gas demand documented by Burke & Yang (2016).
# Base curve approximates P = 220 - 5.26*Q over Q in [24, 40] bcm:
# an essential block of 24 bcm at the rationing ceiling, then 8 blocks of
# 2 bcm stepping down. Pre-crisis winter clears around EUR 36-47; the
# crisis supply contraction moves the marginal block 2-3 tiers up to
# EUR 57-78, matching the observed TTF spike profile.
#
# Asia: larger base (~44 bcm addressable), shallower observed response;
# pre-crisis clears EUR 30-38, crisis EUR 48-68 (JKM $15-24/MMBtu range).
# Blocks 6-9 = deepened price-elastic tail (calibration v3): Indian /
# SE-Asian price-sensitive buyers absorb diverted cargoes at $8-12/MMBtu
# (EUR 25-38); Asia is the world's residual LNG sink. Without this tail
# the crisis-displaced US volumes could not clear in Asia and were dumped
# into the EU, crashing the model's EU price to EUR 19 while the observed
# crisis TTF was EUR 42-57 (set by EU-Asia cargo competition).
#
# WTP grid REPOSITIONED (calibration v5): equilibrium prices snap to block
# WTPs, so the grid must be dense in the OBSERVED clearing ranges. The v4
# grid had no EU step between 36 and 47 -- both runs (one-shot RMSE 14.71,
# rolling 14.50) cleared EU at exactly 47 pre-crisis (obs 33-38) and fell
# through to <=36 in the crisis (obs 42-57); Asia winter snapped to 58
# (obs 38). v5 keeps block counts and totals identical (no extra binaries)
# but places steps at ~5 EUR spacing across EU 31-55 and Asia 34-60.
# =============================================================================

# Asia tail re-anchored to Fulwood (2026, OIES): Indian demand destruction
# only "starts to trigger" at TTF ~$25/MMBtu (~EUR 79/MWh), and South-Asian
# + Chinese LNG imports fall by merely 3-18 bcm per YEAR even in the severe
# scenarios -- nothing like the 12 bcm per MONTH the v5 tail absorbed at
# EUR 23-34. v6 shrinks the deep tail and moves volume into the EUR 40-60
# mid-range where the crisis-month JKM (obs EUR 55-63) actually cleared.
demand_blocks_base = {
    # EU mid-rungs widened 2.0 -> 2.5 bcm (v7) so the staircase passes
    # exactly through the observed anchors: Jan-26 cumulative demand above
    # EUR 37 = 24*1.41 + 15 = 48.8 bcm at the observed (49 bcm, EUR 37);
    # Mar-26 cumulative above EUR 55 = 27.1 + 7.5 = 34.6 at the observed
    # (35 bcm, EUR 57).
    "EU":   [(24.0, 120.0), (2.5, 80.0), (2.5, 65.0), (2.5, 55.0),
             (2.5,  48.0), (2.5, 42.0), (2.5, 37.0), (2.0, 31.0), (2.0, 24.0)],
    "Asia": [(30.0, 105.0), (2.0, 79.0), (2.0, 68.0), (2.0, 60.0),
             (2.0,  52.0), (3.0, 46.0), (4.0, 40.0), (3.0, 32.0), (3.0, 24.0)],
}

# =============================================================================
# SEASONALITY
# =============================================================================

# Month-specific EU demand factors derived from observed EU monthly gas
# demand 2023-2025 (eu_demand_monthly.csv), normalised to the annual mean
# (~28.2 bcm/month). The real seasonal shape peaks at 1.70x in December
# and troughs at 0.62x in July.
EU_MONTH_FACTOR = {
    1: 1.41, 2: 1.33, 3: 1.13, 4: 0.87, 5: 0.69, 6: 0.63,
    7: 0.62, 8: 0.64, 9: 0.78, 10: 0.96, 11: 1.20, 12: 1.70,
}

# Asia: flatter seasonality (Japan/Korea heating partially offset by flat
# Chinese industrial demand) -- stylised two-level scheme.
# Winter factor lowered 1.25 -> 1.10 (calibration v5): the observed JKM
# winter premium 2025-26 was only ~+5-8% (EUR 38-39 winter vs ~36 autumn,
# calibration_targets.csv); 1.25 pushed the winter marginal block to 58.
WINTER = {11, 12, 1, 2, 3}
SUMMER = {6, 7, 8}
ASIA_WINTER_FACTOR = 1.10
ASIA_SUMMER_FACTOR = 0.85

# =============================================================================
# STORAGE
# =============================================================================

HOLDING_COST = 0.10    # constant physical storage holding cost (EUR/MWh-month)

storage = {
    # EU: aggregate working gas volume ~100 bcm, fill ~85% at model start
    # Sep 2025 (GIE AGSI+). S_term = typical end-of-winter level, imposed
    # at t = +24 (Feb 2028), far outside the reported window.
    "EU":   {"S_max": 100.0, "S_init": 85.0, "S_term": 30.0},
    # Asia: stylised -- NE-Asian LNG tank capacity is small relative to EU
    # underground storage.
    "Asia": {"S_max":  20.0, "S_init": 10.0, "S_term": 10.0},
}

# EU Regulation 2017/1938 (as amended by Reg. (EU) 2022/1032): 90%
# storage-filling target on 1 November, applied at every realized Nov node.
# Minimum operational storage level (v7.1): storage may not be drawn below
# this fraction of working-gas capacity in any month -- a security cushion
# the operators/regulation maintain, i.e. precautionary gas not released
# even under scarcity. Set at 30%: Fulwood (2026, OIES) reports EU storage
# at 30 bcm = 30% of working capacity at end-March 2026, the observed low
# point even in this tight year. Acts as a simple, data-anchored proxy for
# precautionary (risk-averse) storage behaviour without a risk measure.
STORAGE_FLOOR_FRAC = 0.30

EU_NOV_TARGET_FRAC = 0.90
# Crisis-year November (1 Nov 2026, t = +8): the 90% target is unattainable
# and the EU flexibility mechanism applies. Fulwood (2026, OIES) projects
# 1 Nov 2026 stocks of 76-81 bcm; we impose 80% (80 bcm).
EU_NOV_TARGET_FRAC_2026 = 0.80
NOV_2026_T = +8             # tree month of the crisis-year 1-Nov checkpoint

# Observed end-of-month EU storage (bcm, working-gas; model S_max = 100 scale).
# Reported alongside the model's realized-path storage trajectory so the
# storage behaviour can be eyeballed against reality, not only the prices.
#
# Values on a ~100 bcm working-gas basis (1% fill ~ 1 bcm), the same scale as
# S_max=100 and consistent with the firm 30 bcm = 30% end-March anchor. Series
# compiled from GIE AGSI+ -derived public sources (ACER winter 2025/26 report;
# Bruegel gas tracker; CEEnergyNews/AGSI+; OIES/Fulwood 2026; AGSI+ live
# mirrors), retrieved Jun 2026 -- exact daily AGSI+ readouts need a (free) API
# key, so the non-anchor months carry the confidence flags below.
# Confidence:
#   FIRM      : t=-2 Dec 2025 (63%, AGSI+/CEEnergyNews); t=+1 Mar 2026 (30 bcm =
#               30%, OIES/Fulwood 2026, corroborated by ACER "below 30%").
#   PROJECTION: t=+8 1-Nov-2026 78 bcm (= end-Oct stock; Fulwood 2026 proj.
#               76-81; the model's 80% Nov mandate targets 80).
#   APPROX    : Sep-Nov 2025 (ACER "~82% end-summer, withdrawals from mid-Nov");
#               Apr-Jun 2026 (ACER "1 Apr close to last year ~30%"; Bruegel
#               ~37% late-May; AGSI+ live ~40%->46% over June, injection season).
#   ESTIMATE  : Jan-Feb 2026 interpolated on the steep cold-Q1 withdrawal path
#               between the firm Dec (63%) and Mar (30%) anchors (+-4 pp).
STORAGE_OBS_EU = {
    -5: 83.0,   # Sep 2025  approx (ACER/S&P ~83% end-summer)
    -4: 82.0,   # Oct 2025  approx (ACER ~82%, withdrawals from mid-Nov)
    -3: 80.0,   # Nov 2025  approx (peak ~82% drawing from mid-Nov)
    -2: 63.0,   # Dec 2025  FIRM   (AGSI+/CEEnergyNews "finished 2025 at 63%")
    -1: 48.0,   # Jan 2026  est.   (interp. cold-Q1 withdrawal path)
     0: 38.0,   # Feb 2026  est.   (interp.; heading into the 30% pre-Hormuz low)
    +1: 30.0,   # Mar 2026  FIRM   (Fulwood 2026 / AGSI+ / ACER)
    +2: 31.0,   # Apr 2026  approx (ACER "1 Apr close to last year"; trough)
    +3: 37.0,   # May 2026  approx (Bruegel ~37% late-May, lowest seasonal level)
    +4: 46.0,   # Jun 2026  approx (AGSI+ live ~40%->46% over June)
    +8: 78.0,   # 1 Nov 2026 snapshot (= end-Oct stock; Fulwood 2026 proj. 76-81)
}
# Back-compat alias: the single firm crisis anchor used by older reports.
STORAGE_TARGETS_EU = {+1: STORAGE_OBS_EU[+1]}

# NOTE: the v4 "storage cycling envelope" (month-specific max-fill bounds
# from AGSI+ data) was REMOVED in v6 as too restrictive: seasonal storage
# release should emerge endogenously from the price arbitrage (Euler
# condition). The hoarding equilibrium it guarded against is now addressed
# by the tightened price Big-M (no phantom dual spikes financing the
# hoard) and by the rolling-horizon re-solving (plans whose payoff lives
# in never-reached futures are re-audited monthly).

# Physical injection / withdrawal deliverability limits (GIE aggregate
# technical capacity, EU-wide): prevents pathological storage jumps such
# as a 70 bcm refill within two months.
EU_MAX_INJECT_BCM   = 12.0   # per month
EU_MAX_WITHDRAW_BCM = 25.0   # per month

# =============================================================================
# NUMERICS -- Fortuny-Amat & McCarl (1981) Big-M complementarity constants
#
# Each constant must safely exceed the largest value the bounded quantity
# can take -- no economic content. M_PRICE tightened 600 -> 150 and M_KKT
# 600 -> 250 (calibration v5): the maximum WTP is 120, so no nodal price
# can exceed 120 and no stationarity expression can exceed ~240. The loose
# 600 admitted phantom dual spikes in low-probability counterfactual
# branches that financed hoarding / Euler-violating storage paths in
# time-limited (aborted) solves, and it weakened the LP relaxation,
# slowing every MIP solve.
# =============================================================================

M_FRINGE  = 60.0    # > max fringe supply per (region, supplier, month), bcm
M_DEMAND  = 60.0    # > max demand-block size, bcm
M_PRICE   = 150.0   # > max possible market price (max WTP = 120 EUR/MWh)
M_KKT     = 250.0   # > max value of any KKT stationarity expression
M_STORAGE = 150.0   # > max storage level (EU S_max = 100 bcm)

# =============================================================================
# ROLLING-HORIZON SETTINGS (12_rolling_epec.py)
# =============================================================================

# Roll over the reported window. Each re-solve still plans to T_LAST=+24,
# so even the last implemented month is chosen with a 12-month lookahead.
ROLL_START = T_FIRST      # -5  (Sep 2025)
ROLL_END   = +12          # Feb 2027

# Diagonalization settings per monthly re-solve: few iterations + moderate
# time limit. The damped equilibrium changes only incrementally from month
# to month, and early iterations capture most of the adjustment.
ROLL_MAX_ITER   = 4
ROLL_TIME_LIMIT = 120     # seconds per leader MPCC solve
