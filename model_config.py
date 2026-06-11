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
pipeline = {
    "EU":   {"Norway_pipe":  {"cost": 16.0, "cap_open": 10.0, "cap_closed": 10.0},
             "Algeria_pipe": {"cost": 20.0, "cap_open":  4.0, "cap_closed":  4.0}},
    "Asia": {"Sakhalin_pipe":{"cost": 16.0, "cap_open":  3.0, "cap_closed":  3.0}},
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
    "EU":   [(24.0, 120.0), (2.0, 80.0), (2.0, 65.0), (2.0, 55.0),
             (2.0,  48.0), (2.0, 42.0), (2.0, 37.0), (2.0, 31.0), (2.0, 24.0)],
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
EU_NOV_TARGET_FRAC = 0.90
# Crisis-year November (1 Nov 2026, t = +8): the 90% target is unattainable
# and the EU flexibility mechanism applies. Fulwood (2026, OIES) projects
# 1 Nov 2026 stocks of 76-81 bcm; we impose 80% (80 bcm).
EU_NOV_TARGET_FRAC_2026 = 0.80
NOV_2026_T = +8             # tree month of the crisis-year 1-Nov checkpoint

# Observed EU storage calibration targets (end-of-month stocks, bcm):
# end-March 2026 EU-27 stocks were 30 bcm -- 6 bcm lower year-on-year and
# 16 bcm below the 2022-25 end-March average (Fulwood 2026, OIES / GIE
# AGSI+). Reported against the model's realized-path storage trajectory.
STORAGE_TARGETS_EU = {+1: 30.0}

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
