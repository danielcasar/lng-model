"""
LNG market data, calibrated from:

    Zwickl-Bernhard, S. & Neumann, A. (2024).
    Modeling Europe's role in the global LNG market 2040.
    Energy, 301, 131612. https://doi.org/10.1016/j.energy.2024.131612

Specifically:
    • BEP (break-even prices per exporter) — their Table 6, column 2.
    • Liquefaction capacities Q^Liq_e — their Table 6, column 3.
    • Transport-cost decomposition (chartering, fuel, boil-off, fees, port)
      following their Eq. 8–15 (Appendix A). Values here are simplified
      round-number estimates per (exporter, importer) pair.

Their model is annual at $/MMBtu and billion MMBtu/year — we convert to the
monthly bcm and EUR/MWh conventions used in the bilevel model.

CLOSURE MECHANISM (this is the part the user emphasised):
    Only routes that physically transit the Strait of Hormuz are affected by
    a Hormuz closure. Production (BEP) does NOT change. Transport costs of
    untouched routes (e.g. USA → EU, Australia → Asia) do NOT change.
"""

# =============================================================================
# Unit conversions
# =============================================================================

USD_TO_EUR    = 0.93                # rough mid-2024 FX (ECB reference rate, 2024 avg)
MMBTU_TO_MWH  = 0.293                # 1 MMBtu = 0.293 MWh (IEA standard conversion)

def usd_per_mmbtu_to_eur_per_mwh(usd):
    """$/MMBtu  →  EUR/MWh. Multiplier ≈ 3.17."""
    return usd * USD_TO_EUR / MMBTU_TO_MWH

BCM_PER_BILLION_MMBTU = 27.8         # 1 bcm ≈ 0.036 bn MMBtu (IEA Natural Gas Information)

def annual_bn_mmbtu_to_monthly_bcm(annual):
    return annual * BCM_PER_BILLION_MMBTU / 12

# =============================================================================
# Break-even prices  ($/MMBtu)  — Table 6 (column 2)
# =============================================================================

BEP_USD = {
    "Algeria":           4.9,
    "Australia":         9.5,        # updated upward — Zou et al. (2022) Table 3:
                                     #   Gorgon FOB $12.32, Ichthys $13.55 (high-cost
                                     #   newer projects); average across Aussie
                                     #   projects ≈ $9.5 (older NWS / Pluto are cheaper)
    "Indonesia":         6.0,
    "Malaysia":          6.0,
    "Nigeria":           4.1,
    "Oman":              6.0,
    "Other_Africa":      4.5,
    "Other_Americas":    6.0,
    "Other_Asia_Pacific":8.4,
    "Other_Europe":      5.0,        # incl. Norway
    "Other_Middle_East": 3.0,
    "Qatar":             2.4,        # Zou Table 3: Qatargas II $1.96, RasGas I $2.20
    "Russia":            4.5,        # Zou Table 3: Yamal $4.33
    "Trinidad":          5.1,
    "USA":               5.9,        # Zou Table 3: Sabine Pass $5.46, Corpus Christi $5.81
}

# =============================================================================
# Transport costs  ($/MMBtu)  — simplified per (exporter, importer)
# Derived qualitatively from paper Eq. 8–15: longer routes / canal fees / boil-
# off raise TC. Pipeline routes are essentially zero. The figures below are
# round-number estimates suitable for an illustrative model; calibrating these
# exactly per their Appendix A formulas would be a refinement.
# =============================================================================

TC_USD = {
    # ---- to EU ----------------------------------------------------------
    ("Other_Europe",      "EU"):  0.0,    # Norway / Other Europe pipeline
    ("Algeria",           "EU"):  1.5,    # short-haul Med shipping / pipeline
    ("Nigeria",           "EU"):  2.0,
    ("Qatar",             "EU"):  1.5,    # via Hormuz + Suez
    ("Trinidad",          "EU"):  1.8,    # Atlantic crossing
    ("Other_Americas",    "EU"):  2.2,
    ("Other_Africa",      "EU"):  2.0,
    ("USA",               "EU"):  2.0,
    # ---- to Asia --------------------------------------------------------
    ("Russia",            "Asia"): 0.5,   # Sakhalin → NE Asia, short
    ("Australia",         "Asia"): 0.7,   # Zou Table 4/5: Gorgon → NE Asia $0.67-0.68
    ("Indonesia",         "Asia"): 1.0,
    ("Malaysia",          "Asia"): 1.0,
    ("Qatar",             "Asia"): 1.2,   # via Hormuz
    ("Oman",              "Asia"): 1.2,
    ("Other_Middle_East", "Asia"): 1.4,
    ("Other_Asia_Pacific","Asia"): 1.6,
    ("USA",               "Asia"): 3.5,   # Panama or Cape, long-haul
}

# =============================================================================
# EVENT DEFINITIONS — what's blocked under each scenario
#
# Each event specifies which suppliers are unavailable during the closure
# period. The model is event-agnostic: switch scenarios by changing which
# entry you pass to regional_supply().
#
#   blocked_suppliers: exporters whose entire output is unavailable
#                      (used when the exporter cannot ship anywhere, e.g.
#                       Qatar during a Hormuz closure)
#
# Route-level blocking (Suez, Panama) was removed for simplicity — it can
# be re-added if needed by re-introducing a blocked_routes parameter.
# =============================================================================

EVENTS = {
    "hormuz_closure": {
        "name":        "Strait of Hormuz closure",
        "description": "Qatari, Omani and Other ME LNG stranded — all routes via Hormuz blocked.",
        "blocked_suppliers": ["Qatar", "Oman", "Other_Middle_East"],
    },
    "russia_eu_cutoff": {
        "name":        "Russia supply cutoff",
        "description": "Sanctions / pipeline outage — entire Russian export capacity unavailable.",
        "blocked_suppliers": ["Russia", "Other_Europe"],
    },
    "no_event": {
        "name":        "Baseline (no disruption)",
        "description": "Counterfactual: no chokepoint closure.",
        "blocked_suppliers": [],
    },
}

# =============================================================================
# Liquefaction capacities  (billion MMBtu / year)  — Table 6 (column 3)
# Converted to bcm/month via BCM_PER_BILLION_MMBTU / 12.
# =============================================================================

LIQ_CAP_BN_MMBTU_YR = {
    "Algeria":            0.720,
    "Australia":          5.040,
    "Indonesia":          1.357,
    "Malaysia":           1.548,
    "Nigeria":            2.520,
    "Oman":               0.571,
    "Other_Africa":       3.600,
    "Other_Americas":     2.160,
    "Other_Asia_Pacific": 0.752,
    "Other_Europe":       0.310,
    "Other_Middle_East":  0.277,
    "Qatar":              6.255,
    "Russia":             3.060,
    "Trinidad":           0.612,
    "USA":                7.920,
}

# =============================================================================
# Helpers — assemble a region's supply dictionary in the model's expected shape
# =============================================================================

def delivered_cost_eur_mwh(exporter, importer):
    """BEP + TC, in EUR/MWh. Constant across closure/open states."""
    bep = usd_per_mmbtu_to_eur_per_mwh(BEP_USD[exporter])
    tc  = usd_per_mmbtu_to_eur_per_mwh(TC_USD[(exporter, importer)])
    return bep + tc

def regional_supply(importer, exporters, region_share, blocked_suppliers=None):
    """Build a {supplier: {cost, cap_open, cap_closed}} dict for one importer.

    region_share:      {exporter: float in [0,1]} — fraction of the exporter's
                       global liquefaction we assume can serve this importer.
    blocked_suppliers: list of exporters whose entire output is unavailable
                       during the closure (e.g. Qatar in a Hormuz closure).
    """
    blocked_suppliers = blocked_suppliers or []

    out = {}
    for exp in exporters:
        is_blocked = exp in blocked_suppliers
        cap = annual_bn_mmbtu_to_monthly_bcm(LIQ_CAP_BN_MMBTU_YR[exp]) \
              * region_share.get(exp, 1.0)
        out[exp] = {
            "cost":       delivered_cost_eur_mwh(exp, importer),
            "cap_open":   cap,
            "cap_closed": 0.0 if is_blocked else cap,
        }
    return out
