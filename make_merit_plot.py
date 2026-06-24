import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import market as m11
from model_config import LEADERS, LEADER_REGIONS, demand_blocks_base

REGIONS = ("EU", "Asia")

def supply_items(region, closed=False):
    items = []
    for L in LEADERS:
        if region in LEADER_REGIONS[L]:
            cap = m11._LEADER_CAP_BASE[L] * (0.0 if (closed and L == "Gulf") else 1.0)
            items.append((m11.leader_cost[L][region], cap, L, True))
    for s in m11.FRINGE_BY_REGION[region]:
        cap = m11.fringe[region][s]["cap_closed" if closed else "cap_open"]
        items.append((m11.fringe_cost(region, s), cap, s, False))
    items = [it for it in items if it[1] > 1e-6]
    items.sort(key=lambda x: x[0])
    return items

def demand_blocks(region):           # (size, wtp) descending wtp
    return sorted(demand_blocks_base[region], key=lambda b: -b[1])

def marg(curve, q):                  # marginal value at cumulative quantity q
    x = 0.0
    for val, size in curve:
        if q < x + size - 1e-9:
            return val
        x += size
    return None

def clearing(items, blocks):
    sup = [(c, cap) for c, cap, *_ in items]          # (cost, size) asc cost
    dem = [(w, s) for s, w in blocks]                 # (wtp, size) desc wtp
    qmax = min(sum(c for _, c in sup), sum(s for s, _ in blocks))
    q, step, qstar = 0.0, qmax/2000.0, 0.0
    while q < qmax:
        d, s = marg(dem, q), marg(sup, q)
        if d is None or s is None or d < s:
            break
        qstar = q
        q += step
    price = marg(dem, max(qstar - 1e-6, 0.0))         # marginal buyer's WTP
    return qstar, price

fig, axes = plt.subplots(2, 2, figsize=(15, 11), sharey="row")
cmap = plt.get_cmap("tab20")
for r, region in enumerate(REGIONS):
    for c, (closed, label) in enumerate([(False, "open / pre-crisis"),
                                         (True, "crisis (Gulf + Hormuz suppliers removed)")]):
        ax = axes[r][c]
        items = supply_items(region, closed=closed)
        x = 0.0
        for i, (cost, cap, name, isL) in enumerate(items):
            ax.bar(x, cost, width=cap, align="edge", color=cmap(i % 20),
                   edgecolor="black", lw=0.6, alpha=0.95 if isL else 0.7)
            ax.text(x + cap/2, cost + 0.7, f"{name}\n{cap:.1f}", ha="center",
                    va="bottom", fontsize=6, fontweight="bold" if isL else "normal")
            x += cap
        total_cap = x
        blocks = demand_blocks(region)
        dx, dy, xx = [], [], 0.0
        for size, wtp in blocks:
            dx += [xx, xx + size]; dy += [wtp, wtp]; xx += size
        ax.step(dx, dy, where="post", color="crimson", lw=2.4, label="demand WTP ladder")
        qstar, price = clearing(items, blocks)
        ax.axhline(price, color="black", ls="--", lw=1.5)
        ax.plot([qstar], [price], "o", color="black", ms=9, zorder=5)
        ax.annotate(f"clearing price ≈ {price:.0f} EUR/MWh\n(rationed at {qstar:.0f} bcm)",
                    xy=(qstar, price), xytext=(qstar*0.30, price + 14),
                    fontsize=9, fontweight="bold",
                    arrowprops=dict(arrowstyle="->", lw=1.3))
        ax.set_title(f"{region} — {label}", fontsize=11)
        ax.set_xlabel("cumulative capacity / demand  [bcm/month]")
        ax.set_xlim(0, max(total_cap, xx)*1.02); ax.set_ylim(0, 125)
        ax.grid(axis="y", ls=":", alpha=0.4)
        if r == 0 and c == 0: ax.legend(loc="center right", fontsize=9)
    axes[r][0].set_ylabel(f"{region}: cost / WTP  [EUR/MWh]")
fig.suptitle("LNG supply merit order vs demand-response ladder — price is set by the "
             "marginal BUYER (demand ladder), not supply cost\n"
             "(static snapshot, base demand; leaders bold. Removing the Gulf shifts the "
             "clearing point UP the demand ladder.)", fontsize=11)
fig.tight_layout(rect=[0, 0, 1, 0.95])
out = os.path.join("plots", "merit_order.png")
os.makedirs("plots", exist_ok=True)
fig.savefig(out, dpi=135)
print("saved", os.path.abspath(out))
for region in REGIONS:
    for closed in (False, True):
        q, p = clearing(supply_items(region, closed), demand_blocks(region))
        print(f"{region:4} {'crisis' if closed else 'open  '}: clearing price {p:5.1f} EUR/MWh at {q:5.1f} bcm")
