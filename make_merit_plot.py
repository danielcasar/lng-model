import os, importlib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
m11 = importlib.import_module("11_epec_2leader")
from model_config import LEADERS, LEADER_REGIONS, demand_blocks_base

REGIONS = ("EU", "Asia")

def supply_items(region, closed=False):
    """(cost, cap, name, is_leader) sorted by delivered cost, open or closed state."""
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

def demand_steps(region):
    """cumulative-quantity, wtp pairs (descending WTP) for the base staircase."""
    blocks = demand_blocks_base[region]
    xs, ys, x = [], [], 0.0
    for size, wtp in blocks:
        xs += [x, x + size]; ys += [wtp, wtp]; x += size
    return xs, ys

fig, axes = plt.subplots(1, 2, figsize=(14, 6), sharey=True)
cmap = plt.get_cmap("tab20")
for ax, region in zip(axes, REGIONS):
    items = supply_items(region, closed=False)
    x = 0.0
    for i, (cost, cap, name, is_leader) in enumerate(items):
        ax.bar(x, cost, width=cap, align="edge",
               color=cmap(i % 20), edgecolor="black", linewidth=0.6,
               alpha=0.95 if is_leader else 0.7)
        ax.text(x + cap/2, cost + 0.6,
                f"{name}\n{cap:.1f}", ha="center", va="bottom", fontsize=6.5,
                fontweight="bold" if is_leader else "normal", rotation=0)
        x += cap
    total_cap = x
    # demand staircase overlay
    dx, dy = demand_steps(region)
    dx = [min(v, total_cap*1.05) for v in dx]
    ax.step(dx, dy, where="post", color="crimson", lw=2.2, label="demand WTP ladder")
    ax.set_title(f"{region}: supply merit order (open state) + demand ladder", fontsize=11)
    ax.set_xlabel("cumulative capacity / demand  [bcm/month]")
    ax.set_xlim(0, total_cap*1.02)
    ax.set_ylim(0, 90)
    ax.grid(axis="y", ls=":", alpha=0.4)
    ax.legend(loc="upper right", fontsize=9)
axes[0].set_ylabel("delivered cost / willingness-to-pay  [EUR/MWh]")
fig.suptitle("LNG supply merit order vs demand-response ladder "
             "(leaders bold; costs = ZB&N 2024 BEP + transport)", fontsize=12)
fig.tight_layout(rect=[0,0,1,0.96])
out = os.path.join("plots", "merit_order.png")
os.makedirs("plots", exist_ok=True)
fig.savefig(out, dpi=140)
print("saved", os.path.abspath(out))
# also print the tables
for region in REGIONS:
    print(f"\n{region} merit order (open):")
    for cost, cap, name, isL in supply_items(region):
        print(f"  {cost:6.1f} EUR/MWh  {cap:5.2f} bcm/mo  {'LEADER ' if isL else ''}{name}")
