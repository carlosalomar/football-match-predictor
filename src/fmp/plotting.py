"""Shared chart styling: one categorical palette in a fixed order, recessive axes."""
COLORS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
INK = "#0b0b0b"
SECONDARY = "#52514e"
MUTED = "#9a998f"
GRID = "#e6e5e0"


def style(ax):
    ax.spines[["top", "right"]].set_visible(False)
    for s in ["left", "bottom"]:
        ax.spines[s].set_color(MUTED)
    ax.tick_params(colors=SECONDARY, labelsize=9)
    ax.yaxis.label.set_color(SECONDARY)
    ax.xaxis.label.set_color(SECONDARY)
    ax.grid(axis="y", color=GRID, lw=0.8)
    ax.set_axisbelow(True)
