"""Polished mechanism-path diagrams for the Streamlit discovery view."""

from textwrap import wrap

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch


ROLE_STYLE = {
    "Drug": {"accent": "#60A5FA", "fill": "#16243D", "edge": "#326196"},
    "Gene": {"accent": "#5EEAD4", "fill": "#123134", "edge": "#23716E"},
    "Pathway": {"accent": "#FBBF24", "fill": "#392D16", "edge": "#8B6819"},
    "Disease": {"accent": "#FCA5A5", "fill": "#3B2029", "edge": "#854154"},
}

RELATIONSHIPS = ("interacts with", "participates in", "associated with")


def _roles_for_path(path_nodes):
    """Assign roles by graph-path position, avoiding brittle label keyword checks."""
    default_roles = ("Drug", "Gene", "Pathway", "Disease")
    if len(path_nodes) == len(default_roles):
        return default_roles
    return tuple("Evidence node" for _ in path_nodes)


def _display_label(node, width=17, max_lines=3):
    """Wrap labels consistently and cap unusually long source vocabulary names."""
    # Chemical names often contain a single long parenthesised or hyphenated
    # token. Allow that token to break so it remains inside its graph card.
    chunks = wrap(str(node), width=width, break_long_words=True, break_on_hyphens=True)
    if not chunks:
        return "Unnamed node"
    if len(chunks) > max_lines:
        chunks = chunks[:max_lines]
        chunks[-1] = chunks[-1].rstrip(".,;:") + "…"
    return "\n".join(chunks)


def draw_mechanism(path_nodes):
    """Return a clear left-to-right figure for a graph mechanism path.

    The standard path is Drug → Gene → Pathway → Disease. The function also
    renders shorter or longer paths without relying on words in node labels.
    """
    if len(path_nodes) < 2:
        raise ValueError("A mechanism diagram requires at least two path nodes.")

    roles = _roles_for_path(path_nodes)
    node_count = len(path_nodes)
    card_width = 2.3
    gap = 0.78
    figure_width = max(10.5, node_count * (card_width + gap) + 0.7)

    fig, ax = plt.subplots(figsize=(figure_width, 4.25), dpi=160)
    fig.patch.set_facecolor("#0B1120")
    ax.set_facecolor("#0B1120")
    ax.set_xlim(-0.25, node_count * (card_width + gap) - gap + 0.25)
    ax.set_ylim(-0.1, 4.25)
    ax.axis("off")

    # Quiet header establishes what the diagram represents without competing
    # with the surrounding Streamlit card title.
    ax.text(
        0,
        3.94,
        "GRAPH-SUPPORTED MECHANISM",
        color="#94A3B8",
        fontsize=8,
        fontweight="bold",
        va="center",
    )
    ax.plot([0, 1.15], [3.7, 3.7], color="#2563EB", linewidth=2.2, solid_capstyle="round")

    centers = []
    for index, (node, role) in enumerate(zip(path_nodes, roles)):
        style = ROLE_STYLE.get(role, {"accent": "#CBD5E1", "fill": "#1E293B", "edge": "#475569"})
        x = index * (card_width + gap)
        centers.append(x + card_width / 2)

        card = FancyBboxPatch(
            (x, 0.78), card_width, 2.28,
            boxstyle="round,pad=0.035,rounding_size=0.15",
            facecolor=style["fill"], edgecolor=style["edge"], linewidth=1.2,
        )
        ax.add_patch(card)
        ax.plot([x + 0.16, x + card_width - 0.16], [2.91, 2.91], color=style["accent"], linewidth=2.6, solid_capstyle="round")

        ax.text(x + 0.18, 2.62, f"{index + 1:02d}", color=style["accent"], fontsize=8, fontweight="bold", va="center")
        ax.text(x + 0.55, 2.62, role.upper(), color="#B8C7DC", fontsize=7.6, fontweight="bold", va="center")
        ax.text(
            x + card_width / 2, 1.78, _display_label(node),
            color="#F8FAFC", fontsize=10.2, fontweight="semibold",
            ha="center", va="center", linespacing=1.35,
        )

    # Put arrows between cards rather than behind them; labels identify each
    # relation while remaining readable at Streamlit's usual column width.
    for index in range(node_count - 1):
        left = centers[index] + card_width / 2 + 0.05
        right = centers[index + 1] - card_width / 2 - 0.05
        arrow = FancyArrowPatch(
            (left, 1.92), (right, 1.92), arrowstyle="-|>", mutation_scale=14,
            linewidth=1.5, color="#7185A3", shrinkA=0, shrinkB=0,
        )
        ax.add_patch(arrow)
        relation = RELATIONSHIPS[index] if index < len(RELATIONSHIPS) else "supports"
        ax.text((left + right) / 2, 2.2, relation, color="#94A3B8", fontsize=6.8, ha="center", va="bottom")

    ax.text(
        0, 0.28,
        "Directional links show the selected graph path; they indicate association, not causality.",
        color="#7185A3", fontsize=7.6, va="center",
    )
    fig.subplots_adjust(left=0.025, right=0.975, top=0.95, bottom=0.08)
    return fig
