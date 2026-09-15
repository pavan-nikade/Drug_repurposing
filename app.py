from pathlib import Path
import sys
import streamlit as st
import pandas as pd

# Add the project directory containing 'src' to the system path to handle relative module routing safely
sys.path.append(str(Path(__file__).resolve().parent))

# Import your optimized discovery and visualization modules safely
from discover import G, discover, label, type_of
from network_view import draw_mechanism  

st.set_page_config(
    page_title="Explainable Drug Discovery",
    page_icon="🧬",
    layout="wide"
)

# ----------------------------
# Header
# ----------------------------

st.markdown("""
# 🧬 Explainable Drug Target Discovery
### Knowledge Graph + Machine Learning + Mechanistic Reasoning
""")

st.caption("Demo prototype for explainable drug repurposing and target discovery")

# ----------------------------
# Drug selector
# ----------------------------

@st.cache_data(show_spinner=False)
def get_drug_options():
    """List every drug available in the loaded knowledge graph."""
    return sorted(
        [(label(node), node.removeprefix("drug:")) for node in G.nodes if type_of(node) == "drug"],
        key=lambda option: option[0].lower(),
    )


drug_options = get_drug_options()


def drug_option_label(option):
    """Keep duplicate chemical names distinguishable in the selector."""
    name, chemical_id = option
    return f"{name}  ·  {chemical_id}"

selected = st.selectbox(
    "Select a drug",
    options=drug_options,
    index=None,
    placeholder="Search or choose a drug from the knowledge graph…",
    format_func=drug_option_label,
    help=f"{len(drug_options):,} drugs are available. Type to search by name or chemical ID.",
)

# ----------------------------
# Run discovery
# ----------------------------

# ----------------------------
# Run discovery
# ----------------------------

if st.button("🔍 Run Discovery", use_container_width=True, disabled=selected is None):

    with st.spinner("Analyzing biomedical knowledge graph..."):
        selected_name, selected_drug_id = selected
        # Unpack the two separate tiers returned by the updated backend function
        high_res, low_res = discover(selected_drug_id, top_k=5)

    # ---------- Dynamic Tier Allocation Guardrails ----------
    if not high_res:
        st.warning(
            "⚠️ No high-confidence associations met the strict network thresholds for this compound. "
            "Displaying exploratory, lower-tier structural hypotheses instead."
        )
        results = low_res[:5]
    else:
        results = high_res[:5]

    if not results:
        st.error("No connectivity mappings could be extracted for this compound profile.")
    else:
        st.success("Discovery completed")

        # Summary metrics
        c1, c2, c3 = st.columns(3)
        c1.metric("Diseases Found", len(results))
        c2.metric("Best Score", f"{results[0]['final']:.3f}")
        c3.metric("Evidence Paths", results[0]['paths'])

        st.divider()

        # ----------------------------
        # Top-5 cards
        # ----------------------------
        for i, r in enumerate(results, 1):

            with st.container(border=True):
                c1, c2, c3 = st.columns([4, 2, 1])

                with c1:
                    st.markdown(f"## {i}. {r['disease']}")

                with c2:
                    # NEW: Renders visual tier status badge directly on top of card blocks
                    st.markdown(f"### {r['tier']}")

                with c3:
                    st.markdown(f"### {r['final']:.3f}")

                # Confidence bar
                st.progress(min(1.0, r['final']))

                m1, m2, m3 = st.columns(3)
                m1.metric("Predictive", f"{r['predictive']:.3f}")
                m2.metric("TMS", f"{r['tms']:.3f}")
                m3.metric("Paths", r['paths'])

                # Visual Column Splits
                layout_col1, layout_col2 = st.columns([1, 1.2])

                with layout_col1:
                    st.markdown("### 🧭 Mechanistic Path")
                    st.code(r['best_path'])

                    st.markdown("### 🧪 Supporting Pathways")
                    clean_pathway_names = [p[0] if isinstance(p, tuple) else str(p) for p in r['top_pathways']]
                    chips = " ".join([f"`{p}`" for p in clean_pathway_names[:3]])
                    st.markdown(chips)

                with layout_col2:
                    st.markdown("### 🔗 Discovered Mechanism Graph Pipeline")
                    path_nodes = r['best_path'].split(' → ')
                    st.pyplot(draw_mechanism(path_nodes))

                with st.expander("📝 Plain-language explanation"):
                    st.write(r['explanation'])

                st.markdown("---")

# ----------------------------
# Footer
# ----------------------------

st.divider()

st.caption(
    "This system generates graph-based research hypotheses and does not provide clinical recommendations." 
)
