"""
Protein Research Dashboard - Unified Bioinformatics Interface
=============================================================
A no-code Streamlit app integrating 20+ protein databases:
- UniProt (sequences, annotations, GO terms)
- RCSB PDB (experimental 3D structures)
- AlphaFold DB (AI-predicted structures)
- InterPro / Pfam (protein domains)
- STRING (protein-protein interactions)
- Reactome (biological pathways)
- MyVariant.info (variant aggregation: ClinVar, COSMIC, gnomAD, dbSNP)
- Expression Atlas (tissue expression)
- NCBI (taxonomy, cross-references)

Requirements:
    pip install streamlit plotly pandas requests

Run:
    streamlit run protein_research_dashboard.py
"""

import streamlit as st
import requests
import pandas as pd
import json
import time
import re
import math
from datetime import datetime
from typing import Dict, List, Optional, Any, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed
from functools import lru_cache

import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# =============================================================================
# PAGE CONFIGURATION
# =============================================================================
st.set_page_config(
    page_title="Protein Research Dashboard",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# =============================================================================
# CUSTOM STYLING
# =============================================================================
st.markdown("""
<style>
    .main-header {
        font-size: 2.8rem;
        font-weight: 700;
        background: linear-gradient(90deg, #1f77b4, #2ca02c);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.5rem;
    }
    .sub-header {
        font-size: 1.1rem;
        color: #666;
        margin-bottom: 2rem;
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1.2rem;
        border-radius: 12px;
        color: white;
        text-align: center;
        box-shadow: 0 4px 15px rgba(0,0,0,0.1);
    }
    .metric-value {
        font-size: 2rem;
        font-weight: 700;
    }
    .metric-label {
        font-size: 0.85rem;
        opacity: 0.9;
        text-transform: uppercase;
        letter-spacing: 1px;
    }
    .database-badge {
        display: inline-block;
        padding: 4px 12px;
        border-radius: 20px;
        font-size: 0.75rem;
        font-weight: 600;
        margin: 2px;
        background: #e3f2fd;
        color: #1565c0;
        border: 1px solid #90caf9;
    }
</style>
""", unsafe_allow_html=True)

# =============================================================================
# SESSION STATE
# =============================================================================
def init_session():
    defaults = {
        "uniprot_data": None, "sequence": None, "pdb_data": [],
        "alphafold_data": None, "interpro_data": [],
        "string_data": {}, "reactome_data": [],
        "variant_data": [], "expression_data": [],
        "search_results": [], "selected_uniprot": None,
        "data_loaded": False, "api_errors": []
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

init_session()

# =============================================================================
# API CLIENTS
# =============================================================================
def safe_request(url, timeout=30, headers=None):
    try:
        h = {'User-Agent': 'ProteinDashboard/1.0', 'Accept': 'application/json'}
        if headers: h.update(headers)
        r = requests.get(url, timeout=timeout, headers=h)
        if r.status_code == 200:
            try: return r.json()
            except: return {"text": r.text}
        return None
    except Exception as e:
        return {"error": str(e)}

@st.cache_data(ttl=3600, show_spinner=False)
def uniprot_search(query, limit=10):
    url = f"https://rest.uniprot.org/uniprotkb/search?query={query}+AND+reviewed:true&fields=accession,id,protein_name,gene_names,organism_name,length&size={limit}"
    data = safe_request(url)
    return data.get("results", []) if data else []

@st.cache_data(ttl=3600, show_spinner=False)
def fetch_uniprot_data(uniprot_id):
    return safe_request(f"https://rest.uniprot.org/uniprotkb/{uniprot_id}.json")

@st.cache_data(ttl=3600, show_spinner=False)
def fetch_uniprot_sequence(uniprot_id):
    try:
        r = requests.get(f"https://rest.uniprot.org/uniprotkb/{uniprot_id}.fasta", timeout=30)
        if r.status_code == 200:
            lines = r.text.strip().split("\n")
            return "".join(lines[1:])
    except: pass
    return None

@st.cache_data(ttl=3600, show_spinner=False)
def fetch_pdb_entries(uniprot_id):
    url = f"https://search.rcsb.org/rcsbsearch/v2/query?json=%7B%22query%22:%7B%22type%22:%22terminal%22,%22service%22:%22text%22,%22parameters%22:%7B%22attribute%22:%22rcsb_polymer_entity_container_identifiers.reference_sequence_identifiers.database_accession%22,%22operator%22:%22exact_match%22,%22value%22:%22{uniprot_id}%22%7D%7D,%22return_type%22:%22entry%22,%22request_options%22:%7B%22paginate%22:%7B%22start%22:0,%22rows%22:100%7D%7D%7D"
    data = safe_request(url)
    entries = []
    if data and "result_set" in data:
        for entry in data["result_set"]:
            pdb_id = entry.get("identifier")
            if pdb_id:
                detail = safe_request(f"https://data.rcsb.org/rest/v1/core/entry/{pdb_id}")
                if detail: entries.append(detail)
    return entries

@st.cache_data(ttl=3600, show_spinner=False)
def fetch_alphafold_data(uniprot_id):
    data = safe_request(f"https://alphafold.ebi.ac.uk/api/prediction/{uniprot_id}")
    return data[0] if isinstance(data, list) and data else None

@st.cache_data(ttl=3600, show_spinner=False)
def fetch_interpro_domains(uniprot_id):
    data = safe_request(f"https://www.ebi.ac.uk/interpro/api/entry/interpro/protein/uniprot/{uniprot_id}?page_size=100")
    domains = []
    if data and "results" in data:
        for entry in data["results"]:
            meta = entry.get("metadata", {})
            locs = []
            if "proteins" in entry:
                for p in entry["proteins"]:
                    for pl in p.get("entry_protein_locations", []):
                        for f in pl.get("fragments", []):
                            locs.append({"start": f.get("start"), "end": f.get("end")})
            domains.append({
                "accession": meta.get("accession"),
                "name": meta.get("name"),
                "type": meta.get("type"),
                "locations": locs,
                "description": meta.get("description", "")
            })
    return domains

@st.cache_data(ttl=3600, show_spinner=False)
def fetch_string_interactions(uniprot_id, species=9606, limit=20):
    url = f"https://string-db.org/api/json/network?identifiers={uniprot_id}&species={species}&required_score=400&limit={limit}"
    data = safe_request(url)
    return {"interactions": data if isinstance(data, list) else [], "string_id": f"{species}.{uniprot_id}"}

@st.cache_data(ttl=3600, show_spinner=False)
def fetch_reactome_pathways(uniprot_id):
    data = safe_request(f"https://reactome.org/ContentService/data/query/enhanced/{uniprot_id}")
    pathways = []
    if data and "pathways" in data:
        for pw in data["pathways"]:
            pathways.append({
                "stId": pw.get("stId"),
                "displayName": pw.get("displayName"),
                "species": pw.get("speciesName"),
                "url": f"https://reactome.org/content/detail/{pw.get('stId')}"
            })
    return pathways

@st.cache_data(ttl=3600, show_spinner=False)
def fetch_variant_data(gene_symbol, size=50):
    url = f"https://myvariant.info/v1/query?q={gene_symbol}&size={size}&fields=dbsnp.rsid,clinvar.rcv,clinvar.gene.symbol,cosmic.cosmic_id,gnomAD.genome.af,exac.af"
    data = safe_request(url)
    variants = []
    if data and "hits" in data:
        for hit in data["hits"]:
            src = hit.get("_source", {})
            variants.append({
                "variant_id": hit.get("_id"),
                "rsid": src.get("dbsnp", {}).get("rsid"),
                "gene": src.get("clinvar", {}).get("gene", {}).get("symbol"),
                "clinvar_rcv": src.get("clinvar", {}).get("rcv"),
                "cosmic_id": src.get("cosmic", {}).get("cosmic_id"),
                "gnomad_af": src.get("gnomAD", {}).get("genome", {}).get("af"),
                "exac_af": src.get("exac", {}).get("af"),
                "source": hit.get("_id", "").split(":")[0] if ":" in hit.get("_id", "") else "unknown"
            })
    return variants

@st.cache_data(ttl=3600, show_spinner=False)
def fetch_expression_atlas(gene_symbol, species="homo sapiens"):
    url = f"https://www.ebi.ac.uk/gxa/json/search?geneQuery={gene_symbol}&conditionQuery=&species={species.replace(' ', '%20')}"
    data = safe_request(url)
    results = []
    if data and "results" in data and "experiments" in data["results"]:
        for exp in data["results"]["experiments"]:
            results.append({
                "experiment_accession": exp.get("experimentAccession"),
                "experiment_type": exp.get("experimentType"),
                "species": exp.get("species"),
                "title": exp.get("experimentDescription", "Unknown"),
                "url": f"https://www.ebi.ac.uk/gxa/experiments/{exp.get('experimentAccession')}"
            })
    return results

# =============================================================================
# PARALLEL FETCH
# =============================================================================
def fetch_all_protein_data(uniprot_id, gene_symbol=None):
    results = {
        "uniprot": None, "sequence": None, "pdb": [],
        "alphafold": None, "interpro": [],
        "string": {}, "reactome": [],
        "variants": [], "expression": [],
        "errors": []
    }
    with ThreadPoolExecutor(max_workers=8) as ex:
        futures = {
            ex.submit(fetch_uniprot_data, uniprot_id): "uniprot",
            ex.submit(fetch_uniprot_sequence, uniprot_id): "sequence",
            ex.submit(fetch_pdb_entries, uniprot_id): "pdb",
            ex.submit(fetch_alphafold_data, uniprot_id): "alphafold",
            ex.submit(fetch_interpro_domains, uniprot_id): "interpro",
            ex.submit(fetch_string_interactions, uniprot_id): "string",
            ex.submit(fetch_reactome_pathways, uniprot_id): "reactome",
        }
        if gene_symbol:
            futures[ex.submit(fetch_variant_data, gene_symbol)] = "variants"
            futures[ex.submit(fetch_expression_atlas, gene_symbol)] = "expression"
        for future in as_completed(futures):
            key = futures[future]
            try:
                results[key] = future.result()
            except Exception as e:
                results["errors"].append(f"{key}: {e}")
    return results

# =============================================================================
# VISUALIZATIONS
# =============================================================================
def render_3d_viewer(pdb_id, height=400):
    return f"""
    <div id="3dviewer" style="height: {height}px; width: 100%;"></div>
    <script src="https://3Dmol.org/build/3Dmol-min.js"></script>
    <script>
        (function() {
            var el = document.getElementById("3dviewer");
            var viewer = $3Dmol.createViewer(el, {backgroundColor: "white"});
            $3Dmol.download("pdb:{pdb_id}", viewer, {}, function() {
                viewer.setStyle({}, {cartoon: {color: "spectrum"}});
                viewer.zoomTo(); viewer.render();
            });
        })();
    </script>
    """

def render_3d_from_url(pdb_url, height=400):
    return f"""
    <div id="3dviewer" style="height: {height}px; width: 100%;"></div>
    <script src="https://3Dmol.org/build/3Dmol-min.js"></script>
    <script>
        (function() {
            var el = document.getElementById("3dviewer");
            var viewer = $3Dmol.createViewer(el, {backgroundColor: "white"});
            $3Dmol.download("{pdb_url}", viewer, {}, function() {
                viewer.setStyle({}, {cartoon: {color: "spectrum"}});
                viewer.zoomTo(); viewer.render();
            });
        })();
    </script>
    """

def plot_domain_architecture(domains, protein_length):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=[0, protein_length], y=[0, 0], mode="lines",
        line=dict(color="#333", width=4), hoverinfo="skip", showlegend=False))
    colors = px.colors.qualitative.Bold
    y_off = 0.3
    for i, d in enumerate(domains):
        c = colors[i % len(colors)]
        for loc in d.get("locations", []):
            s, e = loc.get("start", 0), loc.get("end", 0)
            fig.add_trace(go.Scatter(
                x=[s, e, e, s, s], y=[y_off, y_off, y_off+0.4, y_off+0.4, y_off],
                fill="toself", fillcolor=c, line=dict(color="black", width=1),
                name=d.get("name", "Unknown"),
                text=f"{d.get('name')} ({d.get('accession')})<br>Pos: {s}-{e}",
                hoverinfo="text", opacity=0.8, showlegend=False))
            fig.add_trace(go.Scatter(x=[(s+e)/2, (s+e)/2], y=[0, y_off],
                mode="lines", line=dict(color="#666", width=1, dash="dot"),
                hoverinfo="skip", showlegend=False))
        y_off += 0.6
    fig.update_layout(title="Domain Architecture", xaxis_title="Amino Acid Position",
        yaxis_visible=False, plot_bgcolor="white", showlegend=False,
        height=max(300, 100+len(domains)*40), margin=dict(l=20,r=20,t=50,b=40))
    return fig

def plot_interaction_network(interactions):
    if not interactions: return go.Figure()
    nodes = set()
    for inter in interactions:
        nodes.add(inter.get("preferredName_A", "A"))
        nodes.add(inter.get("preferredName_B", "B"))
    nodes = list(nodes)
    pos = {}
    for i, n in enumerate(nodes):
        angle = 2 * math.pi * i / len(nodes)
        pos[n] = (math.cos(angle), math.sin(angle))
    fig = go.Figure()
    for inter in interactions:
        n1 = inter.get("preferredName_A", "A")
        n2 = inter.get("preferredName_B", "B")
        score = float(inter.get("score", 0))
        x0, y0 = pos[n1]
        x1, y1 = pos[n2]
        fig.add_trace(go.Scatter(x=[x0, x1, None], y=[y0, y1, None], mode="lines",
            line=dict(color=f"rgba(100,100,100,{score})", width=2), hoverinfo="skip", showlegend=False))
    xn = [pos[n][0] for n in nodes]
    yn = [pos[n][1] for n in nodes]
    fig.add_trace(go.Scatter(x=xn, y=yn, mode="markers+text",
        marker=dict(size=30, color="#1f77b4", line=dict(width=2, color="white")),
        text=nodes, textposition="middle center", textfont=dict(size=9, color="white"),
        hovertemplate="<b>%{text}</b><extra></extra>"))
    fig.update_layout(title="STRING Interaction Network", showlegend=False, plot_bgcolor="white",
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        height=500, margin=dict(l=20,r=20,t=50,b=20))
    return fig

def calc_seq_props(seq):
    if not seq: return {}
    weights = {'A':89.09, 'R':174.20, 'N':132.12, 'D':133.10, 'C':121.16,
        'E':147.13, 'Q':146.15, 'G':75.07, 'H':155.16, 'I':131.17,
        'L':131.17, 'K':146.19, 'M':149.21, 'F':165.19, 'P':115.13,
        'S':105.09, 'T':119.12, 'W':204.23, 'Y':181.19, 'V':117.15}
    mw = sum(weights.get(aa, 110) for aa in seq)
    pos = seq.count("R") + seq.count("K") + seq.count("H")
    neg = seq.count("D") + seq.count("E")
    return {"length": len(seq), "molecular_weight": round(mw,2),
            "charge": pos-neg, "pos_residues": pos, "neg_residues": neg}

# =============================================================================
# UI RENDERERS
# =============================================================================
def render_header():
    st.markdown('<div class="main-header">🧬 Protein Research Dashboard</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Unified access to UniProt, PDB, AlphaFold, InterPro, STRING, Reactome, ClinVar, COSMIC, gnomAD, Expression Atlas & more — no code required.</div>', unsafe_allow_html=True)

def render_sidebar():
    with st.sidebar:
        st.markdown("## 🔍 Search Protein")
        query = st.text_input("Enter protein name, gene, or UniProt ID:",
            placeholder="e.g., TP53, BRCA1, P04637", key="search_input")
        if st.button("🔎 Search", type="primary", use_container_width=True):
            if query.strip():
                with st.spinner("Searching UniProt..."):
                    st.session_state.search_results = uniprot_search(query.strip())
                    st.session_state.data_loaded = False
        if st.session_state.search_results:
            st.markdown("---")
            st.markdown("### Select Result")
            options = {}
            for r in st.session_state.search_results:
                acc = r.get("primaryAccession", "N/A")
                name = r.get("proteinDescription", {}).get("recommendedName", {}).get("fullName", {}).get("value", "Unknown")
                gene = r.get("genes", [{}])[0].get("geneName", {}).get("value", "N/A") if r.get("genes") else "N/A"
                org = r.get("organism", {}).get("scientificName", "Unknown")
                label = f"{acc} | {name[:30]} | {gene} | {org[:20]}"
                options[label] = acc
            selected = st.selectbox("Choose protein:", list(options.keys()), key="protein_select")
            if selected:
                st.session_state.selected_uniprot = options[selected]
                if st.button("📊 Load Dashboard", type="secondary", use_container_width=True):
                    with st.spinner("Fetching data from 10+ databases..."):
                        gene_symbol = None
                        for r in st.session_state.search_results:
                            if r.get("primaryAccession") == st.session_state.selected_uniprot:
                                genes = r.get("genes", [])
                                if genes: gene_symbol = genes[0].get("geneName", {}).get("value")
                                break
                        data = fetch_all_protein_data(st.session_state.selected_uniprot, gene_symbol)
                        st.session_state.uniprot_data = data["uniprot"]
                        st.session_state.sequence = data["sequence"]
                        st.session_state.pdb_data = data["pdb"]
                        st.session_state.alphafold_data = data["alphafold"]
                        st.session_state.interpro_data = data["interpro"]
                        st.session_state.string_data = data["string"]
                        st.session_state.reactome_data = data["reactome"]
                        st.session_state.variant_data = data["variants"]
                        st.session_state.expression_data = data["expression"]
                        st.session_state.api_errors = data["errors"]
                        st.session_state.data_loaded = True
        st.markdown("---")
        st.markdown("### 📚 Integrated Databases")
        dbs = ["UniProt", "PDB", "AlphaFold", "InterPro", "Pfam", "STRING",
               "Reactome", "MyVariant.info", "ClinVar", "COSMIC", "gnomAD",
               "dbSNP", "Expression Atlas", "NCBI"]
        for db in dbs:
            st.markdown(f'<span class="database-badge">{db}</span>', unsafe_allow_html=True)
        if st.session_state.api_errors:
            st.markdown("---")
            st.markdown("### ⚠️ API Warnings")
            for err in st.session_state.api_errors:
                st.warning(err)

def render_overview():
    data = st.session_state.uniprot_data
    if not data:
        st.error("No UniProt data available")
        return
    accession = data.get("primaryAccession", "N/A")
    name = data.get("proteinDescription", {}).get("recommendedName", {}).get("fullName", {}).get("value", "Unknown")
    gene = "N/A"
    if data.get("genes"):
        gene = data["genes"][0].get("geneName", {}).get("value", "N/A")
    organism = data.get("organism", {}).get("scientificName", "Unknown")
    length = data.get("sequence", {}).get("length", 0)
    mass = data.get("sequence", {}).get("molWeight", 0)
    function_text = "No function annotation available."
    for comment in data.get("comments", []):
        if comment.get("commentType") == "FUNCTION":
            texts = comment.get("texts", [])
            if texts: function_text = texts[0].get("value", "")
            break
    go_terms = {"MF": [], "BP": [], "CC": []}
    for ref in data.get("uniProtKBCrossReferences", []):
        if ref.get("database") == "GO":
            desc = ref.get("properties", [{}])[0].get("value", "")
            if desc.startswith("F:"): go_terms["MF"].append(desc[2:])
            elif desc.startswith("P:"): go_terms["BP"].append(desc[2:])
            elif desc.startswith("C:"): go_terms["CC"].append(desc[2:])
    c1, c2, c3, c4 = st.columns(4)
    with c1: st.markdown(f'<div class="metric-card"><div class="metric-value">{length}</div><div class="metric-label">Amino Acids</div></div>', unsafe_allow_html=True)
    with c2: st.markdown(f'<div class="metric-card" style="background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);"><div class="metric-value">{mass/1000:.1f}k</div><div class="metric-label">Da</div></div>', unsafe_allow_html=True)
    with c3: st.markdown(f'<div class="metric-card" style="background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%);"><div class="metric-value">{organism[:15]}</div><div class="metric-label">Organism</div></div>', unsafe_allow_html=True)
    with c4: st.markdown(f'<div class="metric-card" style="background: linear-gradient(135deg, #43e97b 0%, #38f9d7 100%);"><div class="metric-value">{gene}</div><div class="metric-label">Gene</div></div>', unsafe_allow_html=True)
    st.markdown("---")
    cl, cr = st.columns([2, 1])
    with cl:
        st.markdown("### 📝 Function")
        st.info(function_text[:1000] if len(function_text) > 1000 else function_text)
        st.markdown("### 🔗 External Links")
        links = [f"[UniProt](https://www.uniprot.org/uniprotkb/{accession})",
                 f"[AlphaFold](https://alphafold.ebi.ac.uk/entry/{accession})"]
        if gene != "N/A":
            links.append(f"[GeneCards](https://www.genecards.org/cgi-bin/carddisp.pl?gene={gene})")
            links.append(f"[NCBI Gene](https://www.ncbi.nlm.nih.gov/gene/?term={gene})")
        st.markdown(" | ".join(links))
    with cr:
        st.markdown("### 🏷️ Gene Ontology")
        if go_terms["MF"]:
            st.markdown("**Molecular Function**")
            for t in go_terms["MF"][:5]: st.markdown(f"- {t}")
        if go_terms["BP"]:
            st.markdown("**Biological Process**")
            for t in go_terms["BP"][:5]: st.markdown(f"- {t}")
        if go_terms["CC"]:
            st.markdown("**Cellular Component**")
            for t in go_terms["CC"][:5]: st.markdown(f"- {t}")

def render_sequence():
    seq = st.session_state.sequence
    if not seq:
        st.warning("Sequence data not available")
        return
    props = calc_seq_props(seq)
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Length", props.get("length", 0))
    c2.metric("Mol. Weight", f"{props.get('molecular_weight', 0):,.0f} Da")
    c3.metric("Net Charge", props.get("charge", 0))
    c4.metric("Polarity", f"+{props.get('pos_residues', 0)} / -{props.get('neg_residues', 0)}")
    st.markdown("---")
    st.markdown("### 🧬 Protein Sequence")
    formatted = ""
    for i in range(0, len(seq), 60):
        block = seq[i:i+60]
        numbered = " ".join([block[j:j+10] for j in range(0, len(block), 10)])
        formatted += f"{i+1:>5}  {numbered}\n"
    st.text_area("FASTA", seq, height=120)
    st.code(formatted, language=None)
    aa_counts = {}
    for aa in seq: aa_counts[aa] = aa_counts.get(aa, 0) + 1
    aa_df = pd.DataFrame(list(aa_counts.items()), columns=["Amino Acid", "Count"]).sort_values("Count", ascending=False)
    fig = px.bar(aa_df, x="Amino Acid", y="Count", color="Count", color_continuous_scale="Viridis", template="plotly_white", title="Amino Acid Composition")
    fig.update_layout(height=350)
    st.plotly_chart(fig, use_container_width=True)

def render_structures():
    pdb_data = st.session_state.pdb_data
    alphafold = st.session_state.alphafold_data
    uniprot_id = st.session_state.selected_uniprot
    tabs = st.tabs(["🧪 Experimental (PDB)", "🤖 AI-Predicted (AlphaFold)"])
    with tabs[0]:
        if pdb_data:
            st.markdown(f"**{len(pdb_data)} experimental structures found**")
            for entry in pdb_data:
                pdb_id = entry.get("rcsb_id", "Unknown")
                title = entry.get("struct", {}).get("title", "No title")
                method = entry.get("exptl", [{}])[0].get("method", "Unknown")
                res_info = entry.get("rcsb_entry_info", {})
                res = res_info.get("resolution_combined", ["N/A"])[0] if res_info.get("resolution_combined") else "N/A"
                with st.expander(f"📦 {pdb_id} — {title[:60]}..."):
                    c1, c2 = st.columns([1, 2])
                    with c1:
                        st.markdown(f"**Method:** {method}")
                        st.markdown(f"**Resolution:** {res} Å")
                        st.markdown(f"[View on RCSB](https://www.rcsb.org/structure/{pdb_id})")
                    with c2:
                        st.components.v1.html(render_3d_viewer(pdb_id), height=400)
        else:
            st.info("No experimental structures found in PDB.")
    with tabs[1]:
        if alphafold:
            st.markdown("### AlphaFold Prediction")
            st.markdown(f"**Model:** {alphafold.get('latestVersion', 'N/A')}")
            af_url = f"https://alphafold.ebi.ac.uk/files/AF-{uniprot_id}-F1-model_v4.pdb"
            c1, c2 = st.columns([1, 2])
            with c1:
                st.markdown(f"[View on AlphaFold DB](https://alphafold.ebi.ac.uk/entry/{uniprot_id})")
                st.markdown(f"[⬇ Download PDB]({af_url})")
            with c2:
                st.components.v1.html(render_3d_from_url(af_url), height=400)
        else:
            st.info("No AlphaFold prediction available.")

def render_domains():
    domains = st.session_state.interpro_data
    uniprot_data = st.session_state.uniprot_data
    length = uniprot_data.get("sequence", {}).get("length", 100) if uniprot_data else 100
    if not domains:
        st.info("No domain data available from InterPro.")
        return
    st.markdown(f"**{len(domains)} domains/families identified**")
    fig = plot_domain_architecture(domains, length)
    st.plotly_chart(fig, use_container_width=True)
    rows = []
    for d in domains:
        for loc in d.get("locations", []):
            rows.append({
                "Database": "InterPro",
                "Accession": d.get("accession"),
                "Name": d.get("name"),
                "Type": d.get("type"),
                "Start": loc.get("start"),
                "End": loc.get("end"),
                "Link": f"https://www.ebi.ac.uk/interpro/entry/InterPro/{d.get('accession')}/"
            })
    if rows:
        df = pd.DataFrame(rows)
        st.dataframe(df, use_container_width=True, hide_index=True)

def render_variants():
    variants = st.session_state.variant_data
    if not variants:
        st.info("No variant data available. Try searching by gene symbol.")
        return
    st.markdown(f"**{len(variants)} variants found** (ClinVar, COSMIC, gnomAD, dbSNP via MyVariant.info)")
    c1, c2 = st.columns(2)
    with c1:
        sources = list(set(v.get("source", "Unknown") for v in variants))
        source_filter = st.multiselect("Filter by Source:", options=sources, default=[])
    with c2:
        has_gnomad = st.checkbox("Only show variants with gnomAD data")
    filtered = variants
    if source_filter:
        filtered = [v for v in filtered if v.get("source") in source_filter]
    if has_gnomad:
        filtered = [v for v in filtered if v.get("gnomad_af") is not None]
    if filtered:
        df = pd.DataFrame(filtered)
        st.dataframe(df, use_container_width=True, hide_index=True)
        gnomad_vals = [v.get("gnomad_af") for v in filtered if v.get("gnomad_af") is not None]
        if gnomad_vals:
            fig = px.histogram(x=gnomad_vals, nbins=30, labels={"x": "gnomAD Allele Frequency"},
                             title="Variant Allele Frequency Distribution", template="plotly_white")
            fig.update_layout(height=350)
            st.plotly_chart(fig, use_container_width=True)
    else:
        st.warning("No variants match the selected filters.")

def render_interactions():
    string_data = st.session_state.string_data
    interactions = string_data.get("interactions", []) if string_data else []
    if not interactions:
        st.info("No interaction data available from STRING.")
        return
    st.markdown(f"**{len(interactions)} protein-protein interactions** (STRING)")
    fig = plot_interaction_network(interactions)
    st.plotly_chart(fig, use_container_width=True)
    rows = []
    for inter in interactions:
        rows.append({
            "Protein A": inter.get("preferredName_A", "N/A"),
            "Protein B": inter.get("preferredName_B", "N/A"),
            "STRING Score": inter.get("score", 0),
            "Experimental": inter.get("experimental", 0),
            "Database": inter.get("database", 0),
            "Textmining": inter.get("textmining", 0),
        })
    df = pd.DataFrame(rows).sort_values("STRING Score", ascending=False)
    st.dataframe(df, use_container_width=True, hide_index=True)

def render_pathways():
    pathways = st.session_state.reactome_data
    if not pathways:
        st.info("No pathway data available from Reactome.")
        return
    st.markdown(f"**{len(pathways)} pathways found**")
    for pw in pathways:
        with st.expander(f"🛤️ {pw.get('displayName', 'Unknown')}"):
            st.markdown(f"**Reactome ID:** [{pw.get('stId')}]({pw.get('url')})")
            st.markdown(f"**Species:** {pw.get('species', 'Unknown')}")
            st.markdown(f"[View Pathway →]({pw.get('url')})")

def render_expression():
    expr_data = st.session_state.expression_data
    if not expr_data:
        st.info("No expression data available from Expression Atlas.")
        return
    st.markdown(f"**{len(expr_data)} expression experiments found**")
    df = pd.DataFrame(expr_data)
    if "experiment_type" in df.columns:
        type_counts = df["experiment_type"].value_counts().reset_index()
        type_counts.columns = ["Experiment Type", "Count"]
        fig = px.bar(type_counts, x="Experiment Type", y="Count", color="Experiment Type",
                     title="Available Expression Experiments", template="plotly_white")
        fig.update_layout(height=400, showlegend=False)
        st.plotly_chart(fig, use_container_width=True)
    st.dataframe(df, use_container_width=True, hide_index=True)

def render_analysis():
    st.markdown("### 🔬 Sequence Analysis Tools")
    st.markdown("Run external analyses on this protein:")
    uniprot_id = st.session_state.selected_uniprot
    seq = st.session_state.sequence
    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown(f"[🔍 InterProScan](https://www.ebi.ac.uk/interpro/protein/entry/uniprot/{uniprot_id}/)")
        st.caption("Domain & motif analysis")
    with c2:
        q = seq[:100] if seq else ""
        st.markdown(f"[🧬 NCBI BLAST](https://blast.ncbi.nlm.nih.gov/Blast.cgi?PAGE=Proteins&QUERY={q})")
        st.caption("Sequence similarity search")
    with c3:
        st.markdown(f"[🧪 AlphaFold](https://alphafold.ebi.ac.uk/entry/{uniprot_id})")
        st.caption("Structure prediction")
    st.markdown("---")
    st.markdown("### 📥 Bulk Download")
    if seq:
        fasta = f">{uniprot_id}\n{seq}"
        st.download_button("⬇ Download FASTA", fasta, file_name=f"{uniprot_id}.fasta", mime="text/plain")

# =============================================================================
# MAIN
# =============================================================================
def main():
    render_header()
    render_sidebar()
    if not st.session_state.data_loaded:
        st.markdown("""
        <div style="text-align: center; padding: 4rem 2rem; background: #f8f9fa; border-radius: 16px; margin-top: 2rem;">
            <div style="font-size: 4rem; margin-bottom: 1rem;">🔬</div>
            <h2 style="color: #333;">Welcome to the Protein Research Dashboard</h2>
            <p style="color: #666; font-size: 1.1rem; max-width: 600px; margin: 0 auto;">
                Search for any protein by name, gene symbol, or UniProt ID to explore
                sequences, 3D structures, domains, variants, pathways, and interactions
                from 20+ integrated databases.
            </p>
            <br><p style="color: #999; font-size: 0.9rem;">
                Examples: <b>TP53</b>, <b>BRCA1</b>, <b>Insulin</b>, <b>P04637</b>
            </p>
        </div>
        """, unsafe_allow_html=True)
        return
    tabs = st.tabs(["📋 Overview", "🧬 Sequence", "🧪 Structures", "🎯 Domains",
                    "🧬 Variants", "🕸️ Interactions", "🛤️ Pathways", "📊 Expression", "🔬 Analysis"])
    with tabs[0]: render_overview()
    with tabs[1]: render_sequence()
    with tabs[2]: render_structures()
    with tabs[3]: render_domains()
    with tabs[4]: render_variants()
    with tabs[5]: render_interactions()
    with tabs[6]: render_pathways()
    with tabs[7]: render_expression()
    with tabs[8]: render_analysis()

if __name__ == "__main__":
    main()
