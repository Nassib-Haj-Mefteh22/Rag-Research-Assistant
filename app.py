"""Step 12.2: Streamlit demo. Run: .\.venv\Scripts\streamlit.exe run app.py

WHY a thin page: all RAG logic lives in src/rag (answer, format_sources); this file only draws
the question box, the answer, the sources and the retrieved passages. Same pipeline as the CLI.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "src"))  # so `import rag` works without PYTHONPATH

import streamlit as st  # noqa: E402

from rag.cli import format_sources  # noqa: E402
from rag.generate import answer  # noqa: E402
from rag.tools import DEFAULT_MODE, load_titles  # noqa: E402

MODES = ["hybrid", "dense", "hybrid+rerank", "dense+rerank"]


@st.cache_data
def titles() -> dict[str, str]:
    """Read metadata.json once per session instead of on every click."""
    return load_titles()


st.set_page_config(page_title="RAG Assistant", page_icon="📚")
st.title("📚 Ask the RAG & LLM-evaluation papers")
st.caption("50 arXiv papers (2025+). Answers cite sources (paper, page) or say they don't know.")

question = st.text_input("Question", placeholder="e.g. What is faithfulness in RAG evaluation?")
col_mode, col_k = st.columns(2)
mode = col_mode.selectbox("Retrieval mode", MODES, index=MODES.index(DEFAULT_MODE))
k = col_k.slider("Passages given to the model (k)", min_value=2, max_value=8, value=4)

if st.button("Ask", type="primary", disabled=not question.strip()):
    with st.spinner("Retrieving passages and generating the answer..."):
        result = answer(question, k=k, mode=mode)

    st.subheader("Answer")
    st.write(result["answer"])

    sources = format_sources(result["sources"], titles())
    if sources:
        st.subheader("Sources")
        for line in sources:
            st.markdown(f"- {line}")
    if result["invalid_citations"]:
        st.warning(f"The model cited passages that do not exist: {result['invalid_citations']}")

    with st.expander(f"Retrieved passages ({len(result['chunks'])})"):
        for chunk in result["chunks"]:
            title = titles().get(chunk["doc_id"], chunk["doc_id"])
            st.markdown(f"**[{chunk['rank']}] {title}**, p.{chunk['page']}")
            st.text(chunk["text"])
