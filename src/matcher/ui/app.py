from __future__ import annotations

import streamlit as st

from matcher.pipeline.score import score_resume


def main() -> None:
    st.set_page_config(page_title="Resume-JD Matcher", layout="wide")
    st.title("Resume to Job Description Matcher")
    st.caption(
        "Two-stage retrieval: bi-encoder + cross-encoder. "
        "Paste a resume to find the best-fit roles."
    )

    with st.sidebar:
        top_k = st.slider("Top-K results", 1, 10, 5)
        use_rerank = st.checkbox("Use cross-encoder rerank", value=True)

    resume = st.text_area("Resume text", height=300, placeholder="Paste your resume here...")

    if st.button("Find matching roles", type="primary"):
        if not resume.strip():
            st.warning("Paste a resume first.")
            return
        with st.spinner("Embedding and scoring..."):
            hits = score_resume(resume, top_k=top_k, use_rerank=use_rerank)
        for h in hits:
            with st.container(border=True):
                cols = st.columns([3, 1])
                cols[0].markdown(f"**{h['title']}** &nbsp; `{h['job_id']}`")
                cols[1].metric("Score", f"{h['score']:.3f}")
                if h["matched_skills"]:
                    st.markdown("**Matched skills:** " + ", ".join(h["matched_skills"]))
                if h["missing_skills"]:
                    st.markdown("**Gap:** " + ", ".join(h["missing_skills"]))


if __name__ == "__main__":
    main()
