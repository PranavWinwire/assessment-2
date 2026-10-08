from datetime import datetime, timezone
from uuid import uuid4
import json
import mimetypes

import httpx
import pandas as pd
import plotly.express as px
import streamlit as st


API = "/api/v1"
METRICS = ["faithfulness", "answer_relevancy", "context_precision", "context_recall"]
LABELS = ["Faithfulness", "Answer relevancy", "Context precision", "Context recall"]

st.set_page_config(page_title="HR Policy Assistant", page_icon="✦", layout="wide")
st.markdown("""
<style>
.stApp { --text-color:#17243a; --background-color:#f4f7fb; color:#17243a; background:#f4f7fb; }
[data-testid="stMain"] *, [data-testid="stChatMessage"] { color:#17243a; }
[data-testid="stChatMessage"] { background:white; border:1px solid #e3eaf2; border-radius:14px; }
[data-testid="stChatInput"] textarea, [data-testid="stMain"] input,
[data-testid="stMain"] textarea { color:#17243a; background:white; }
[data-testid="stSidebar"] { background:#101d35; }
[data-testid="stSidebar"] * { color:#e8eef9; }
[data-testid="stSidebar"] input { color:#17243a; background:white; }
</style>""", unsafe_allow_html=True)


def init():
    defaults = {
        "api_base": "http://127.0.0.1:8000", "api_timeout": 300,
        "chat": [], "conversation_id": str(uuid4()), "eval_runs": [],
    }
    for key, value in defaults.items():
        st.session_state.setdefault(key, value)


def api(method, path, *, body=None, files=None):
    try:
        response = httpx.request(
            method, st.session_state.api_base.rstrip("/") + path,
            json=body, files=files, timeout=st.session_state.api_timeout,
        )
        response.raise_for_status()
        return response.json() if response.content else {}
    except httpx.HTTPStatusError as exc:
        raise RuntimeError(f"Backend returned HTTP {exc.response.status_code}: {exc.response.text[:500]}") from exc
    except httpx.RequestError as exc:
        raise RuntimeError("Cannot connect to the backend. Check that it is running and the API URL is correct.") from exc


def navigation():
    st.sidebar.markdown("## Northstar")
    page = st.sidebar.radio("Pages", ["Chat", "Upload", "Evaluation"], label_visibility="collapsed")
    with st.sidebar.expander("Backend connection"):
        st.text_input("API URL", key="api_base")
        st.number_input("Timeout (seconds)", 10, 900, key="api_timeout")
        if st.button("Check connection"):
            try:
                api("GET", "/health")
                st.success("Backend is online")
            except RuntimeError as exc:
                st.error(str(exc))
    return page


def filters():
    keys = ["document_id", "file_name", "category", "subcategory", "policy_id"]
    active = sum(bool(st.session_state.get(f"filter_{key}", "").strip()) for key in keys)
    values = {}
    with st.expander(f"Policy filters ({active} active)" if active else "Policy filters"):
        for key in keys:
            value = st.text_input(key.replace("_", " ").title(), key=f"filter_{key}").strip()
            if value:
                values[key] = value
        if active and st.button("Clear filters"):
            for key in keys:
                st.session_state[f"filter_{key}"] = ""
            st.rerun()
    return values


def search_options(policy_filters):
    with st.expander("Search options"):
        a, b, c = st.columns(3)
        strategy = a.selectbox("Search", ["hybrid", "vector", "keyword"], key="strategy")
        prompt = b.selectbox("Answer style", ["zero_shot", "one_shot", "few_shot"], key="prompt")
        top_k = c.number_input("Sources", 1, 50, 5, key="top_k")
        rerank = st.checkbox("Rerank results", True, key="rerank")
        reranker = st.selectbox("Reranker", ["score", "mmr"], key="reranker") if rerank else "score"
        mmr = st.slider("Relevance to diversity", 0.0, 1.0, 0.7) if rerank and reranker == "mmr" else 0.7
        with st.popover("More options"):
            hyde = st.checkbox("HyDE retrieval")
            multi = st.checkbox("Alternate search queries")
    return {
        "top_k": int(top_k), "candidate_k": max(20, int(top_k)),
        "retrieval_strategy": strategy, "use_hybrid": strategy != "keyword",
        "prompt_strategy": prompt, "use_reranker": rerank,
        "reranker_strategy": reranker, "mmr_lambda": mmr,
        "use_hyde": hyde, "use_multi_query": multi,
        "filters": policy_filters,
    }


def show_sources(sources):
    for index, source in enumerate(sources, 1):
        title = " · ".join(str(value) for value in [
            source.get("citation_id", f"Source {index}"), source.get("file_name"),
            source.get("section"), f"page {source['page']}" if source.get("page") else None,
        ] if value)
        with st.expander(title):
            st.write(source.get("content", ""))
            st.json({k: source[k] for k in ("score", "category", "subcategory", "policy_id", "document_id") if k in source})


def show_answer(response):
    metadata = response.get("metadata", {})
    if metadata.get("refusal"):
        st.warning(response.get("answer", "I could not answer from the available policy documents."))
    else:
        st.markdown(response.get("answer", ""))
    if metadata.get("low_confidence"):
        st.caption("Low retrieval confidence")
    show_sources(response.get("sources", []))


def chat_page():
    st.title("HR Policy Assistant")
    st.caption("Ask questions about your indexed HR policies.")
    options = search_options(filters())
    for item in st.session_state.chat:
        with st.chat_message(item["role"]):
            show_answer(item["response"]) if item.get("response") else st.markdown(item["content"])
    question = st.chat_input("Ask an HR policy question")
    if question:
        history = [{"role": m["role"], "content": m["content"]} for m in st.session_state.chat[-10:]]
        st.session_state.chat.append({"role": "user", "content": question})
        with st.chat_message("user"):
            st.markdown(question)
        with st.chat_message("assistant"):
            try:
                with st.spinner("Searching policy documents..."):
                    result = api("POST", f"{API}/chat", body={
                        "query": question, "conversation_id": st.session_state.conversation_id,
                        "chat_history": history, **options,
                    })
                show_answer(result)
                st.session_state.chat.append({"role": "assistant", "content": result.get("answer", ""), "response": result})
            except RuntimeError as exc:
                st.error(str(exc))
                st.session_state.chat.pop()
    if st.session_state.chat and st.button("Clear chat"):
        st.session_state.chat, st.session_state.conversation_id = [], str(uuid4())
        st.rerun()


def upload_page():
    st.title("Upload policy documents")
    with st.form("upload"):
        file = st.file_uploader("Choose a file", type=["pdf", "docx", "txt", "pptx", "csv"])
        a, b, c = st.columns(3)
        category = a.text_input("Category", "hr_policy")
        subcategory = b.text_input("Subcategory")
        policy_id = c.text_input("Policy ID")
        submit = st.form_submit_button("Upload and index", type="primary")
    if not submit:
        return
    if not file:
        st.warning("Choose a document first.")
        return
    progress = st.status("Indexing document...", expanded=True)
    try:
        mime = file.type or mimetypes.guess_type(file.name)[0] or "application/octet-stream"
        uploaded = api("POST", f"{API}/documents/upload", files={"file": (file.name, file.getvalue(), mime)})
        processed = api("POST", f"{API}/documents/process", body={"document_id": uploaded["document_id"], "blob_name": uploaded["blob_name"]})
        chunked = api("POST", f"{API}/documents/chunk", body={
            "document_id": uploaded["document_id"], "text_blob_name": processed["text_blob_name"],
            "file_name": uploaded["file_name"], "category": category.strip() or "hr_policy",
            "subcategory": subcategory.strip() or None, "policy_id": policy_id.strip() or None,
        })
        indexed = api("POST", f"{API}/documents/embed", body={"document_id": uploaded["document_id"], "chunk_blob_name": chunked["chunk_blob_name"]})
        progress.update(label="Document indexed", state="complete")
        st.success(f"Indexed {indexed['indexed_chunks']} chunks from {uploaded['file_name']}.")
        st.code(f"Document ID: {uploaded['document_id']}")
        st.caption("Use the document ID or policy filters on the Chat page to narrow searches.")
    except (RuntimeError, KeyError) as exc:
        progress.update(label="Upload failed", state="error")
        st.error(str(exc))


def evaluation_page():
    st.title("RAGAS evaluation")
    st.caption("Compare generated answers with known answers. Each batch makes additional model calls.")
    policy_filters = filters()
    rows = st.data_editor(
        pd.DataFrame({"question": [""], "ground_truth": [""]}),
        num_rows="dynamic", use_container_width=True,
        column_config={
            "question": st.column_config.TextColumn("Question", required=True, width="large"),
            "ground_truth": st.column_config.TextColumn("Ground truth", required=True, width="large"),
        }, key="eval_rows",
    )
    a, b, c = st.columns(3)
    strategy = a.selectbox("Search", ["hybrid", "vector", "keyword"], key="eval_search")
    prompt = b.selectbox("Answer style", ["zero_shot", "one_shot", "few_shot"], key="eval_prompt")
    top_k = c.number_input("Sources", 1, 50, 5, key="eval_top_k")
    if st.button("Run evaluation", type="primary"):
        records = [{k: str(row.get(k, "")).strip() for k in ("question", "ground_truth")} for row in rows.fillna("").to_dict("records")]
        records = [row for row in records if row["question"] or row["ground_truth"]]
        if not records or any(not row["question"] or not row["ground_truth"] for row in records):
            st.error("Provide both a question and ground truth for each row.")
        elif len(records) > 20:
            st.error("A batch can contain at most 20 questions.")
        else:
            try:
                with st.spinner("Evaluating answers..."):
                    result = api("POST", f"{API}/evaluate", body={
                        "records": records, "retrieval_strategy": strategy,
                        "prompt_strategy": prompt, "top_k": int(top_k),
                        "candidate_k": max(20, int(top_k)), "filters": policy_filters,
                    })
                st.session_state.eval_runs.append({
                    "id": str(uuid4())[:8], "time": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC"),
                    "results": result["results"], "averages": result["averages"],
                })
            except (RuntimeError, KeyError) as exc:
                st.error(str(exc))
    if st.session_state.eval_runs:
        run = st.session_state.eval_runs[-1]
        for card, key, label in zip(st.columns(4), METRICS, LABELS):
            score = run["averages"].get(key)
            card.metric(label, "—" if score is None else f"{score:.3f}")
        scores = pd.DataFrame(run["results"])
        data = scores.melt(id_vars="question", value_vars=METRICS, var_name="Metric", value_name="Score").dropna()
        data["Metric"] = data["Metric"].map(dict(zip(METRICS, LABELS)))
        st.plotly_chart(px.bar(data, x="question", y="Score", color="Metric", barmode="group", range_y=[0, 1]), use_container_width=True)
        st.dataframe(scores, use_container_width=True, hide_index=True)
        st.download_button("Download results", json.dumps(run, indent=2), file_name="ragas_results.json")


def main():
    init()
    {"Chat": chat_page, "Upload": upload_page, "Evaluation": evaluation_page}[navigation()]()


if __name__ == "__main__":
    main()
