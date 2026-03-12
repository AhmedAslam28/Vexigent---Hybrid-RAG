"""
rag_service.py — Embeddings, Pinecone vector store, document processing,
                 RAG retrieval, and LangChain agent tool definitions.

Exports
-------
embedding_manager         Singleton MultiModalEmbeddingManager
get_vector_store()        Returns (LangChain Pinecone vectorstore, index_name).
process_documents()       Chunk + embed + index documents into Pinecone.
rag_tool()                Retrieve relevant docs from Pinecone for a query.
build_agent_tools()       Build session-scoped LangChain @tools for the ReAct agent.
_format_docs_for_prompt() Format retrieved docs into an LLM context string.
hydrate_qa_chains()       Re-populate qa_chains from existing Pinecone indexes on startup.
"""

import logging
import os
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import openai
from langchain.chains.retrieval_qa.base import RetrievalQA
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.tools import tool as lc_tool
from langchain_core.documents import Document
from langchain_core.prompts import PromptTemplate
from pinecone import Pinecone, ServerlessSpec
from langchain_pinecone import Pinecone as LC_Pinecone

try:
    from langchain_huggingface import HuggingFaceEmbeddings
except ImportError:
    from langchain_community.embeddings import HuggingFaceEmbeddings

from config import (
    OPENAI_API_KEY, PINECONE_API_KEY,
    PINECONE_INDEX_TEXT, PINECONE_INDEX_IMAGE,
    PINECONE_INDEX_AUDIO, PINECONE_INDEX_VIDEO,
    INDEX_CT_MAP,
)

logger = logging.getLogger(__name__)

# ── Pinecone client ───────────────────────────────────────────────────────────
pc = Pinecone(api_key=PINECONE_API_KEY)

# ── OpenAI client (for query rewriting + startup connection test) ─────────────
openai_client = openai.OpenAI(api_key=OPENAI_API_KEY, timeout=30.0, max_retries=2)

# ── In-memory QA chain registry ───────────────────────────────────────────────
qa_chains: Dict[str, Any] = {}


# ─────────────────────────────────────────────────────────────────────────────
# EMBEDDING MANAGER
# ─────────────────────────────────────────────────────────────────────────────

from langchain_core.embeddings import Embeddings


class CLIPEmbeddings(Embeddings):
    def __init__(self):
        self.model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        return self.model.embed_documents(texts)

    def embed_query(self, text: str) -> List[float]:
        return self.model.embed_query(text)


class AudioEmbeddings(Embeddings):
    def __init__(self):
        self.text_embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        return self.text_embeddings.embed_documents(texts)

    def embed_query(self, text: str) -> List[float]:
        return self.text_embeddings.embed_query(text)


class MultiModalEmbeddingManager:
    def __init__(self):
        self.text_embeddings  = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        self.clip_embeddings:  Optional[CLIPEmbeddings]  = None
        self.audio_embeddings: Optional[AudioEmbeddings] = None

    def get_text_embeddings(self):
        return self.text_embeddings

    def get_clip_embeddings(self):
        if self.clip_embeddings is None:
            self.clip_embeddings = CLIPEmbeddings()
        return self.clip_embeddings

    def get_audio_embeddings(self):
        if self.audio_embeddings is None:
            self.audio_embeddings = AudioEmbeddings()
        return self.audio_embeddings


embedding_manager = MultiModalEmbeddingManager()


# ─────────────────────────────────────────────────────────────────────────────
# QA PROMPT TEMPLATES
# ─────────────────────────────────────────────────────────────────────────────

PROMPT = PromptTemplate(
    template=(
        "Use the following pieces of context to answer the question at the end.\n"
        "If you don't know the answer, just say that you don't know, don't try to make up an answer.\n"
        "When answering questions about the main topics or themes, try to identify common subjects "
        "across the documents.\n"
        "Provide specific information and avoid vague responses.\n\n"
        "Context:\n{context}\n\n"
        "Question: {question}\n\n"
        "Provide a detailed answer based only on the given context:"
    ),
    input_variables=["context", "question"],
)

VIDEO_PROMPT = PromptTemplate(
    template=(
        "You are analyzing content from a VIDEO. The context below contains two types of information:\n"
        "1. TRANSCRIPT: the spoken audio from the video\n"
        "2. SCENE NARRATIVE: a temporal description of what was visually happening at different timestamps\n\n"
        "Use BOTH sources together to answer the question. Be specific about:\n"
        "- What was being discussed or said\n"
        "- What was happening visually at the same time\n"
        "- Who was involved\n"
        "- The overall topic and purpose of the video\n\n"
        "If you don't know, say so.\n\n"
        "Context:\n{context}\n\n"
        "Question: {question}\n\n"
        "Provide a detailed answer using both the transcript and scene descriptions:"
    ),
    input_variables=["context", "question"],
)


# ─────────────────────────────────────────────────────────────────────────────
# VECTOR STORE HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def get_vector_store(content_type: str = "text", topic: str = None, namespace: str = None):
    """Return (LangChain Pinecone vectorstore, index_name), creating the index if needed."""
    if content_type == "image":
        embeddings = embedding_manager.get_clip_embeddings()
        idx_name   = PINECONE_INDEX_IMAGE
    elif content_type == "audio":
        embeddings = embedding_manager.get_audio_embeddings()
        idx_name   = PINECONE_INDEX_AUDIO
    elif content_type == "video":
        embeddings = embedding_manager.get_text_embeddings()
        idx_name   = PINECONE_INDEX_VIDEO
    else:
        embeddings = embedding_manager.get_text_embeddings()
        idx_name   = PINECONE_INDEX_TEXT

    if idx_name not in pc.list_indexes().names():
        logger.info(f"Creating Pinecone index: {idx_name}")
        pc.create_index(
            name=idx_name, dimension=384, metric="cosine",
            spec=ServerlessSpec(cloud="aws", region="us-east-1"),
        )
        time.sleep(10)

    kwargs: dict = {"index_name": idx_name, "embedding": embeddings, "text_key": "text"}
    if namespace:
        kwargs["namespace"] = namespace
    return LC_Pinecone.from_existing_index(**kwargs), idx_name


def clean_text(text: str) -> str:
    import re
    text = text.replace("This content downloaded from", "")
    text = text.replace("All use subject to https://about.jstor.org/terms", "")
    text = re.sub(
        r'\d+\.\d+\.\d+\.\d+\s+on\s+[A-Za-z]+,\s+\d+\s+[A-Za-z]+\s+\d+\s+\d+:\d+:\d+\s+UTC',
        '', text,
    )
    text = re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f-\xff]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text


# ─────────────────────────────────────────────────────────────────────────────
# DOCUMENT PROCESSING
# ─────────────────────────────────────────────────────────────────────────────

def process_documents(
    documents,
    source_name: str,
    content_type: str = "text",
    topic: str = None,
    session_id: str = None,
    namespace: str = None,
):
    """Chunk, embed and index documents into Pinecone. Returns a qa_chain on success."""
    from llm_service import OpenAILLM  # local import to avoid circular dependency

    if not documents:
        return {"status": f"No documents loaded from {source_name}.", "success": False}

    try:
        cleaned = []
        for doc in documents:
            if hasattr(doc, "page_content"):
                doc.page_content = clean_text(doc.page_content)
            if session_id:
                doc.metadata["session_id"] = session_id
            if namespace:
                doc.metadata["namespace"]  = namespace
            cleaned.append(doc)

        # Content-type specific splitting
        if content_type == "image":
            texts = cleaned
        elif content_type in ("audio", "video"):
            already_segmented = [
                d for d in cleaned
                if d.metadata.get("type") in (
                    "audio_segment", "video_segment", "audio_header",
                    "video_header", "video_scene", "video_frame",
                )
            ]
            needs_splitting = [d for d in cleaned if d not in already_segmented]
            texts = already_segmented[:]
            if needs_splitting:
                splitter = RecursiveCharacterTextSplitter(
                    chunk_size=500, chunk_overlap=100,
                    separators=[". ", "! ", "? ", "\n", " "],
                )
                texts.extend(splitter.split_documents(needs_splitting))
        else:
            splitter = RecursiveCharacterTextSplitter(
                chunk_size=700, chunk_overlap=300,
                separators=["\n\n", "\n", ". ", " ", ""],
            )
            texts = splitter.split_documents(cleaned)

        logger.info(f"Created {len(texts)} chunks from {len(documents)} {content_type} documents")

        docsearch, idx_used = get_vector_store(content_type, topic, namespace=namespace)

        for t in texts:
            t.metadata.update({
                "content_type": content_type,
                "topic":        topic or "general",
                "source_name":  source_name,
            })

        max_retries = 3
        for attempt in range(max_retries):
            try:
                docsearch.add_documents(texts)
                logger.info(f"✅ Added {len(texts)} {content_type} docs to {idx_used}")
                break
            except Exception as e:
                logger.error(f"❌ Error adding docs (attempt {attempt+1}/{max_retries}): {e}")
                if attempt == max_retries - 1:
                    raise
                time.sleep(2)

        retriever = docsearch.as_retriever(
            search_type="similarity",
            search_kwargs={"k": 8, "filter": {"content_type": content_type}},
        )

        qa_chain = RetrievalQA.from_chain_type(
            llm=OpenAILLM(model="gpt-4o"),
            chain_type="stuff",
            retriever=retriever,
            return_source_documents=True,
            chain_type_kwargs={"prompt": PROMPT},
        )

        return {
            "status":       f"Processed {len(texts)} {content_type} chunks from {len(documents)} documents via {source_name}.",
            "success":      True,
            "qa_chain":     qa_chain,
            "content_type": content_type,
            "topic":        topic,
            "index_name":   idx_used,
        }
    except Exception as e:
        msg = f"Error processing {content_type} documents from {source_name}: {str(e)}"
        logger.error(msg)
        return {"status": msg, "success": False}


# ─────────────────────────────────────────────────────────────────────────────
# RAG TOOL
# ─────────────────────────────────────────────────────────────────────────────

def _rewrite_query(query: str) -> str:
    try:
        r = openai_client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content":
                    "Rewrite the user's search query to be more specific and "
                    "descriptive for document retrieval. Return only the rewritten query."},
                {"role": "user", "content": query},
            ],
            max_tokens=100, timeout=10,
        )
        rewritten = r.choices[0].message.content.strip()
        logger.info(f"🖊️ Query rewritten: '{query}' → '{rewritten}'")
        return rewritten
    except Exception as e:
        logger.warning(f"⚠️ Query rewrite failed: {e}")
        return query


def _rerank_docs(query: str, docs: List[Document], top_k: int) -> List[Document]:
    try:
        from sentence_transformers import CrossEncoder
        if not hasattr(_rerank_docs, "_model"):
            _rerank_docs._model = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
        scores = _rerank_docs._model.predict([(query, d.page_content) for d in docs])
        ranked = sorted(zip(scores, docs), key=lambda x: x[0], reverse=True)
        logger.info(f"🔄 Reranked {len(docs)} docs → top {top_k}")
        return [d for _, d in ranked[:top_k]]
    except Exception as e:
        logger.warning(f"⚠️ Reranking failed: {e}")
        return docs[:top_k]


def _format_docs_for_prompt(docs: List[Document]) -> str:
    """Format retrieved documents into a context string for the LLM."""
    if not docs:
        return "No relevant documents found."
    parts = []
    seen  = set()
    for i, doc in enumerate(docs):
        key = doc.page_content[:80]
        if key in seen:
            continue
        seen.add(key)
        src   = doc.metadata.get("source", "unknown")
        fname = os.path.basename(src) if isinstance(src, str) else str(src)
        ct    = doc.metadata.get("content_type", "")
        dtype = doc.metadata.get("type", "")
        label = f"[{ct}] " if ct else ""

        ts_info  = ""
        start_ts = doc.metadata.get("start_ts") or doc.metadata.get("frame_timestamp")
        end_ts   = doc.metadata.get("end_ts")
        if start_ts and end_ts:
            ts_info = f" | ⏱ {start_ts}–{end_ts}"
        elif start_ts:
            ts_info = f" | ⏱ {start_ts}"

        parts.append(
            f"--- Chunk {i+1} | {label}{fname}{ts_info} [{dtype}] ---\n"
            f"{doc.page_content.strip()}"
        )
    return "\n\n".join(parts)


def rag_tool(
    query: str,
    session_id: str,
    content_types: List[str],
    top_k: int = 8,
    use_reranker: bool = False,
    use_query_rewriter: bool = False,
    namespace: str = None,
) -> List[Document]:
    """
    Retrieve relevant document chunks from Pinecone.
    Optionally rewrites the query and/or reranks results with a cross-encoder.
    """
    if use_query_rewriter:
        query = _rewrite_query(query)

    all_docs: List[Document] = []

    for ct in content_types:
        chain_key  = f"{session_id}_{ct}_general"
        legacy_key = f"legacy_{ct}_general"
        if chain_key not in qa_chains:
            if legacy_key in qa_chains:
                chain_key = legacy_key
            else:
                continue
        chain = qa_chains[chain_key]
        try:
            k_fetch = top_k * 2 if use_reranker else top_k
            try:
                if namespace:
                    ns_store, _ = get_vector_store(ct, namespace=namespace)
                    retriever   = ns_store.as_retriever(
                        search_type="similarity",
                        search_kwargs={"k": k_fetch},
                    )
                else:
                    retriever = chain.retriever.vectorstore.as_retriever(
                        search_type="similarity",
                        search_kwargs={"k": k_fetch, "filter": {"content_type": ct}},
                    )
            except Exception:
                retriever = chain.retriever
            docs = retriever.get_relevant_documents(query)
            for d in docs:
                d.metadata.setdefault("content_type", ct)
            all_docs.extend(docs)
        except Exception as e:
            logger.warning(f"RAG tool error for {ct}: {e}")

    if use_reranker and all_docs:
        all_docs = _rerank_docs(query, all_docs, top_k)
    else:
        all_docs = all_docs[:top_k]

    return all_docs


# ─────────────────────────────────────────────────────────────────────────────
# AGENT TOOLS (session-scoped LangChain @tools for the ReAct agent)
# ─────────────────────────────────────────────────────────────────────────────

def build_agent_tools(
    session_id: str,
    content_types: List[str],
    top_k: int,
    use_reranker: bool,
    use_query_rewriter: bool,
    namespace: str = None,
    qa_chains: dict = None,
):
    """Return a list of LangChain tools scoped to this request's session/settings."""
    _qa_chains = qa_chains  # capture for closures

    @lc_tool
    def search_images(query: str) -> str:
        """
        Search uploaded IMAGE content (photos, screenshots, diagrams).
        Use for: 'what do you see in the image', 'describe the photo',
        'what text is in the screenshot', 'who is in the image'.
        """
        docs = rag_tool(
            query=query, session_id=session_id,
            content_types=["image"], top_k=top_k,
            use_reranker=use_reranker, use_query_rewriter=use_query_rewriter,
            namespace=namespace,
        )
        if not docs:
            return "No images found for this session."
        return _format_docs_for_prompt(docs)

    @lc_tool
    def search_audio(query: str) -> str:
        """
        Search uploaded AUDIO content (speech transcripts with timestamps).
        Use for: 'what was said', 'what was mentioned at 30 seconds',
        'first/last sentence', 'pauses in the audio'.
        """
        docs = rag_tool(
            query=query, session_id=session_id,
            content_types=["audio"], top_k=top_k,
            use_reranker=use_reranker, use_query_rewriter=use_query_rewriter,
            namespace=namespace,
        )
        if not docs:
            return "No audio files found for this session."
        return _format_docs_for_prompt(docs)

    @lc_tool
    def search_video(query: str) -> str:
        """
        Search uploaded VIDEO content (visual frame descriptions + audio transcript).
        Use for: 'what happens in the video', 'who appears in the video',
        'what was said in the video', 'describe the scene at 1 minute'.
        """
        docs = rag_tool(
            query=query, session_id=session_id,
            content_types=["video"], top_k=top_k,
            use_reranker=use_reranker, use_query_rewriter=use_query_rewriter,
            namespace=namespace,
        )
        if not docs:
            return "No video files found for this session."
        return _format_docs_for_prompt(docs)

    @lc_tool
    def search_documents(query: str) -> str:
        """
        Search uploaded TEXT/PDF documents.
        Use for questions about PDFs, Word docs, text files, research papers.
        Also use as a fallback when other search tools return no results.
        """
        docs = rag_tool(
            query=query, session_id=session_id,
            content_types=["text"], top_k=top_k,
            use_reranker=use_reranker, use_query_rewriter=use_query_rewriter,
            namespace=namespace,
        )
        if not docs:
            docs = rag_tool(
                query=query, session_id=session_id,
                content_types=content_types, top_k=top_k,
                use_reranker=use_reranker, use_query_rewriter=use_query_rewriter,
            )
        if not docs:
            return "No documents found for this session."
        return _format_docs_for_prompt(docs)

    @lc_tool
    def count_and_list_files(content_type: str = "") -> str:
        """
        Count and list files indexed for this session by type.
        Use ONLY when the user explicitly asks HOW MANY files are uploaded,
        or what files are available.
        Input: content type (image | audio | video | text | blank for all).
        """
        index_map = {
            "image": PINECONE_INDEX_IMAGE,
            "audio": PINECONE_INDEX_AUDIO,
            "video": PINECONE_INDEX_VIDEO,
            "text":  PINECONE_INDEX_TEXT,
        }
        types_to_check = (
            [content_type.lower().strip()] if content_type.strip()
            else ["image", "audio", "video", "text"]
        )
        results = []
        for ct in types_to_check:
            idx_name = index_map.get(ct)
            if not idx_name:
                continue
            try:
                existing = [i.name for i in pc.list_indexes()]
                if idx_name not in existing:
                    continue
                index     = pc.Index(idx_name)
                stats     = index.describe_index_stats()
                total_vecs = stats.get("total_vector_count", 0)
                if total_vecs == 0:
                    continue
                dummy_vec = [0.0] * stats.get("dimension", 384)
                res       = index.query(
                    vector=dummy_vec,
                    top_k=min(total_vecs, 10000),
                    include_values=False, include_metadata=True,
                )
                matches = res.get("matches", [])
                if not matches:
                    continue
                sources = set()
                for m in matches:
                    fname = (m.get("metadata") or {}).get("filename") \
                         or (m.get("metadata") or {}).get("source", "unknown")
                    sources.add(fname)
                results.append(
                    f"• {ct}: {len(sources)} file(s), {len(matches)} indexed chunk(s)\n"
                    f"  Files: {', '.join(sorted(sources))}"
                )
            except Exception as e:
                logger.warning(f"⚠️ count_and_list_files failed for {ct}: {e}")

        if not results:
            return "No files indexed for this session yet."
        return "Files indexed for this session:\n" + "\n".join(results)

    @lc_tool
    def list_indexed_content(dummy_input: str = "") -> str:
        """
        Quick summary of what content TYPES are indexed (not counts).
        Use to understand what kinds of content are available before searching.
        """
        current_chains = [k for k in (_qa_chains or qa_chains) if k.startswith(session_id)]
        if not current_chains:
            return "No documents are currently indexed for this session."
        lines = []
        for c in current_chains:
            parts = c.replace(session_id + "_", "").split("_")
            ct    = parts[0] if parts else "unknown"
            topic = parts[1] if len(parts) > 1 else "general"
            lines.append(f"• {ct} content (topic: {topic})")
        return "Currently indexed content types:\n" + "\n".join(lines)

    return [search_images, search_audio, search_video, search_documents,
            count_and_list_files, list_indexed_content]


# ─────────────────────────────────────────────────────────────────────────────
# STARTUP HYDRATION
# ─────────────────────────────────────────────────────────────────────────────

def hydrate_qa_chains(namespace: str = None):
    """
    Re-populate qa_chains from existing Pinecone indexes on startup.
    Ensures previous uploads are immediately queryable without re-uploading.
    """
    from llm_service import OpenAILLM  # local import to avoid circular dependency

    try:
        existing_indexes = [i.name for i in pc.list_indexes()]
    except Exception as e:
        logger.error(f"❌ Startup hydration: could not list Pinecone indexes: {e}")
        return

    hydrated = 0
    for idx_name, ct in INDEX_CT_MAP.items():
        if idx_name not in existing_indexes:
            continue
        try:
            index = pc.Index(idx_name)
            stats = index.describe_index_stats()
            total = stats.get("total_vector_count", 0)
            if total == 0:
                continue

            dim       = stats.get("dimension", 384)
            dummy_vec = [0.0] * dim

            query_kwargs: dict = dict(
                vector=dummy_vec,
                top_k=min(total, 10000),
                include_values=False,
                include_metadata=True,
            )
            if namespace:
                query_kwargs["namespace"] = namespace

            res     = index.query(**query_kwargs)
            matches = res.get("matches", [])

            session_ids = set()
            for m in matches:
                sid = (m.get("metadata") or {}).get("session_id")
                if sid:
                    session_ids.add(sid)

            if not session_ids:
                # Legacy uploads without session_id in metadata
                session_ids = {"legacy"}

            for sid in session_ids:
                chain_key = f"{sid}_{ct}_general"
                if chain_key in qa_chains:
                    continue
                try:
                    docsearch = LC_Pinecone.from_existing_index(
                        index_name=idx_name,
                        embedding=embedding_manager.get_text_embeddings(),
                        text_key="text",
                        **({"namespace": namespace} if namespace else {}),
                    )
                    retriever = docsearch.as_retriever(
                        search_type="similarity",
                        search_kwargs={"k": 8, "filter": {"content_type": ct}},
                    )
                    chain = RetrievalQA.from_chain_type(
                        llm=OpenAILLM(model="gpt-4o"),
                        chain_type="stuff",
                        retriever=retriever,
                        return_source_documents=True,
                        chain_type_kwargs={"prompt": PROMPT},
                    )
                    qa_chains[chain_key] = chain
                    hydrated += 1
                    logger.info(f"✅ Hydrated chain: {chain_key} (index={idx_name}, vectors={total})")
                except Exception as ce:
                    logger.error(f"❌ Could not build chain for {chain_key}: {ce}")

        except Exception as e:
            logger.error(f"❌ Startup hydration failed for index {idx_name}: {e}")

    logger.info(f"🔄 Startup hydration complete — {hydrated} chain(s) restored from Pinecone")


def test_openai_connection() -> bool:
    try:
        openai_client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": "ping"}],
            max_tokens=5, timeout=10,
        )
        return True
    except Exception as e:
        logger.error(f"OpenAI test failed: {e}")
        return False
