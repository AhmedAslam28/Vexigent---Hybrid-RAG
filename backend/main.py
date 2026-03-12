"""
main.py — FastAPI application: routes only.

All business logic lives in the service modules:
  config.py      → env vars, constants, domain prompts, model catalogue
  llm_service.py → LLM wrappers, hybrid_rag_query, run_langchain_agent
  rag_service.py → embeddings, vector store, document processing, RAG retrieval
  media_utils.py → image / audio / video processing

This file is intentionally kept thin:
  • Request / Response Pydantic models
  • FastAPI app + CORS
  • Auth routes (SQLite-backed)
  • Upload / query / utility routes
  • Startup event wiring
"""

import hashlib
import logging
import os
import shutil
import sqlite3
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch
from fastapi import BackgroundTasks, FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from langchain_community.document_loaders import (
    DirectoryLoader, PyPDFLoader, WebBaseLoader,
)
from pydantic import BaseModel

# ── Service modules ───────────────────────────────────────────────────────────
from config import (
    AVAILABLE_DOMAINS, AWS_ACCESS_KEY, AWS_SECRET_KEY,
    OPENAI_API_KEY, ANTHROPIC_API_KEY, GOOGLE_API_KEY, PINECONE_API_KEY,
    PROVIDER_MODELS, UPLOAD_DIR, USER_DB_PATH,
)
from llm_service import hybrid_rag_query, run_langchain_agent
from media_utils import (
    detect_content_type,
    image_captioning_service,
    process_audio,
    process_image,
    process_video,
)
from rag_service import (
    hydrate_qa_chains,
    pc,
    process_documents,
    qa_chains,
    test_openai_connection,
)

# ── Optional loaders ──────────────────────────────────────────────────────────
try:
    from langchain_community.document_loaders import GoogleDriveLoader
except Exception:
    GoogleDriveLoader = None

try:
    from langchain_community.document_loaders import S3DirectoryLoader, S3FileLoader
except Exception:
    S3DirectoryLoader = S3FileLoader = None

try:
    from langchain_community.document_loaders import DropboxLoader
except Exception:
    DropboxLoader = None

from pinecone import ServerlessSpec

logger = logging.getLogger(__name__)
os.makedirs(UPLOAD_DIR, exist_ok=True)


# ─────────────────────────────────────────────────────────────────────────────
# ENHANCED LOGGER
# ─────────────────────────────────────────────────────────────────────────────

class EnhancedLogger:
    def __init__(self):
        self.upload_logs: Dict[str, list] = {}
        self.query_logs:  Dict[str, list] = {}

    def log_upload(self, session_id: str, filename: str, file_type: str, topic: str = None):
        entry = {
            "timestamp":  datetime.now().isoformat(),
            "filename":   filename,
            "file_type":  file_type,
            "topic":      topic,
            "session_id": session_id,
        }
        self.upload_logs.setdefault(session_id, []).append(entry)
        logger.info(f"📁 UPLOAD LOG: {entry}")

    def log_query(self, session_id: str, query: str, model_used: str, results_count: int):
        entry = {
            "timestamp":     datetime.now().isoformat(),
            "session_id":    session_id,
            "query":         query[:100] + "..." if len(query) > 100 else query,
            "model_used":    model_used,
            "results_count": results_count,
        }
        self.query_logs.setdefault(session_id, []).append(entry)
        logger.info(f"🔍 QUERY LOG: {entry}")


enhanced_logger = EnhancedLogger()


# ─────────────────────────────────────────────────────────────────────────────
# FILE STATUS TRACKER
# ─────────────────────────────────────────────────────────────────────────────

class FileStatusTracker:
    STAGES = ["queued", "parsing", "chunking", "indexing", "indexed"]

    def __init__(self):
        self._records: Dict[str, dict] = {}
        self._lock = __import__("threading").Lock()

    def register(self, upload_id: str, filename: str, session_id: str):
        with self._lock:
            self._records[upload_id] = {
                "upload_id":  upload_id,
                "filename":   filename,
                "session_id": session_id,
                "stage":      "queued",
                "error":      None,
                "updated_at": datetime.now().isoformat(),
            }

    def set_stage(self, upload_id: str, stage: str, error: str = None):
        with self._lock:
            if upload_id in self._records:
                self._records[upload_id].update({
                    "stage": stage, "error": error,
                    "updated_at": datetime.now().isoformat(),
                })
                logger.info(
                    f"📊 File status [{upload_id}] "
                    f"{self._records[upload_id]['filename']} → {stage}"
                    + (f" ({error})" if error else "")
                )

    def get(self, upload_id: str) -> dict:
        with self._lock:
            return dict(self._records.get(upload_id, {}))

    def get_session(self, session_id: str) -> list:
        with self._lock:
            return [dict(v) for v in self._records.values() if v["session_id"] == session_id]


file_status_tracker = FileStatusTracker()


# ─────────────────────────────────────────────────────────────────────────────
# FILE HASH HELPERS  (duplicate detection)
# ─────────────────────────────────────────────────────────────────────────────

def calculate_file_hash(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()


def check_file_exists(file_hash: str, session_id: str) -> bool:
    if not hasattr(check_file_exists, "hashes"):
        check_file_exists.hashes = {}
    return file_hash in check_file_exists.hashes.get(session_id, set())


def store_file_hash(file_hash: str, session_id: str):
    if not hasattr(check_file_exists, "hashes"):
        check_file_exists.hashes = {}
    check_file_exists.hashes.setdefault(session_id, set()).add(file_hash)


def get_session_id() -> str:
    return str(uuid.uuid4())


def save_uploaded_file(file: UploadFile) -> str:
    path = os.path.join(UPLOAD_DIR, datetime.now().strftime("%Y%m%d_%H%M%S"))
    os.makedirs(path, exist_ok=True)
    fpath = os.path.join(path, file.filename)
    with open(fpath, "wb") as buf:
        shutil.copyfileobj(file.file, buf)
    return fpath


# ─────────────────────────────────────────────────────────────────────────────
# BACKGROUND UPLOAD TASK
# ─────────────────────────────────────────────────────────────────────────────

def process_multimodal_upload_task(
    file_paths, content_types, session_id,
    topic=None, upload_ids=None,
    provider: str = None, api_key: str = None,
    namespace: str = None,
):
    if upload_ids is None:
        upload_ids = [""] * len(file_paths)

    logger.info(f"🎨 Vision provider for this upload: {provider or 'auto (env key priority)'}")

    try:
        type_groups: Dict[str, list] = {}
        for fp, ct, uid in zip(file_paths, content_types, upload_ids):
            type_groups.setdefault(ct, []).append((fp, uid))

        for ct, pairs in type_groups.items():
            try:
                all_docs, all_uids = [], [uid for _, uid in pairs]

                for fp, uid in pairs:
                    try:
                        file_status_tracker.set_stage(uid, "parsing")

                        if ct == "text":
                            docs = PyPDFLoader(fp).load()
                            all_docs.extend(docs)

                        elif ct == "image":
                            from langchain_core.documents import Document
                            all_docs.append(Document(
                                page_content=process_image(fp, provider=provider, api_key=api_key),
                                metadata={
                                    "source":          fp,
                                    "type":            "image",
                                    "filename":        Path(fp).name,
                                    "origin":          "image_upload",
                                    "vision_provider": provider or "auto",
                                },
                            ))

                        elif ct == "audio":
                            audio_docs = process_audio(fp)
                            all_docs.extend(audio_docs)
                            logger.info(
                                f"🎵 Audio produced {len(audio_docs)} timestamped docs from {Path(fp).name}"
                            )

                        elif ct == "video":
                            vr    = process_video(fp, provider=provider, api_key=api_key)
                            vdocs = vr.get("transcript_segments", [])
                            if not vdocs and vr["transcript_doc"]:
                                vdocs = [vr["transcript_doc"]]
                            vdocs.extend(vr["frame_docs"])
                            if vdocs:
                                file_status_tracker.set_stage(uid, "chunking")
                                file_status_tracker.set_stage(uid, "indexing")
                                vres = process_documents(
                                    vdocs, "video", "video", topic,
                                    session_id=session_id, namespace=namespace,
                                )
                                if vres["success"]:
                                    qa_chains[f"{session_id}_video_{topic or 'general'}"] = vres["qa_chain"]
                                    file_status_tracker.set_stage(uid, "indexed")
                                    logger.info(f"✅ Video processed for session {session_id} ({len(vdocs)} docs)")
                                else:
                                    file_status_tracker.set_stage(uid, "error", vres["status"])
                            else:
                                file_status_tracker.set_stage(uid, "error", "No video docs produced")
                            continue

                    except Exception as fe:
                        file_status_tracker.set_stage(uid, "error", str(fe))
                        logger.error(f"❌ Parse error {fp}: {fe}")

                if not all_docs:
                    continue
                for uid in all_uids:
                    file_status_tracker.set_stage(uid, "chunking")
                for uid in all_uids:
                    file_status_tracker.set_stage(uid, "indexing")
                result = process_documents(
                    all_docs, f"{ct} upload", ct, topic,
                    session_id=session_id, namespace=namespace,
                )
                if result["success"]:
                    qa_chains[f"{session_id}_{ct}_{topic or 'general'}"] = result["qa_chain"]
                    for uid in all_uids:
                        file_status_tracker.set_stage(uid, "indexed")
                    logger.info(f"✅ Processed {ct} files for session {session_id}")
                else:
                    for uid in all_uids:
                        file_status_tracker.set_stage(uid, "error", result["status"])

            except Exception as e:
                logger.error(f"❌ Error processing {ct} files: {e}")

    except Exception as e:
        logger.error(f"❌ Upload task error: {e}")


# ─────────────────────────────────────────────────────────────────────────────
# AUTH HELPERS (SQLite-backed)
# ─────────────────────────────────────────────────────────────────────────────

def _get_user_db() -> sqlite3.Connection:
    conn = sqlite3.connect(USER_DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def _init_user_db():
    with _get_user_db() as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS users (
                user_id    TEXT PRIMARY KEY,
                username   TEXT NOT NULL,
                email      TEXT NOT NULL UNIQUE,
                password   TEXT NOT NULL,
                created_at TEXT NOT NULL DEFAULT (datetime('now'))
            )
        """)
        conn.commit()
    logger.info("✅ User DB initialised at %s", USER_DB_PATH)


def _hash_pw(password: str) -> str:
    return hashlib.sha256(password.encode("utf-8")).hexdigest()


def _user_namespace(user_id: str) -> str:
    return f"u_{user_id[:8]}"


# ─────────────────────────────────────────────────────────────────────────────
# PYDANTIC MODELS
# ─────────────────────────────────────────────────────────────────────────────

class RegisterRequest(BaseModel):
    username: str
    email:    str
    password: str


class LoginRequest(BaseModel):
    email:    str
    password: str


class HybridQueryRequest(BaseModel):
    session_id:        str
    query:             str
    provider:          str           = "openai"
    model_name:        str           = "gpt-4o"
    domain:            str           = "General"
    api_key:           Optional[str] = None
    namespace:         Optional[str] = None
    top_k:             int           = 8
    use_reranker:      bool          = False
    use_query_rewriter: bool         = False
    content_types:     List[str]     = ["text", "image", "audio", "video"]


class QueryResponse(BaseModel):
    answer:      str
    explanation: str
    success:     bool


class ProcessResponse(BaseModel):
    status:     str
    success:    bool
    session_id: str
    upload_ids: Optional[List[str]] = []


class DirectoryRequest(BaseModel):
    session_id:     Optional[str] = None
    directory_path: str


class WebsiteRequest(BaseModel):
    session_id: Optional[str] = None
    url:        str


class S3Request(BaseModel):
    session_id:  Optional[str] = None
    bucket_name: str
    prefix:      Optional[str] = ""


class DropboxRequest(BaseModel):
    session_id:   Optional[str] = None
    access_token: str
    folder_path:  str


class GoogleDriveRequest(BaseModel):
    session_id:       Optional[str] = None
    folder_id:        str
    credentials_path: str


# ─────────────────────────────────────────────────────────────────────────────
# FASTAPI APP
# ─────────────────────────────────────────────────────────────────────────────

app = FastAPI(title="Hybrid Prompt + RAG Tool API")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_credentials=True,
    allow_methods=["*"], allow_headers=["*"],
)


# ── Startup ───────────────────────────────────────────────────────────────────

@app.on_event("startup")
async def startup_event():
    logger.info("🚀 Hybrid Prompt + RAG Tool API starting up...")
    _init_user_db()
    image_captioning_service.initialize_blip()

    if not test_openai_connection():
        logger.warning("⚠️ OpenAI connection test failed")

    try:
        indexes = pc.list_indexes().names()
        logger.info(f"✅ Pinecone connected. Indexes: {indexes}")
    except Exception as e:
        logger.error(f"❌ Pinecone connection failed: {e}")
        return

    hydrate_qa_chains()


# ── Auth routes ───────────────────────────────────────────────────────────────

@app.post("/auth/register")
async def auth_register(req: RegisterRequest):
    if not req.username.strip() or not req.email.strip() or not req.password:
        raise HTTPException(status_code=400, detail="username, email and password are all required")
    user_id = str(uuid.uuid4())
    try:
        with _get_user_db() as conn:
            conn.execute(
                "INSERT INTO users (user_id, username, email, password) VALUES (?,?,?,?)",
                (user_id, req.username.strip(), req.email.strip().lower(), _hash_pw(req.password)),
            )
            conn.commit()
    except sqlite3.IntegrityError:
        raise HTTPException(status_code=409, detail="Email is already registered")
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))
    ns = _user_namespace(user_id)
    logger.info("✅ Registered user=%s ns=%s", req.email, ns)
    return {"user_id": user_id, "username": req.username.strip(),
            "email": req.email.strip().lower(), "namespace": ns}


@app.post("/auth/login")
async def auth_login(req: LoginRequest):
    if not req.email or not req.password:
        raise HTTPException(status_code=400, detail="email and password are required")
    with _get_user_db() as conn:
        row = conn.execute(
            "SELECT * FROM users WHERE email=? AND password=?",
            (req.email.strip().lower(), _hash_pw(req.password)),
        ).fetchone()
    if not row:
        raise HTTPException(status_code=401, detail="Invalid email or password")
    ns = _user_namespace(row["user_id"])
    logger.info("✅ Login user=%s ns=%s", row["email"], ns)
    return {"user_id": row["user_id"], "username": row["username"],
            "email": row["email"], "namespace": ns}


@app.get("/auth/me")
async def auth_me(user_id: str):
    with _get_user_db() as conn:
        row = conn.execute("SELECT * FROM users WHERE user_id=?", (user_id,)).fetchone()
    if not row:
        raise HTTPException(status_code=404, detail="User not found")
    return {"user_id": row["user_id"], "username": row["username"],
            "email": row["email"], "namespace": _user_namespace(row["user_id"])}


# ── Model / config routes ─────────────────────────────────────────────────────

@app.get("/models/")
async def get_available_models():
    return {
        "providers":    PROVIDER_MODELS,
        "domains":      AVAILABLE_DOMAINS,
        "architecture": "hybrid_prompt_rag_tool",
    }


@app.get("/health/")
async def health_check():
    return {"status": "healthy", "architecture": "hybrid_prompt_rag_tool"}


@app.get("/config-check/")
async def check_configuration():
    return {
        "openai_available":    bool(OPENAI_API_KEY),
        "anthropic_available": bool(ANTHROPIC_API_KEY),
        "google_available":    bool(GOOGLE_API_KEY),
        "pinecone_available":  bool(PINECONE_API_KEY),
        "cuda_available":      torch.cuda.is_available(),
        "architecture":        "hybrid_prompt_rag_tool",
        "agent_framework":     "langchain_react_agent_executor",
        "agent_tools":         ["search_images", "search_audio", "search_video",
                                "search_documents", "count_and_list_files", "list_indexed_content"],
        "agent_providers":     ["openai", "anthropic", "gemini"],
        "domains":             AVAILABLE_DOMAINS,
    }


@app.get("/test-openai/")
async def test_openai_endpoint():
    return {"openai_connection": "success" if test_openai_connection() else "failed"}


# ── Upload routes ─────────────────────────────────────────────────────────────

@app.post("/upload/", response_model=ProcessResponse)
async def upload_files(
    background_tasks: BackgroundTasks,
    files:      List[UploadFile]    = File(...),
    session_id: Optional[str]       = Form(None),
    topic:      Optional[str]       = Form(None),
    overwrite:  bool                = Form(False),
    provider:   Optional[str]       = Form(None),
    api_key:    Optional[str]       = Form(None),
    namespace:  Optional[str]       = Form(None),
):
    if not session_id:
        session_id = get_session_id()
    file_paths, content_types_list, upload_ids, warnings = [], [], [], []
    try:
        for file in files:
            content = await file.read()
            fhash   = calculate_file_hash(content)
            if check_file_exists(fhash, session_id) and not overwrite:
                warnings.append(f"{file.filename} already exists")
                continue
            file.file.seek(0)
            fpath = save_uploaded_file(file)
            ct    = detect_content_type(fpath)
            uid   = str(uuid.uuid4())
            file_paths.append(fpath)
            content_types_list.append(ct)
            upload_ids.append(uid)
            file_status_tracker.register(uid, file.filename, session_id)
            enhanced_logger.log_upload(session_id, file.filename, ct, topic)
            store_file_hash(fhash, session_id)

        if warnings and not file_paths:
            return {"status": f"All files already exist. {'; '.join(warnings)}",
                    "success": False, "session_id": session_id}

        background_tasks.add_task(
            process_multimodal_upload_task,
            file_paths, content_types_list, session_id, topic, upload_ids,
            provider, api_key, namespace,
        )

        msg = f"Uploaded {len(file_paths)} files. Processing in background."
        if topic:    msg += f" Topic: {topic}"
        if warnings: msg += f" Warnings: {'; '.join(warnings)}"
        return {"status": msg, "success": True, "session_id": session_id, "upload_ids": upload_ids}
    except Exception as e:
        return {"status": f"Upload error: {e}", "success": False, "session_id": session_id}


@app.get("/upload-status/{upload_id}")
async def get_upload_status(upload_id: str):
    record = file_status_tracker.get(upload_id)
    if not record:
        raise HTTPException(status_code=404, detail="upload_id not found")
    return record


@app.get("/upload-status-session/{session_id}")
async def get_session_upload_statuses(session_id: str):
    return {"files": file_status_tracker.get_session(session_id)}


# ── Source-specific ingestion routes ──────────────────────────────────────────

@app.post("/directory/", response_model=ProcessResponse)
async def process_directory_endpoint(request: DirectoryRequest):
    session_id = request.session_id or get_session_id()
    try:
        if not os.path.exists(request.directory_path):
            return {"status": f"Directory not found: {request.directory_path}",
                    "success": False, "session_id": session_id}
        loader    = DirectoryLoader(request.directory_path, glob="**/*.pdf", loader_cls=PyPDFLoader)
        documents = loader.load()
        result    = process_documents(documents, "local directory")
        if result["success"]:
            qa_chains[f"{session_id}_text_general"] = result["qa_chain"]
        return {**result, "session_id": session_id}
    except Exception as e:
        return {"status": f"Directory error: {e}", "success": False, "session_id": session_id}


@app.post("/website/", response_model=ProcessResponse)
async def process_website_endpoint(request: WebsiteRequest):
    session_id = request.session_id or get_session_id()
    try:
        loader    = WebBaseLoader(request.url)
        documents = loader.load()
        result    = process_documents(documents, "website")
        if result["success"]:
            qa_chains[f"{session_id}_text_general"] = result["qa_chain"]
        return {**result, "session_id": session_id}
    except Exception as e:
        return {"status": f"Website error: {e}", "success": False, "session_id": session_id}


@app.post("/s3/", response_model=ProcessResponse)
async def process_s3_endpoint(request: S3Request):
    session_id = request.session_id or get_session_id()
    try:
        import boto3
        boto_session = boto3.Session(
            aws_access_key_id=AWS_ACCESS_KEY,
            aws_secret_access_key=AWS_SECRET_KEY,
        )
        loader    = S3DirectoryLoader(bucket=request.bucket_name, prefix=request.prefix,
                                      boto3_session=boto_session)
        documents = loader.load()
        result    = process_documents(documents, "S3")
        if result["success"]:
            qa_chains[f"{session_id}_text_general"] = result["qa_chain"]
        return {**result, "session_id": session_id}
    except Exception as e:
        return {"status": f"S3 error: {e}", "success": False, "session_id": session_id}


@app.post("/gdrive/", response_model=ProcessResponse)
async def process_gdrive_endpoint(request: GoogleDriveRequest):
    session_id = request.session_id or get_session_id()
    try:
        loader    = GoogleDriveLoader(
            folder_id=request.folder_id,
            credentials_path=request.credentials_path,
            file_types=["pdf"],
        )
        documents = loader.load()
        result    = process_documents(documents, "Google Drive")
        if result["success"]:
            qa_chains[f"{session_id}_text_general"] = result["qa_chain"]
        return {**result, "session_id": session_id}
    except Exception as e:
        return {"status": f"GDrive error: {e}", "success": False, "session_id": session_id}


# ── Query routes ──────────────────────────────────────────────────────────────

def _run_hybrid(request: HybridQueryRequest) -> dict:
    return hybrid_rag_query(
        query=request.query,
        session_id=request.session_id,
        provider=request.provider,
        model_name=request.model_name,
        domain=request.domain,
        api_key=request.api_key,
        namespace=request.namespace,
        top_k=request.top_k,
        use_reranker=request.use_reranker,
        use_query_rewriter=request.use_query_rewriter,
        content_types=request.content_types,
        enhanced_logger=enhanced_logger,
    )


@app.post("/query/", response_model=QueryResponse)
async def query_documents(request: HybridQueryRequest):
    if not request.query.strip():
        return QueryResponse(answer="Empty query.", explanation="", success=False)
    return QueryResponse(**_run_hybrid(request))


@app.post("/query-enhanced/", response_model=QueryResponse)
async def query_documents_enhanced(request: HybridQueryRequest):
    """Alias for /query/."""
    return await query_documents(request)


@app.post("/query-topic/", response_model=QueryResponse)
async def query_documents_by_topic(request: HybridQueryRequest):
    """Topic-aware query — same architecture, domain from request.domain."""
    return await query_documents(request)


@app.post("/query-agent/", response_model=QueryResponse)
async def query_agent(request: HybridQueryRequest):
    """Real LangChain ReAct agent endpoint. Falls back to hybrid_rag_query on error."""
    if not request.query.strip():
        return QueryResponse(answer="Empty query.", explanation="", success=False)
    result = run_langchain_agent(
        query=request.query,
        session_id=request.session_id,
        provider=request.provider,
        model_name=request.model_name,
        domain=request.domain,
        api_key=request.api_key,
        namespace=request.namespace,
        top_k=request.top_k,
        use_reranker=request.use_reranker,
        use_query_rewriter=request.use_query_rewriter,
        content_types=request.content_types,
        max_iterations=8,
        enhanced_logger=enhanced_logger,
        qa_chains=qa_chains,
    )
    return QueryResponse(**result)


# ── Reset / session / DB routes ───────────────────────────────────────────────

@app.post("/reset/", response_model=ProcessResponse)
async def reset_vector_store(session_id: Optional[str] = Form(None)):
    try:
        index_name = "pdf-query-index"
        if index_name in pc.list_indexes().names():
            pc.delete_index(index_name)
            import time; time.sleep(5)
        pc.create_index(
            name=index_name, dimension=384, metric="cosine",
            spec=ServerlessSpec(cloud="aws", region="us-east-1"),
        )
        import time; time.sleep(10)
        qa_chains.clear()
        sid = session_id or get_session_id()
        return {"status": "Vector store reset successfully.", "success": True, "session_id": sid}
    except Exception as e:
        return {"status": f"Reset error: {e}", "success": False, "session_id": session_id or get_session_id()}


@app.get("/sessions/{session_id}/logs")
async def get_session_logs(session_id: str):
    return {
        "session_id": session_id,
        "uploads":    enhanced_logger.upload_logs.get(session_id, []),
        "queries":    enhanced_logger.query_logs.get(session_id, []),
        "active_chains": [k for k in qa_chains if k.startswith(session_id)],
    }


@app.get("/sessions/{session_id}/topics")
async def get_session_topics(session_id: str):
    topics = set()
    for key in qa_chains:
        if key.startswith(session_id):
            parts = key.split("_")
            if len(parts) >= 3:
                topics.add(parts[2])
    return {"session_id": session_id, "topics": list(topics)}


@app.get("/db-files/")
async def get_db_files(namespace: Optional[str] = None):
    """Return all unique files stored across all Pinecone indexes, grouped by content type."""
    from config import INDEX_CT_MAP
    from pathlib import Path

    try:
        existing_indexes = [i.name for i in pc.list_indexes()]
    except Exception as e:
        return {"files": [], "error": str(e)}

    all_files = []
    seen: set = set()

    for idx_name, ct in INDEX_CT_MAP.items():
        if idx_name not in existing_indexes:
            continue
        try:
            index = pc.Index(idx_name)
            stats = index.describe_index_stats()
            total = stats.get("total_vector_count", 0)
            if total == 0:
                continue
            dim = stats.get("dimension", 384)
            query_kwargs: dict = dict(
                vector=[0.0] * dim,
                top_k=min(total, 10000),
                include_values=False, include_metadata=True,
            )
            if namespace:
                query_kwargs["namespace"] = namespace
            res = index.query(**query_kwargs)

            file_chunks: Dict[tuple, int]  = {}
            file_meta:   Dict[tuple, dict] = {}

            for m in res.get("matches", []):
                meta     = m.get("metadata") or {}
                filename = meta.get("filename") or Path(meta.get("source", "")).name or "unknown"
                sid      = meta.get("session_id", "")
                key      = (filename, ct, sid)
                file_chunks[key] = file_chunks.get(key, 0) + 1
                if key not in file_meta:
                    file_meta[key] = {
                        "filename":     filename,
                        "content_type": ct,
                        "session_id":   sid,
                        "source":       meta.get("source", ""),
                    }

            for key, count in file_chunks.items():
                dedup_key = (key[0], key[1])
                if dedup_key in seen:
                    continue
                seen.add(dedup_key)
                entry = dict(file_meta[key])
                entry["chunk_count"] = count
                all_files.append(entry)

        except Exception as e:
            logger.warning(f"⚠️ db-files scan failed for {idx_name}: {e}")

    type_order = {"image": 0, "audio": 1, "video": 2, "text": 3}
    all_files.sort(key=lambda f: (type_order.get(f["content_type"], 9), f["filename"].lower()))

    return {"files": all_files, "total": len(all_files)}


# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)
