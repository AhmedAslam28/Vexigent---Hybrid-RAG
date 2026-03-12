"""
llm_service.py — LLM provider wrappers and query engines.

Exports
-------
build_llm()             Build a plain LangChain LLM (for hybrid_rag_query).
build_agent_llm()       Build a LangChain ChatModel (for the ReAct agent).
hybrid_rag_query()      Hybrid Prompt + RAG query pipeline.
run_langchain_agent()   Full LangChain ReAct agent loop.
"""

import logging
import time
from typing import Any, Dict, List, Optional

import openai
import anthropic
import google.generativeai as genai

from langchain_core.language_models.llms import LLM
from langchain_core.callbacks.manager import CallbackManagerForLLMRun
from langchain_core.prompts import PromptTemplate as CorePromptTemplate
from langchain.tools import tool as lc_tool
from langchain.agents import AgentExecutor, create_react_agent
from langchain_core.documents import Document

from config import (
    OPENAI_API_KEY, ANTHROPIC_API_KEY, GOOGLE_API_KEY,
    get_domain_system_prompt,
)

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# LLM WRAPPERS  (used by hybrid_rag_query / build_llm)
# ─────────────────────────────────────────────────────────────────────────────

class OpenAILLM(LLM):
    model: str = "gpt-4o"
    temperature: float = 0
    max_retries: int = 2
    _client: Any = None

    def __init__(self, model="gpt-4o", temperature=0, max_retries=2, client=None):
        super().__init__()
        self.model = model
        self.temperature = temperature
        self.max_retries = max_retries
        object.__setattr__(
            self, "_client",
            client or openai.OpenAI(api_key=OPENAI_API_KEY, timeout=30.0, max_retries=2),
        )

    @property
    def _llm_type(self): return "openai"

    def _call(self, prompt, stop=None, run_manager: Optional[CallbackManagerForLLMRun] = None, **kwargs):
        client = object.__getattribute__(self, "_client")
        for attempt in range(self.max_retries):
            try:
                resp = client.chat.completions.create(
                    model=self.model,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=self.temperature,
                    max_tokens=2000,
                    timeout=45,
                )
                return resp.choices[0].message.content.strip()
            except Exception as e:
                if attempt == self.max_retries - 1:
                    return f"⚠️ OpenAI error: {e}"
                time.sleep(2 ** attempt)
        return "⚠️ OpenAI request failed."


class AnthropicLLM(LLM):
    model: str = "claude-sonnet-4-5"
    api_key: str = ""
    _client: Any = None

    def __init__(self, model="claude-sonnet-4-5", api_key=""):
        super().__init__()
        self.model = model
        self.api_key = api_key or ANTHROPIC_API_KEY or ""
        object.__setattr__(self, "_client", anthropic.Anthropic(api_key=self.api_key))

    @property
    def _llm_type(self): return "anthropic"

    def _call(self, prompt, stop=None, run_manager: Optional[CallbackManagerForLLMRun] = None, **kwargs):
        try:
            client = object.__getattribute__(self, "_client")
            resp = client.messages.create(
                model=self.model,
                max_tokens=2000,
                messages=[{"role": "user", "content": prompt}],
            )
            return resp.content[0].text
        except Exception as e:
            return f"⚠️ Anthropic error: {e}"


class GoogleLLM(LLM):
    model: str = "gemini-1.5-flash"
    api_key: str = ""
    _gclient: Any = None

    def __init__(self, model="gemini-1.5-flash", api_key=""):
        super().__init__()
        self.model = model
        self.api_key = api_key or GOOGLE_API_KEY or ""
        genai.configure(api_key=self.api_key)
        object.__setattr__(self, "_gclient", genai.GenerativeModel(model))

    @property
    def _llm_type(self): return "google"

    def _call(self, prompt, stop=None, run_manager: Optional[CallbackManagerForLLMRun] = None, **kwargs):
        try:
            client = object.__getattribute__(self, "_gclient")
            return client.generate_content(prompt).text
        except Exception as e:
            return f"⚠️ Gemini error: {e}"


def build_llm(provider: str, model_name: str, api_key: str = None) -> LLM:
    """Build a plain LangChain LLM for the hybrid_rag_query pipeline."""
    p = provider.lower()
    if p in ("anthropic", "claude"):
        key = api_key or ANTHROPIC_API_KEY or ""
        return AnthropicLLM(model=model_name or "claude-sonnet-4-5", api_key=key)
    elif p in ("gemini", "google"):
        key = api_key or GOOGLE_API_KEY or ""
        return GoogleLLM(model=model_name or "gemini-1.5-flash", api_key=key)
    else:
        key = api_key or OPENAI_API_KEY
        client = openai.OpenAI(api_key=key, timeout=30.0, max_retries=2)
        return OpenAILLM(model=model_name or "gpt-4o", client=client)


# ─────────────────────────────────────────────────────────────────────────────
# AGENT LLM BUILDERS  (LangChain ChatModel wrappers for the ReAct agent)
# ─────────────────────────────────────────────────────────────────────────────

def build_agent_llm(provider: str, model_name: str, api_key: str = None):
    """Return a LangChain BaseChatModel for the given provider."""
    p = provider.lower()

    if p in ("anthropic", "claude"):
        from langchain_anthropic import ChatAnthropic
        key = api_key or ANTHROPIC_API_KEY or ""
        return ChatAnthropic(
            model=model_name or "claude-sonnet-4-5",
            anthropic_api_key=key,
            max_tokens=2048,
            temperature=0,
        )

    elif p in ("gemini", "google"):
        from langchain_google_genai import ChatGoogleGenerativeAI
        key = api_key or GOOGLE_API_KEY or ""
        return ChatGoogleGenerativeAI(
            model=model_name or "gemini-1.5-flash",
            google_api_key=key,
            temperature=0,
        )

    else:  # openai / default
        from langchain_openai import ChatOpenAI
        key = api_key or OPENAI_API_KEY
        return ChatOpenAI(
            model=model_name or "gpt-4o",
            openai_api_key=key,
            temperature=0,
            max_tokens=2048,
        )


# ─────────────────────────────────────────────────────────────────────────────
# HYBRID PROMPT + RAG QUERY ENGINE
# ─────────────────────────────────────────────────────────────────────────────

def hybrid_rag_query(
    query: str,
    session_id: str,
    provider: str,
    model_name: str,
    domain: str,
    api_key: str = None,
    namespace: str = None,
    top_k: int = 8,
    use_reranker: bool = False,
    use_query_rewriter: bool = False,
    content_types: List[str] = None,
    enhanced_logger=None,
) -> Dict[str, Any]:
    """
    Hybrid Prompt + RAG pipeline:
      1. RAG retrieval from Pinecone
      2. Domain-aware system prompt injection
      3. Composed prompt → LLM call
      4. Return answer + explanation
    """
    # Import here to avoid circular dependency (rag_service imports config, not llm_service)
    from rag_service import rag_tool, _format_docs_for_prompt
    import os

    if content_types is None:
        content_types = ["text", "image", "audio", "video"]

    retrieved_docs = rag_tool(
        query=query,
        session_id=session_id,
        content_types=content_types,
        top_k=top_k,
        use_reranker=use_reranker,
        use_query_rewriter=use_query_rewriter,
        namespace=namespace,
    )

    system_prompt = get_domain_system_prompt(domain)
    context_text  = _format_docs_for_prompt(retrieved_docs)

    if retrieved_docs:
        final_prompt = (
            f"{system_prompt}\n\n"
            f"=== RETRIEVED DOCUMENT CONTEXT ===\n{context_text}\n"
            f"===================================\n\n"
            f"Question: {query}\n\n"
            f"Answer based strictly on the context above. "
            f"If the context does not contain a complete answer, say so and provide "
            f"whatever partial information is available."
        )
    else:
        final_prompt = (
            f"{system_prompt}\n\n"
            f"No documents have been uploaded yet, so there is no context to search.\n\n"
            f"Question: {query}\n\n"
            f"Please advise the user to upload relevant documents first, or answer "
            f"from your general knowledge if appropriate."
        )

    llm = build_llm(provider, model_name, api_key)
    answer = llm._call(final_prompt)

    explanation = (
        f"Provider: {provider} | Model: {model_name}\n"
        f"Domain: {domain}\n"
        f"RAG retrieved {len(retrieved_docs)} chunks | "
        f"Top-K: {top_k} | Reranker: {use_reranker} | Query rewriter: {use_query_rewriter}\n\n"
        f"Chunks used:\n"
    )
    for i, doc in enumerate(retrieved_docs):
        src   = doc.metadata.get("source", "?")
        fname = os.path.basename(src) if isinstance(src, str) else str(src)
        ct    = doc.metadata.get("content_type", "")
        explanation += f"\n[{i+1}] {fname} ({ct})\n{doc.page_content[:300].strip()}\n{'-'*40}\n"

    if enhanced_logger:
        enhanced_logger.log_query(session_id, query, f"{provider}/{model_name}", len(retrieved_docs))

    return {"answer": answer, "explanation": explanation, "success": True}


# ─────────────────────────────────────────────────────────────────────────────
# REACT PROMPT TEMPLATE
# ─────────────────────────────────────────────────────────────────────────────

REACT_TEMPLATE = """{system_prompt}

You have access to the following tools:

{tools}

TOOL SELECTION RULES — follow these exactly:
- "what do you see in the image" / "describe the image/photo/picture" / "who is in the image"
  → use search_images
- "what was said" / "summarize the speech" / "what happened at X seconds" / "first/last sentence"
  → use search_audio
- "what happens in the video" / "describe the video" / "what did they say in the video"
  → use search_video
- "summarize the PDF" / "what does the document say" / "find in the report"
  → use search_documents
- "how many images/files/documents" / "what have I uploaded" / "list my files"
  → use count_and_list_files
- "what content types are available" / "what is indexed"
  → use list_indexed_content

Use the following format EXACTLY:

Question: the input question you must answer
Thought: which tool should I use based on the question type?
Action: the action to take, must be one of [{tool_names}]
Action Input: the input to the action
Observation: the result of the action
Thought: I now have enough information to answer
Final Answer: [write your complete answer here]

CRITICAL: After receiving an Observation, go directly to Final Answer if you have enough information.
Do NOT call the same tool twice. Do NOT end on Thought alone — always follow with Action or Final Answer.

Begin!

Question: {input}
Thought:{agent_scratchpad}"""


# ─────────────────────────────────────────────────────────────────────────────
# LANGCHAIN REACT AGENT
# ─────────────────────────────────────────────────────────────────────────────

def run_langchain_agent(
    query: str,
    session_id: str,
    provider: str,
    model_name: str,
    domain: str,
    api_key: str = None,
    namespace: str = None,
    top_k: int = 8,
    use_reranker: bool = False,
    use_query_rewriter: bool = False,
    content_types: List[str] = None,
    max_iterations: int = 8,
    enhanced_logger=None,
    qa_chains: dict = None,
) -> Dict[str, Any]:
    """
    Real LangChain AgentExecutor with a genuine ReAct loop.
    Falls back to hybrid_rag_query on error.
    """
    from rag_service import build_agent_tools

    if content_types is None:
        content_types = ["text", "image", "audio", "video"]

    try:
        chat_llm = build_agent_llm(provider, model_name, api_key)

        tools = build_agent_tools(
            session_id=session_id,
            content_types=content_types,
            top_k=top_k,
            use_reranker=use_reranker,
            use_query_rewriter=use_query_rewriter,
            namespace=namespace,
            qa_chains=qa_chains or {},
        )

        system_prompt = get_domain_system_prompt(domain)
        prompt = CorePromptTemplate.from_template(
            REACT_TEMPLATE.replace("{system_prompt}", system_prompt)
        )

        agent = create_react_agent(llm=chat_llm, tools=tools, prompt=prompt)

        def _handle_parse_error(error) -> str:
            err_str = str(error)
            if "Final Answer:" in err_str:
                return err_str.split("Final Answer:")[-1].strip()
            if "Invalid Format:" in err_str:
                text = err_str.replace(
                    "Invalid Format: Missing 'Action:' after 'Thought:'", ""
                ).strip()
                if len(text) > 20:
                    return text
            return "I encountered a formatting issue. Please try rephrasing your question."

        executor = AgentExecutor(
            agent=agent,
            tools=tools,
            verbose=True,
            max_iterations=max_iterations,
            max_execution_time=60,
            handle_parsing_errors=_handle_parse_error,
            return_intermediate_steps=True,
        )

        result = executor.invoke({"input": query})
        answer = result.get("output", "No answer produced.")
        steps  = result.get("intermediate_steps", [])

        explanation = (
            f"Provider: {provider} | Model: {model_name}\n"
            f"Domain: {domain} | Agent: LangChain ReAct\n"
            f"Iterations: {len(steps)} | Top-K: {top_k} | "
            f"Reranker: {use_reranker} | Query rewriter: {use_query_rewriter}\n\n"
            f"Agent reasoning trace:\n"
        )
        for i, (action, observation) in enumerate(steps, 1):
            explanation += (
                f"\n── Step {i} ──────────────────────────\n"
                f"Tool called : {action.tool}\n"
                f"Tool input  : {action.tool_input}\n"
                f"Observation : {str(observation)[:400]}\n"
            )

        if enhanced_logger:
            enhanced_logger.log_query(
                session_id, query, f"agent/{provider}/{model_name}", len(steps)
            )

        return {"answer": answer, "explanation": explanation, "success": True}

    except Exception as e:
        logger.error(f"❌ LangChain agent error: {e}")
        logger.info("⚠️ Falling back to hybrid_rag_query")
        return hybrid_rag_query(
            query=query, session_id=session_id, provider=provider,
            model_name=model_name, domain=domain, api_key=api_key,
            top_k=top_k, use_reranker=use_reranker,
            use_query_rewriter=use_query_rewriter,
            content_types=content_types,
            enhanced_logger=enhanced_logger,
        )
