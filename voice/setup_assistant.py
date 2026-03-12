"""
setup_assistant.py  —  Index your 3 docs into Pinecone + create OpenAI Assistant
==================================================================================
This replaces the broken Vector Store approach with Pinecone — which you already
have working in your main platform.

What it does:
  1. Reads your 3 .docx files
  2. Chunks + embeds them
  3. Upserts into a Pinecone index  (call-support-index)
  4. Creates a plain OpenAI Assistant (no File Search, no Vector Store)

Run once:
    python setup_assistant.py

Required in .env:
    OPENAI_API_KEY
    PINECONE_API_KEY

Prints:
    OPENAI_ASSISTANT_ID  — paste into .env
"""

import os
import time
from dotenv import load_dotenv
load_dotenv()

import openai
from openai import OpenAI
from pinecone import Pinecone, ServerlessSpec
from docx import Document as DocxDocument

OPENAI_API_KEY  = os.environ["OPENAI_API_KEY"]
PINECONE_API_KEY = os.environ["PINECONE_API_KEY"]
PINECONE_REGION  = os.environ.get("PINECONE_ENV", "us-east-1")

INDEX_NAME = "call-support-index"

DOC_FILES = [
    "platform_documentation.docx",
    "faq.docx",
    "troubleshooting.docx",
]

openai_client = OpenAI(api_key=OPENAI_API_KEY)
pc = Pinecone(api_key=PINECONE_API_KEY)

ASSISTANT_NAME = "Vexigent Platform Support Agent"
ASSISTANT_INSTRUCTIONS = """
You are an AI phone support agent for Vexigent, an AI document-query platform built by Ahmed Aslam.
If anyone asks who built or created Vexigent, say: "Vexigent was built by Ahmed Aslam."

THIS IS A VOICE PHONE CALL — follow these rules exactly:
- Reply in 1 to 2 short sentences only. Maximum 35 words.
- Never use bullet points, numbered lists, markdown, or symbols.
- Speak naturally and warmly like a helpful support agent on the phone.
- Do not say "According to the documentation".

ANSWER RULES:
- Answer only from the context provided to you in each message.
- If the answer is not in the context say: "I don't have that detail right now."
- If you cannot resolve an issue after 2 tries say exactly:
  "I'll connect you to a human agent who can help further."
""".strip()


# ── Step 1: Read .docx files ──────────────────────────────────────────────────

def read_docx(path):
    doc = DocxDocument(path)
    return "\n".join(p.text for p in doc.paragraphs if p.text.strip())


# ── Step 2: Chunk text ────────────────────────────────────────────────────────

def chunk_text(text, source, chunk_size=400, overlap=50):
    words = text.split()
    chunks = []
    i = 0
    while i < len(words):
        chunk = " ".join(words[i:i+chunk_size])
        chunks.append({"text": chunk, "source": source})
        i += chunk_size - overlap
    return chunks


# ── Step 3: Embed with OpenAI ─────────────────────────────────────────────────

def embed(texts):
    resp = openai_client.embeddings.create(
        model="text-embedding-3-small",
        input=texts,
    )
    return [r.embedding for r in resp.data]


# ── Step 4: Upsert to Pinecone ────────────────────────────────────────────────

def setup_pinecone_index():
    existing = [i.name for i in pc.list_indexes()]
    if INDEX_NAME not in existing:
        print(f"  Creating Pinecone index '{INDEX_NAME}' ...")
        pc.create_index(
            name=INDEX_NAME,
            dimension=1536,   # text-embedding-3-small dimension
            metric="cosine",
            spec=ServerlessSpec(cloud="aws", region=PINECONE_REGION),
        )
        time.sleep(10)
        print("  ✓  Index created")
    else:
        print(f"  ✓  Index '{INDEX_NAME}' already exists")
    return pc.Index(INDEX_NAME)


def upsert_chunks(index, chunks):
    BATCH = 50
    for i in range(0, len(chunks), BATCH):
        batch = chunks[i:i+BATCH]
        texts = [c["text"] for c in batch]
        embeddings = embed(texts)
        vectors = [
            {
                "id": f"chunk-{i+j}",
                "values": embeddings[j],
                "metadata": {"text": batch[j]["text"], "source": batch[j]["source"]},
            }
            for j in range(len(batch))
        ]
        index.upsert(vectors=vectors)
        print(f"  Upserted chunks {i} – {i+len(batch)}")


# ── Step 5: Create plain OpenAI Assistant (no Vector Store) ───────────────────

def create_assistant():
    print(f"\nCreating Assistant '{ASSISTANT_NAME}' ...")
    assistant = openai_client.beta.assistants.create(
        name=ASSISTANT_NAME,
        instructions=ASSISTANT_INSTRUCTIONS,
        model="gpt-4o",
    )
    return assistant.id


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    print("=" * 62)
    print("  Vexigent Platform — Call Support Setup")
    print("  Built by Ahmed Aslam")
    print("=" * 62)

    # 1. Read and chunk docs
    print("\n[1/3] Reading and chunking documentation files ...")
    all_chunks = []
    for path in DOC_FILES:
        if not os.path.exists(path):
            print(f"  ⚠  Not found, skipping: {path}")
            continue
        text = read_docx(path)
        chunks = chunk_text(text, source=path)
        all_chunks.extend(chunks)
        print(f"  ✓  {path}  →  {len(chunks)} chunks")

    if not all_chunks:
        print("\n⚠  No files found. Place the 3 .docx files in the same folder.")
        return

    # 2. Setup Pinecone and upsert
    print(f"\n[2/3] Indexing {len(all_chunks)} chunks into Pinecone ...")
    index = setup_pinecone_index()
    upsert_chunks(index, all_chunks)
    print(f"  ✓  All chunks indexed into '{INDEX_NAME}'")

    # 3. Create assistant
    print("\n[3/3] Creating OpenAI Assistant ...")
    asst_id = create_assistant()

    print("\n" + "=" * 62)
    print("  ✓  Done! Add this to your .env file:\n")
    print(f"  OPENAI_ASSISTANT_ID={asst_id}")
    print(f"  PINECONE_CALL_INDEX={INDEX_NAME}")
    print("\n  Then run:  python twili.py")
    print("=" * 62)


if __name__ == "__main__":
    main()