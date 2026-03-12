"""
twili.py  —  Vexigent Platform Call Support Server
Built by Ahmed Aslam
=====================================================
Flow per caller turn:
  Twilio STT → Flask → Pinecone (retrieve context) → OpenAI gpt-4o → TwiML TTS

Install:
    pip install flask twilio openai pinecone-client python-dotenv

Required in .env:
    TWILIO_ACCOUNT_SID
    TWILIO_AUTH_TOKEN
    TWILIO_PHONE_NUMBER
    OPENAI_API_KEY
    PINECONE_API_KEY
    PUBLIC_URL              your ngrok https URL

Optional:
    PINECONE_ENV            default: us-east-1
    PINECONE_CALL_INDEX     default: call-support-index
    ESCALATION_NUMBER       E.164 number for human handoff
    MAX_TURNS               default: 20
"""

import os
from flask import Flask, request, jsonify, Response
from twilio.rest import Client
from twilio.twiml.voice_response import VoiceResponse, Gather, Dial
from openai import OpenAI
from pinecone import Pinecone
from dotenv import load_dotenv

load_dotenv()

app = Flask(__name__)

# ── Config ────────────────────────────────────────────────────────────────────
TWILIO_ACCOUNT_SID  = os.environ["TWILIO_ACCOUNT_SID"]
TWILIO_AUTH_TOKEN   = os.environ["TWILIO_AUTH_TOKEN"]
TWILIO_PHONE_NUMBER = os.environ.get("TWILIO_PHONE_NUMBER", "")
OPENAI_API_KEY      = os.environ["OPENAI_API_KEY"]
PINECONE_API_KEY    = os.environ["PINECONE_API_KEY"]
PINECONE_REGION     = os.environ.get("PINECONE_ENV", "us-east-1")
INDEX_NAME          = os.environ.get("PINECONE_CALL_INDEX", "call-support-index")
PUBLIC_URL          = os.environ.get("PUBLIC_URL", "").rstrip("/")
ESCALATION_NUMBER   = os.environ.get("ESCALATION_NUMBER", "")
MAX_TURNS           = int(os.environ.get("MAX_TURNS", "20"))

# ── Voice settings ────────────────────────────────────────────────────────────
# Polly.Joanna       = US female (original — feeble on phone)
# Polly.Matthew      = US male, deeper and clearer on phone calls
# Polly.Joanna-Neural = Neural version, much more natural (requires Twilio add-on)
# Polly.Raveena      = Indian-English female
VOICE = "Polly.Matthew"

# SSML prosody: rate="slow" gives user more time to process,
# volume="+6dB" makes voice noticeably louder on phone.
def ssml(text: str) -> str:
    """Wrap reply text in SSML prosody for clearer, slower, louder speech."""
    # Escape any special XML characters in the text
    safe = text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
    return (
        '<speak>'
        '<prosody rate="slow" volume="+6dB">'
        f'{safe}'
        '</prosody>'
        '</speak>'
    )

# ── Gather settings ───────────────────────────────────────────────────────────
# speech_timeout = seconds of silence AFTER speech ends before Twilio submits.
#   "auto" is too aggressive (cuts off mid-sentence). Use 3 seconds.
# timeout        = seconds to wait for speech to START before Gather fires.
#   Default is 5s — increase to 8s so users have time to think.
# action_on_empty_result = "true" stops Twilio submitting empty results immediately.
GATHER_KWARGS = dict(
    input="speech",
    method="POST",
    speech_timeout="3",          # wait 3s of silence after speech ends
    timeout=8,                   # wait 8s for speech to start
    action_on_empty_result="true",  # don't fire immediately on silence
    language="en-US",
    speech_model="phone_call",   # optimised STT model for phone audio quality
)

openai_client = OpenAI(api_key=OPENAI_API_KEY)
twilio_client = Client(TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN)
pc_index      = Pinecone(api_key=PINECONE_API_KEY).Index(INDEX_NAME)

# call_sid → {"history": [...], "turns": int, "silence": int}
call_sessions: dict = {}


# ── RAG: retrieve context from Pinecone ───────────────────────────────────────

def retrieve_context(question: str, top_k: int = 4) -> str:
    try:
        resp   = openai_client.embeddings.create(model="text-embedding-3-small", input=[question])
        vector = resp.data[0].embedding
        result = pc_index.query(vector=vector, top_k=top_k, include_metadata=True)
        chunks = [m["metadata"]["text"] for m in result["matches"] if m.get("metadata")]
        return "\n\n".join(chunks) if chunks else ""
    except Exception as e:
        print(f"[pinecone] error: {e}")
        return ""


# ── LLM: get reply from gpt-4o ────────────────────────────────────────────────

SYSTEM_PROMPT = """
You are an AI phone support agent for Vexigent, an AI document-query platform built by Ahmed Aslam.
If anyone asks who built or created Vexigent, say: "Vexigent was built by Ahmed Aslam."

THIS IS A VOICE PHONE CALL — rules you must follow:
- Reply in 1 to 2 sentences only. Maximum 40 words.
- Never use bullet points, lists, markdown, asterisks, or any symbols.
- Speak naturally and warmly, like a friendly support agent on the phone.
- Do not say "According to the documentation" or "Based on the context".
- Answer only from the context provided in the message.
- If the answer is not in the context, say: "I don't have that detail right now, but I can connect you to our team."
- If you cannot resolve an issue after 2 attempts, say exactly:
  "I'll connect you to a human agent who can help further."
""".strip()


def get_reply(history: list, question: str, context: str) -> str:
    try:
        messages = [{"role": "system", "content": SYSTEM_PROMPT}]
        messages.extend(history[-6:])   # last 3 exchanges for memory
        user_content = (
            f"Context from documentation:\n{context}\n\nCaller said: {question}"
            if context else
            f"Caller said: {question}"
        )
        messages.append({"role": "user", "content": user_content})

        resp = openai_client.chat.completions.create(
            model="gpt-4o",
            messages=messages,
            max_tokens=100,     # slightly more room for complete sentences
            temperature=0.2,    # lower = more factual, less creative
            timeout=14,
        )
        return resp.choices[0].message.content.strip()
    except Exception as e:
        print(f"[openai] error: {e}")
        return "I'm having a small technical issue. Please hold while I connect you to an agent."


def wants_human(text: str) -> bool:
    lowered = text.lower()
    return any(t in lowered for t in (
        "connect you to a human", "human agent", "live agent", "escalate", "transfer you"
    ))


# ── Routes ────────────────────────────────────────────────────────────────────

@app.route("/", methods=["GET"])
def index():
    return jsonify({
        "status": "running",
        "platform": "Vexigent Call Support",
        "built_by": "Ahmed Aslam",
        "pinecone_index": INDEX_NAME,
        "twilio_number": TWILIO_PHONE_NUMBER,
        "public_url": PUBLIC_URL or "⚠ NOT SET",
        "escalation": ESCALATION_NUMBER or "not configured",
    })


@app.route("/make-call", methods=["POST", "OPTIONS"])
def make_call():
    if request.method == "OPTIONS":
        return jsonify({}), 200
    data      = request.get_json(silent=True) or {}
    to_number = data.get("phone_number", "").strip()
    if not to_number:
        return jsonify({"error": "phone_number is required"}), 400
    if not to_number.startswith("+"):
        to_number = "+91" + "".join(filter(str.isdigit, to_number))
    if not PUBLIC_URL:
        return jsonify({"error": "PUBLIC_URL env var is not set"}), 500
    try:
        call = twilio_client.calls.create(
            to=to_number,
            from_=TWILIO_PHONE_NUMBER,
            url=f"{PUBLIC_URL}/voice-welcome",
        )
        print(f"[make-call] ✓ {call.sid} → {to_number}")
        return jsonify({"success": True, "call_sid": call.sid})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/voice-welcome", methods=["POST"])
def voice_welcome():
    call_sid = request.form.get("CallSid", "unknown")
    call_sessions[call_sid] = {"history": [], "turns": 0, "silence": 0}
    print(f"[welcome] call={call_sid}")

    resp   = VoiceResponse()
    gather = Gather(action=f"{PUBLIC_URL}/voice-respond", **GATHER_KWARGS)
    gather.say(
        ssml("Hello! Welcome to Vexigent support. How can I help you today?"),
        voice=VOICE,
    )
    resp.append(gather)
    resp.redirect(f"{PUBLIC_URL}/voice-silence")
    return Response(str(resp), mimetype="text/xml")


@app.route("/voice-respond", methods=["POST"])
def voice_respond():
    call_sid = request.form.get("CallSid", "unknown")
    speech   = request.form.get("SpeechResult", "").strip()

    session = call_sessions.setdefault(call_sid, {"history": [], "turns": 0, "silence": 0})

    if not speech:
        return _silence(call_sid)

    session["silence"] = 0
    session["turns"]  += 1
    print(f"[respond] call={call_sid} turn={session['turns']} speech='{speech}'")

    if session["turns"] > MAX_TURNS:
        return _goodbye("We've had a great conversation. Thank you for calling Vexigent. Goodbye!")

    context = retrieve_context(speech)
    reply   = get_reply(session["history"], speech, context)
    print(f"[respond] reply='{reply}'")

    session["history"].append({"role": "user",      "content": speech})
    session["history"].append({"role": "assistant",  "content": reply})

    if wants_human(reply):
        return _escalate(reply)

    resp   = VoiceResponse()
    gather = Gather(action=f"{PUBLIC_URL}/voice-respond", **GATHER_KWARGS)
    gather.say(ssml(reply), voice=VOICE)
    resp.append(gather)
    resp.redirect(f"{PUBLIC_URL}/voice-silence")
    return Response(str(resp), mimetype="text/xml")


@app.route("/voice-silence", methods=["POST"])
def voice_silence():
    call_sid = request.form.get("CallSid", "unknown")
    return _silence(call_sid)


# ── Helpers ───────────────────────────────────────────────────────────────────

def _silence(call_sid: str) -> Response:
    session = call_sessions.setdefault(call_sid, {"history": [], "turns": 0, "silence": 0})
    session["silence"] += 1
    if session["silence"] >= 2:
        return _goodbye("I haven't heard anything. Thank you for calling Vexigent. Goodbye!")
    resp   = VoiceResponse()
    gather = Gather(action=f"{PUBLIC_URL}/voice-respond", **GATHER_KWARGS)
    gather.say(ssml("Sorry, I didn't catch that. Could you please repeat your question?"), voice=VOICE)
    resp.append(gather)
    resp.redirect(f"{PUBLIC_URL}/voice-silence")
    return Response(str(resp), mimetype="text/xml")


def _goodbye(message: str) -> Response:
    resp = VoiceResponse()
    resp.say(ssml(message), voice=VOICE)
    resp.hangup()
    return Response(str(resp), mimetype="text/xml")


def _escalate(ai_message: str) -> Response:
    resp = VoiceResponse()
    resp.say(ssml(ai_message), voice=VOICE)
    if ESCALATION_NUMBER:
        resp.say(ssml("Connecting you now. Please hold for a moment."), voice=VOICE)
        dial = Dial()
        dial.number(ESCALATION_NUMBER)
        resp.append(dial)
    else:
        resp.say(ssml(
            "Our agents are unavailable right now. "
            "Please call back during business hours. Goodbye!"
        ), voice=VOICE)
        resp.hangup()
    return Response(str(resp), mimetype="text/xml")


@app.after_request
def add_cors(r):
    r.headers["Access-Control-Allow-Origin"]  = "*"
    r.headers["Access-Control-Allow-Headers"] = "Content-Type"
    r.headers["Access-Control-Allow-Methods"] = "GET,POST,OPTIONS"
    return r


# ── Entry ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("\n" + "=" * 62)
    print("  Vexigent Platform — Call Support Server")
    print("  Built by Ahmed Aslam")
    print("=" * 62)
    print(f"  Voice          : {VOICE}  (SSML rate=slow, +6dB)")
    print(f"  Speech timeout : 3s after speech ends, 8s to start")
    print(f"  Pinecone index : {INDEX_NAME}")
    print(f"  Twilio number  : {TWILIO_PHONE_NUMBER or '⚠ NOT SET'}")
    print(f"  Public URL     : {PUBLIC_URL or '⚠ NOT SET — run ngrok http 5000'}")
    print(f"  Escalation #   : {ESCALATION_NUMBER or 'not configured'}")
    print("=" * 62 + "\n")
    app.run(debug=True, port=5000)