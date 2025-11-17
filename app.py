import streamlit as st
import requests
import uuid, os
from datetime import datetime

# ====================== App Config ======================
st.set_page_config(page_title="RoomEase+ — Chat Questionnaire", page_icon="🏡", layout="centered")

# ====================== Supabase (PII store) ======================
# Kept exactly as before so you can still persist identity.
from supabase import create_client

def get_supabase():
    try:
        url = os.environ["SUPABASE_URL"]
        key = os.environ["SUPABASE_KEY"]
    except KeyError as e:
        st.error(f"Missing secret: {e}. Add SUPABASE_URL and SUPABASE_KEY in Streamlit → ⋮ → Edit secrets.")
        st.stop()
    return create_client(url, key)

supabase = get_supabase()

def save_user(name: str, user_id: str) -> str:
    """Create a user record in Supabase and return a fresh user_uuid."""
    user_uuid = str(uuid.uuid4())
    created = datetime.utcnow().isoformat()
    # Table schema expected in Supabase:
    # users(user_uuid uuid unique, user_id text unique, name text, created_at timestamp default now())
    supabase.table("users").insert({
        "user_uuid": user_uuid,
        "user_id": user_id,
        "name": name,
        "created_at": created,
    }).execute()
    return user_uuid

# ====================== Questions ======================
QUESTIONS = [
    ("q1",  "Do you prefer quiet nights or social evenings?"),
    ("q2",  "How important is cleanliness to you?"),
    ("q3",  "Are you an early riser or a night owl?"),
    ("q4",  "Do you like hosting guests regularly?"),
    ("q5",  "How do you feel about sharing household items (e.g., cookware)?"),
    ("q6",  "Do you follow routines or prefer spontaneity?"),
    ("q7",  "How punctual are you for classes/appointments?"),
    ("q8",  "How do you handle conflicts with roommates?"),
    ("q9",  "How sensitive are you to noise when studying or sleeping?"),
    ("q10", "Do you prefer to plan your day ahead or go with the flow?"),
    ("q11", "How often do you need alone time to recharge?"),
    ("q12", "Do you prefer tidy minimal spaces or a lived‑in vibe?"),
]

CHOICES = [
    "Strongly prefer first option",
    "Prefer first option",
    "Neutral / depends",
    "Prefer second option",
    "Strongly prefer second option",
]

# ====================== Styles ======================
st.markdown(
    """
    <style>
      .bubble {border-radius: 16px; padding: 10px 14px; margin: 6px 0; max-width: 650px}
      .left  {background:#f1f5f9;}
      .right {background:#e7f3ff; margin-left:auto}
      .brand {font-weight:600}
      .card  {border:1px solid rgba(120,120,120,.2); border-radius:16px; padding:16px}
    </style>
    """,
    unsafe_allow_html=True,
)

# ====================== State ======================
if "mode" not in st.session_state:
    st.session_state.mode = "greeting"  # greeting → info → ask → send → done
if "idx" not in st.session_state:
    st.session_state.idx = 0
if "answers" not in st.session_state:
    st.session_state.answers = {}
if "user_uuid" not in st.session_state:
    st.session_state.user_uuid = None
if "endpoint" not in st.session_state:
    # You can also set RECOMMEND_API_URL in your env; we fall back to this state value.
    st.session_state.endpoint = os.getenv("RECOMMEND_API_URL", "")

# Sidebar controls
with st.sidebar:
    st.caption("Frontend collects answers only. Backend will score/match later.")
    st.session_state.endpoint = st.text_input("Backend base URL (optional)", st.session_state.endpoint, help="Example: https://api.example.com")
    if st.button("Reset Flow"):
        st.session_state.mode = "greeting"; st.session_state.idx = 0
        st.session_state.answers = {}; st.session_state.user_uuid = None
        st.experimental_rerun()

st.title("🏡 RoomEase+ — Chat Questionnaire")

# ====================== Greeting ======================
if st.session_state.mode == "greeting":
    st.markdown("""
    <div class='bubble left'><span class='brand'>RoomEase+</span>: Hello! How are you? Ready to answer a few personality questions?</div>
    """, unsafe_allow_html=True)
    c1, c2 = st.columns(2)
    if c1.button("Yes, let's go", type="primary"):
        st.session_state.mode = "info"
        st.experimental_rerun()
    if c2.button("Not now"):
        st.stop()

# ====================== Collect Name/ID ======================
if st.session_state.mode == "info":
    st.markdown("<div class='bubble left'>Great! First, what's your name and student ID?</div>", unsafe_allow_html=True)
    with st.form("pii_form", clear_on_submit=False):
        c1, c2 = st.columns(2)
        name = c1.text_input("Name")
        user_id = c2.text_input("User ID")
        start = st.form_submit_button("Start questions →", type="primary")
    if start:
        if not name or not user_id:
            st.warning("Please fill both name and user ID.")
        else:
            st.session_state.user_uuid = save_user(name, user_id)
            st.session_state.mode = "ask"
            st.experimental_rerun()

# ====================== Ask Questions (one by one) ======================
if st.session_state.mode == "ask":
    idx = st.session_state.idx
    qid, qtext = QUESTIONS[idx]

    st.progress((idx) / len(QUESTIONS))
    st.markdown(f"<div class='bubble left'><strong>Q{idx+1}.</strong> {qtext}</div>", unsafe_allow_html=True)

    # Use radio for discrete choices; raw text also works if you prefer
    choice = st.radio("Your answer", CHOICES, index=2, key=f"ans_{qid}")

    cols = st.columns(2)
    if cols[0].button("Next", type="primary"):
        st.session_state.answers[qid] = {"question": qtext, "answer": choice}
        if idx + 1 < len(QUESTIONS):
            st.session_state.idx += 1
            st.experimental_rerun()
        else:
            st.session_state.mode = "send"
            st.experimental_rerun()
    if cols[1].button("Back", disabled=(idx == 0)):
        st.session_state.idx = max(0, idx - 1)
        st.experimental_rerun()

# ====================== Send to Backend ======================
if st.session_state.mode == "send":
    payload = {
        "user_uuid": st.session_state.user_uuid,
        "answers": st.session_state.answers,
    }

    st.markdown("<div class='bubble left'>Awesome — wrapping things up…</div>", unsafe_allow_html=True)

    endpoint = st.session_state.endpoint.strip().rstrip("/")
    url = f"{endpoint}/recommend" if endpoint else None

    with st.spinner("Sending answers to backend…"):
        try:
            if url:
                # Safe best-effort POST; ignore the result per spec.
                _ = requests.post(url, json=payload, timeout=30)
        except Exception as e:
            st.info("Couldn't reach the backend yet (that's OK for now).")

    st.success("Your match is being analyzed…")
    st.caption("The backend will interpret, embed, match, and return the final recommendation when ready.")

    if st.button("Start over"):
        st.session_state.mode = "greeting"; st.session_state.idx = 0
        st.session_state.answers = {}; st.session_state.user_uuid = None
        st.experimental_rerun()
