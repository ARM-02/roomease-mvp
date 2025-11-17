import streamlit as st
import uuid, os
from datetime import datetime
from dataclasses import dataclass
from typing import Dict, List

# ====================== 🎯 Config & Theming ======================
st.set_page_config(
    page_title="RoomEase+ Pro",
    page_icon="🏡",
    layout="wide",
    menu_items={
        "About": "RoomEase+ Pro — demo app that stores PII in Supabase and embeds preference signals into Chroma for matching."
    },
)

# --- Minimal dark-ish polish without requiring a custom theme ---
st.markdown(
    """
    <style>
      .re-card {border:1px solid rgba(120,120,120,.2); border-radius: 16px; padding: 18px; background: rgba(255,255,255,.65); backdrop-filter: blur(8px);} 
      .re-pill {display:inline-block; padding:4px 10px; border-radius:999px; background:#eef1ff; font-size:12px; margin-right:6px}
      .re-hero {border-radius: 20px; padding: 24px; background: linear-gradient(135deg, #f1f5ff 0%, #eefcf5 100%);} 
      .stSlider > div[data-baseweb="slider"]{ margin-top: -10px;}
    </style>
    """,
    unsafe_allow_html=True,
)

# ====================== 🔐 Supabase (PII store) ======================
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
    res = supabase.table("users").insert({
        "user_uuid": user_uuid,
        "user_id": user_id,
        "name": name,
        "created_at": created
    }).execute()
    # (Optional) inspect res.data / res.error here
    return user_uuid


def fetch_recent_users(limit: int = 5):
    res = supabase.table("users").select("user_uuid,name,user_id,created_at").order("created_at", desc=True).limit(limit).execute()
    return res.data or []

# ====================== 📊 Chroma (embeddings + matching) ======================
from sentence_transformers import SentenceTransformer
import chromadb

@st.cache_resource(show_spinner=False)
def load_embed_model():
    return SentenceTransformer("all-MiniLM-L6-v2")

@st.cache_resource(show_spinner=False)
def get_chroma_client():
    return chromadb.PersistentClient(path=".chroma")


def get_collections(client):
    roommates = client.get_or_create_collection("roommate_profiles", metadata={"hnsw:space":"cosine"})
    apartments = client.get_or_create_collection("apartment_listings", metadata={"hnsw:space":"cosine"})
    user_profiles = client.get_or_create_collection("user_profiles", metadata={"hnsw:space":"cosine"})
    return roommates, apartments, user_profiles


# ====================== 🧠 Questions (Big Five flavored) ======================
@dataclass
class Q:
    id: str
    text: str
    trait: str

QUESTIONS: List[Q] = [
    Q("q1",  "Do you like trying new experiences, foods, or activities — or do you prefer routines you already enjoy?", "Openness"),
    Q("q2",  "How do you feel about spontaneous plans or surprises?",                                     "Openness"),
    Q("q3",  "Do you like to keep your space organized and plan your day ahead, or do you go with the flow?", "Conscientiousness"),
    Q("q4",  "How punctual are you for classes or appointments?",                                          "Conscientiousness"),
    Q("q5",  "When you come home after a long day, do you enjoy chatting with roommates or prefer quiet time alone?", "Extraversion"),
    Q("q6",  "Would you enjoy hosting small gatherings or prefer a calm apartment?",                       "Extraversion"),
    Q("q7",  "When conflicts arise, do you usually try to find compromise or prefer to stand your ground?", "Agreeableness"),
    Q("q8",  "Would you describe yourself as easygoing and patient?",                                      "Agreeableness"),
    Q("q9",  "When things go wrong (like a noisy neighbor or exam stress), do you get anxious easily or stay calm?", "Neuroticism"),
    Q("q10", "How often do you need alone time to recharge?",                                               "Neuroticism / Extraversion"),
]

TRAIT_ORDER = ["Openness", "Conscientiousness", "Extraversion", "Agreeableness", "Neuroticism"]


def trait_buckets(scores: Dict[str, List[int]]):
    """Average 1-5 scores per trait and return a compact descriptor string used for embeddings."""
    avgs = {}
    for t, vals in scores.items():
        if not vals:
            continue
        avgs[t] = sum(vals) / len(vals)
    # Round to one decimal for readability
    parts = [f"{t}:{avgs.get(t,0):.1f}" for t in TRAIT_ORDER]
    return "; ".join(parts), avgs


# ====================== 🔎 Embedding helpers ======================

def embed_text(text: str) -> List[float]:
    model = load_embed_model()
    vec = model.encode([text], normalize_embeddings=True)[0]
    return vec.tolist()


def seed_demo_if_empty(roommates, apartments):
    # Defensive counts for both older & newer Chroma client versions
    try:
        r_count = roommates.count(); a_count = apartments.count()
    except Exception:
        r_count = len(roommates.peek().get("ids", []) or [])
        a_count = len(apartments.peek().get("ids", []) or [])

    if r_count == 0:
        demo_roommates = [
            {"name":"Andrea","quiet":True,"study":True,"guests":"low"},
            {"name":"Leo","quiet":False,"study":False,"guests":"high"},
            {"name":"Sara","quiet":True,"study":True,"guests":"low"},
        ]
        ids, docs, metas, vecs = [], [], [], []
        for i, rm in enumerate(demo_roommates):
            text = f"name={rm['name']} quiet={rm['quiet']} study={rm['study']} guests={rm['guests']}"
            ids.append(f"rm_{i}")
            docs.append(text)
            metas.append(rm)
            vecs.append(embed_text(text))
        roommates.upsert(ids=ids, documents=docs, metadatas=metas, embeddings=vecs)

    if a_count == 0:
        demo_apartments = [
            {"neighborhood":"Salamanca","rent":900,"quiet":True},
            {"neighborhood":"Chamberí","rent":800,"quiet":True},
            {"neighborhood":"Malasaña","rent":950,"quiet":False},
        ]
        ids, docs, metas, vecs = [], [], [], []
        for i, ap in enumerate(demo_apartments):
            text = f"neighborhood={ap['neighborhood']} rent={ap['rent']} quiet={ap['quiet']}"
            ids.append(f"ap_{i}")
            docs.append(text)
            metas.append(ap)
            vecs.append(embed_text(text))
        apartments.upsert(ids=ids, documents=docs, metadatas=metas, embeddings=vecs)


def query_matches(roommates, apartments, user_vec):
    r = roommates.query(query_embeddings=[user_vec], n_results=3, include=["metadatas","distances"])
    a = apartments.query(query_embeddings=[user_vec], n_results=3, include=["metadatas","distances"])

    roommate_hits = [
        {"id": rid, **meta, "distance": dist}
        for rid, meta, dist in zip(r["ids"][0], r["metadatas"][0], r["distances"][0])
    ]
    apt_hits = [
        {"id": aid, **meta, "distance": dist}
        for aid, meta, dist in zip(a["ids"][0], a["metadatas"][0], a["distances"][0])
    ]
    if not roommate_hits or not apt_hits:
        return None
    best_rm = min(roommate_hits, key=lambda x: x["distance"]) 
    best_ap = min(apt_hits, key=lambda x: x["distance"]) 
    similarity = 1.0 - (best_rm["distance"] + best_ap["distance"]) / 2.0
    return {"roommate": best_rm, "apartment": best_ap, "similarity": similarity}


# ====================== 🧩 App Pages ======================
PAGES = ["Welcome", "Questionnaire", "Results", "Admin"]
page = st.sidebar.radio("Navigation", PAGES, index=0)

st.sidebar.markdown("---")
st.sidebar.caption("PII → Supabase • Answers → Chroma embeddings • Local demo data for matches")

# --- Shared state across pages ---
if "last_match" not in st.session_state:
    st.session_state.last_match = None
if "user_uuid" not in st.session_state:
    st.session_state.user_uuid = None
if "trait_avgs" not in st.session_state:
    st.session_state.trait_avgs = {}


# ====================== 🚪 Welcome ======================
if page == "Welcome":
    st.markdown("""
    <div class='re-hero'>
      <h1>🏡 RoomEase+ Pro</h1>
      <p>Find a compatible roommate & a quiet, study‑friendly apartment. Identity is stored safely in Supabase, while your questionnaire signals are embedded (no PII) into Chroma for matching.</p>
      <div class='re-pill'>Supabase</div>
      <div class='re-pill'>Chroma</div>
      <div class='re-pill'>SentenceTransformers</div>
      <div class='re-pill'>Streamlit</div>
    </div>
    """, unsafe_allow_html=True)

    c1, c2, c3 = st.columns([1.2,1,1])
    with c1:
        st.markdown("### How it works")
        st.write("1) Add your name & ID (PII) → stored in Supabase.")
        st.write("2) Answer 10 quick questions (1–5 scale).")
        st.write("3) We embed the resulting trait profile → Chroma → retrieve best roommate & apartment from demo pool.")
    with c2:
        st.metric("Questions", 10)
        st.metric("Traits Covered", 5)
    with c3:
        st.metric("Demo Roommates", 3)
        st.metric("Demo Apartments", 3)

# ====================== 📝 Questionnaire ======================
if page == "Questionnaire":
    st.header("Your info")
    with st.container():
        c1, c2 = st.columns(2)
        name = c1.text_input("Name")
        user_id = c2.text_input("User ID")

    st.markdown("---")
    st.subheader("Personality & living preferences")
    st.caption("Use the sliders: 1 = strongly prefer the first option, 5 = strongly prefer the second option. 3 = neutral.")

    scores: Dict[str, List[int]] = {t: [] for t in TRAIT_ORDER}
    answers: Dict[str, int] = {}

    for q in QUESTIONS:
        with st.expander(q.text, expanded=False):
            val = st.slider(q.id, min_value=1, max_value=5, value=3, step=1)
            answers[q.id] = val
            # Split hybrid trait label into primary key bucket
            primary_trait = q.trait.split("/")[0].strip()
            if primary_trait not in scores:
                scores[primary_trait] = []
            scores[primary_trait].append(val)

    extra_notes = st.text_area("Anything else you'd want us to consider? (optional)", height=80)

    submitted = st.button("Save, Embed & Get My Match", type="primary")

    if submitted:
        if not name or not user_id:
            st.error("Please fill name and user ID.")
            st.stop()

        # 1) Save only PII in Supabase
        user_uuid = save_user(name, user_id)
        st.session_state.user_uuid = user_uuid

        # 2) Build a compact profile text for embedding
        trait_str, avgs = trait_buckets(scores)
        st.session_state.trait_avgs = avgs
        profile_text = f"traits=[{trait_str}] | notes={extra_notes.strip()}"
        user_vec = embed_text(profile_text)

        # 3) Chroma: upsert + seed demo if empty
        client = get_chroma_client()
        roommates, apartments, user_profiles = get_collections(client)
        seed_demo_if_empty(roommates, apartments)

        user_profiles.upsert(
            ids=[user_uuid],
            documents=[profile_text],
            embeddings=[user_vec],
            metadatas=[{"name": name, "user_id": user_id}],  # you can drop PII here if you prefer
        )

        # 4) Query match
        match = query_matches(roommates, apartments, user_vec)
        if not match:
            st.warning("No candidates found. (Seeding might have failed.)")
            st.stop()

        st.session_state.last_match = match
        st.success("Saved (PII), embedded (answers), and matched! Switch to the Results tab →")

# ====================== ✅ Results ======================
if page == "Results":
    match = st.session_state.get("last_match")
    if not match:
        st.info("Run the questionnaire first to see your match.")
    else:
        st.markdown("### Your Match")
        c1, c2, c3 = st.columns([1.2,1,1])
        with c1:
            st.markdown("<div class='re-card'>", unsafe_allow_html=True)
            st.subheader("👤 Roommate")
            st.write(f"**Name:** {match['roommate'].get('name','(unknown)')}")
            st.write(f"**Quiet:** {match['roommate'].get('quiet')}")
            st.write(f"**Guests:** {match['roommate'].get('guests')}")
            st.markdown("</div>", unsafe_allow_html=True)
        with c2:
            st.markdown("<div class='re-card'>", unsafe_allow_html=True)
            st.subheader("🏘️ Apartment")
            st.write(f"**Neighborhood:** {match['apartment'].get('neighborhood','(n/a)')}")
            st.write(f"**Rent:** €{match['apartment'].get('rent','?')}")
            st.write(f"**Quiet:** {match['apartment'].get('quiet')}")
            st.markdown("</div>", unsafe_allow_html=True)
        with c3:
            st.markdown("<div class='re-card'>", unsafe_allow_html=True)
            st.subheader("📈 Similarity")
            st.metric("Score (0–1)", f"{match['similarity']:.3f}")
            if st.session_state.trait_avgs:
                for t in TRAIT_ORDER:
                    if t in st.session_state.trait_avgs:
                        st.progress(min(max((st.session_state.trait_avgs[t]-1)/4,0),1), text=t)
            st.markdown("</div>", unsafe_allow_html=True)

        st.caption("Similarity is computed as 1 - average cosine distance to the top roommate and apartment candidates.")

# ====================== 🛠️ Admin ======================
if page == "Admin":
    st.subheader("Recent users (from Supabase)")
    data = fetch_recent_users(10)
    st.json(data)

    st.subheader("Debug notes")
    st.write("- Only PII (name, user_id, user_uuid) goes to Supabase.\n- Trait string + optional notes are embedded to Chroma.")
