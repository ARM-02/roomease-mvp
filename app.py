import streamlit as st
import uuid, os
from datetime import datetime

# ====================== Supabase (PII store) ======================
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
    # You could check res.data / res.error here if you want stricter handling
    return user_uuid

def fetch_recent_users(limit: int = 5):
    res = supabase.table("users").select("user_uuid,name,user_id,created_at").order("created_at", desc=True).limit(limit).execute()
    return res.data or []

# ====================== Chroma (embeddings + matching) ======================
from sentence_transformers import SentenceTransformer
import chromadb

@st.cache_resource(show_spinner=False)
def load_embed_model():
    # Downloads once per session in the Cloud container
    return SentenceTransformer("all-MiniLM-L6-v2")

@st.cache_resource(show_spinner=False)
def get_chroma_client():
    # Local persistent path inside container (ephemeral across redeploys).
    # Swap to Chroma Cloud by replacing this with chromadb.HttpClient(...)
    return chromadb.PersistentClient(path=".chroma")

def get_collections(client):
    roommates = client.get_or_create_collection("roommate_profiles", metadata={"hnsw:space":"cosine"})
    apartments = client.get_or_create_collection("apartment_listings", metadata={"hnsw:space":"cosine"})
    user_profiles = client.get_or_create_collection("user_profiles", metadata={"hnsw:space":"cosine"})
    return roommates, apartments, user_profiles

def profile_text_from_answers(name: str, answers: dict) -> str:
    # Do NOT include user_id or other PII — just the content that describes preferences.
    blocks = [
        f"ideal_weekday: {answers['q1']}",
        f"living_space: {answers['q2']}",
        f"evenings: {answers['q3']}",
        f"routines: {answers['q4']}",
        f"important: {answers['q5']}",
        "goals: quiet, compatibility, study-friendly"
    ]
    return " | ".join(blocks)

def embed_text(text: str) -> list[float]:
    model = load_embed_model()
    vec = model.encode([text], normalize_embeddings=True)[0]  # 384-dim vector
    return vec.tolist()

def seed_demo_if_empty(roommates, apartments):
    # Seed 3 roommates
    try:
        r_count = roommates.count()
        a_count = apartments.count()
    except Exception:
        peek_r = roommates.peek()
        peek_a = apartments.peek()
        r_count = len(peek_r.get("ids", []) or [])
        a_count = len(peek_a.get("ids", []) or [])

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

# ====================== UI ======================
st.set_page_config(page_title="RoomEase+ (Supabase + Chroma)", page_icon="🏡", layout="centered")
st.title("🏡 RoomEase+ — PII in Supabase, answers in Chroma, real match")

with st.form("onboarding_form"):
    st.subheader("1) Your info")
    name = st.text_input("Name")
    user_id = st.text_input("User ID")  # your external/student ID (PII)

    st.subheader("2) Personality (free-text answers)")
    q1 = st.text_area("1) Describe your ideal weekday.", height=80)
    q2 = st.text_area("2) How do you like to keep your living space?", height=80)
    q3 = st.text_area("3) Evenings: quiet time or social time?", height=80)
    q4 = st.text_area("4) Routines vs. spontaneity—what fits you?", height=80)
    q5 = st.text_area("5) Anything important for a roommate/apartment match?", height=80)

    submitted = st.form_submit_button("Save (PII), embed answers, and see my match")

if submitted:
    # ---- Validate inputs ----
    if not name or not user_id:
        st.error("Please fill name and user ID.")
        st.stop()
    answers = {"q1": q1.strip(), "q2": q2.strip(), "q3": q3.strip(), "q4": q4.strip(), "q5": q5.strip()}
    if not all(answers.values()):
        st.error("Please answer all five questions (free-text).")
        st.stop()

    # ---- 1) Save ONLY PII to Supabase ----
    user_uuid = save_user(name, user_id)

    # ---- 2) Convert answers → text profile → embedding ----
    profile_text = profile_text_from_answers(name, answers)
    user_vec = embed_text(profile_text)

    # ---- 3) Chroma: upsert user + seed collections if empty ----
    client = get_chroma_client()
    roommates, apartments, user_profiles = get_collections(client)
    seed_demo_if_empty(roommates, apartments)

    user_profiles.upsert(
        ids=[user_uuid],
        documents=[profile_text],
        embeddings=[user_vec],
        metadatas=[{"name": name, "user_id": user_id}]  # you can drop PII here if you prefer
    )

    # ---- 4) Query for a match ----
    match = query_matches(roommates, apartments, user_vec)
    if not match:
        st.warning("No candidates found. (Seeding might have failed.)")
        st.stop()

    st.success("Saved (PII), embedded (answers), and matched!")
    st.markdown("### 3) Your Match")
    st.write(f"**Roommate:** {match['roommate'].get('name','(unknown)')}")
    st.write(f"**Apartment:** {match['apartment'].get('neighborhood','(n/a)')} · €{match['apartment'].get('rent','?')}")
    st.caption(f"Similarity ≈ {match['similarity']:.3f}")

    with st.expander("Debug: recent users (from Supabase) + profile text sent to Chroma"):
        st.json({
            "recent_users": fetch_recent_users(5),
            "user_uuid": user_uuid,
            "profile_text_embedded": profile_text
        })

st.caption("Architecture: Supabase stores identity (name,user_id,user_uuid). Chroma stores embedded text of answers for matching.")
