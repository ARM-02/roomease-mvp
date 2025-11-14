import streamlit as st
import sqlite3, uuid, json, os
from datetime import datetime

# ------------------ SQLite (PII + raw answers) ------------------
DB_PATH = "app.db"

def get_conn():
    conn = sqlite3.connect(DB_PATH)
    conn.execute("PRAGMA foreign_keys = ON")
    return conn

def init_db():
    with get_conn() as conn:
        conn.execute("""
        CREATE TABLE IF NOT EXISTS users (
          id INTEGER PRIMARY KEY AUTOINCREMENT,
          user_uuid TEXT UNIQUE NOT NULL,
          name TEXT NOT NULL,
          student_id TEXT NOT NULL,
          created_at TEXT NOT NULL
        )""")
        conn.execute("""
        CREATE TABLE IF NOT EXISTS responses (
          id INTEGER PRIMARY KEY AUTOINCREMENT,
          user_uuid TEXT NOT NULL,
          q1 TEXT NOT NULL, q2 TEXT NOT NULL, q3 TEXT NOT NULL, q4 TEXT NOT NULL, q5 TEXT NOT NULL,
          created_at TEXT NOT NULL,
          FOREIGN KEY(user_uuid) REFERENCES users(user_uuid) ON DELETE CASCADE
        )""")

def save_submission(name: str, student_id: str, answers: dict) -> str:
    user_uuid = str(uuid.uuid4())
    created = datetime.utcnow().isoformat()
    with get_conn() as conn:
        conn.execute(
            "INSERT INTO users (user_uuid, name, student_id, created_at) VALUES (?, ?, ?, ?)",
            (user_uuid, name, student_id, created),
        )
        conn.execute(
            "INSERT INTO responses (user_uuid, q1, q2, q3, q4, q5, created_at) VALUES (?, ?, ?, ?, ?, ?, ?)",
            (user_uuid, answers["q1"], answers["q2"], answers["q3"], answers["q4"], answers["q5"], created),
        )
    return user_uuid

def fetch_recent(limit: int = 5):
    with get_conn() as conn:
        cur = conn.execute("""
            SELECT u.user_uuid, u.name, u.student_id, r.q1, r.q2, r.q3, r.q4, r.q5, r.created_at
            FROM responses r JOIN users u ON u.user_uuid = r.user_uuid
            ORDER BY r.id DESC LIMIT ?
        """, (limit,))
        cols = [c[0] for c in cur.description]
        return [dict(zip(cols, row)) for row in cur.fetchall()]

# ------------------ Embeddings + Chroma ------------------
import numpy as np
from sentence_transformers import SentenceTransformer
import chromadb

@st.cache_resource(show_spinner=False)
def load_embed_model():
    # Downloads once per session, cached after
    return SentenceTransformer("all-MiniLM-L6-v2")

@st.cache_resource(show_spinner=False)
def get_chroma_client():
    # Dev/demo: local persistent folder INSIDE the Streamlit container
    # (ephemeral across redeploys; fine for demo)
    return chromadb.PersistentClient(path=".chroma")

def get_collections(client):
    roommates = client.get_or_create_collection("roommate_profiles", metadata={"hnsw:space":"cosine"})
    apartments = client.get_or_create_collection("apartment_listings", metadata={"hnsw:space":"cosine"})
    user_profiles = client.get_or_create_collection("user_profiles", metadata={"hnsw:space":"cosine"})
    return roommates, apartments, user_profiles

def profile_text_from_answers(name: str, answers: dict) -> str:
    """
    Turn free-text answers into a compact profile string.
    This is the text we embed; do NOT include student_id here.
    """
    # You can make this smarter later (summarize, extract traits, etc.)
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
    vec = model.encode([text], normalize_embeddings=True)[0]  # 384-d
    return vec.tolist()

def seed_demo_if_empty(roommates, apartments):
    """
    Put a couple of demo candidates into Chroma the first time the app runs.
    Safe to call on every start; it checks counts.
    """
    try:
        r_count = roommates.count()
        a_count = apartments.count()
    except Exception:
        # Older clients might not have count(); fallback by peeking
        r_count = len(roommates.peek()["ids"]) if roommates.peek()["ids"] else 0
        a_count = len(apartments.peek()["ids"]) if apartments.peek()["ids"] else 0

    if r_count == 0:
        demo_roommates = [
            {"name":"Andrea","quiet":True,"study":True,"guests":"low"},
            {"name":"Leo","quiet":False,"study":False,"guests":"high"},
            {"name":"Sara","quiet":True,"study":True,"guests":"low"},
        ]
        # We embed their short descriptions so they live in the same vector space
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
    """
    Query nearest neighbors from both collections and pair the best roommate + best apartment.
    """
    r = roommates.query(query_embeddings=[user_vec], n_results=3, include=["metadatas","distances","ids"])  # type: ignore
    a = apartments.query(query_embeddings=[user_vec], n_results=3, include=["metadatas","distances","ids"])  # type: ignore

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

    # MVP pairing: choose individually closest in each list
    best_rm = min(roommate_hits, key=lambda x: x["distance"])
    best_ap = min(apt_hits, key=lambda x: x["distance"])
    similarity = 1.0 - (best_rm["distance"] + best_ap["distance"]) / 2.0
    return {"roommate": best_rm, "apartment": best_ap, "similarity": similarity}

# ------------------ Streamlit UI ------------------
st.set_page_config(page_title="RoomEase+ (Chroma demo)", page_icon="🏡", layout="centered")
st.title("🏡 RoomEase+ — Save + Embed + Match")

init_db()  # ensure tables

with st.form("onboarding_form"):
    st.subheader("1) Your info")
    name = st.text_input("Name")
    student_id = st.text_input("Student ID")

    st.subheader("2) Personality (free-text answers)")
    q1 = st.text_area("1) Describe your ideal weekday.", height=80)
    q2 = st.text_area("2) How do you like to keep your living space?", height=80)
    q3 = st.text_area("3) Evenings: quiet time or social time?", height=80)
    q4 = st.text_area("4) Routines vs. spontaneity—what fits you?", height=80)
    q5 = st.text_area("5) Anything important for a roommate/apartment match?", height=80)

    submitted = st.form_submit_button("Save, embed, and see my match")

if submitted:
    # --- Validate ---
    if not name or not student_id:
        st.error("Please fill name and student ID.")
        st.stop()
    answers = { "q1": q1.strip(), "q2": q2.strip(), "q3": q3.strip(), "q4": q4.strip(), "q5": q5.strip() }
    if not all(answers.values()):
        st.error("Please answer all five questions (free-text).")
        st.stop()

    # --- 1) Save to SQLite (PII + raw answers) ---
    user_uuid = save_submission(name, student_id, answers)

    # --- 2) Build profile text & embed ---
    profile_text = profile_text_from_answers(name, answers)   # <— explanation above
    user_vec = embed_text(profile_text)                       # 384-d list[float]

    # --- 3) Connect to Chroma, seed demo data if needed ---
    client = get_chroma_client()
    roommates, apartments, user_profiles = get_collections(client)
    seed_demo_if_empty(roommates, apartments)

    # --- 4) Store user's vector in Chroma (no PII needed in metadata) ---
    user_profiles.upsert(
        ids=[user_uuid],
        documents=[profile_text],
        embeddings=[user_vec],
        metadatas=[{"name": name}]  # you can use {} to avoid any PII here
    )

    # --- 5) Query for similar roommates + apartments & display ---
    match = query_matches(roommates, apartments, user_vec)

    if not match:
        st.warning("No candidates found. (Seeding might have failed.)")
        st.stop()

    st.success("Saved, embedded, and matched!")
    st.markdown("### 3) Your Match")
    st.write(f"**Roommate:** {match['roommate'].get('name','(unknown)')}")
    st.write(f"**Apartment:** {match['apartment'].get('neighborhood','(n/a)')} · €{match['apartment'].get('rent','?')}")
    st.caption(f"Similarity ≈ {match['similarity']:.3f}")

    with st.expander("Debug: stored profile + recent DB rows"):
        st.json({
            "user_uuid": user_uuid,
            "profile_text_embedded": profile_text,
            "recent_rows": fetch_recent(5),
        })

st.caption("Note: On Streamlit Cloud this uses an ephemeral local Chroma store (.chroma). For persistence, switch to Chroma Cloud + Postgres.")
