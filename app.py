import streamlit as st
import sqlite3, uuid
from datetime import datetime

# ---------- Database helpers ----------
DB_PATH = "app.db"

def get_conn():
    """Open a connection to the SQLite file and enforce FK integrity."""
    conn = sqlite3.connect(DB_PATH)
    conn.execute("PRAGMA foreign_keys = ON")
    return conn

def init_db():
    """Create tables if they don't exist (idempotent)."""
    with get_conn() as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS users (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_uuid TEXT UNIQUE NOT NULL,
                name TEXT NOT NULL,
                student_id TEXT NOT NULL,
                created_at TEXT NOT NULL
            )
            """
        )
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS responses (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_uuid TEXT NOT NULL,
                q1 TEXT NOT NULL,
                q2 TEXT NOT NULL,
                q3 TEXT NOT NULL,
                q4 TEXT NOT NULL,
                q5 TEXT NOT NULL,
                created_at TEXT NOT NULL,
                FOREIGN KEY(user_uuid) REFERENCES users(user_uuid) ON DELETE CASCADE
            )
            """
        )

def save_submission(name: str, student_id: str, answers: dict) -> str:
    """Insert into users + responses in a single transaction. Returns user_uuid."""
    user_uuid = str(uuid.uuid4())
    created = datetime.utcnow().isoformat()

    with get_conn() as conn:
        # Insert PII
        conn.execute(
            "INSERT INTO users (user_uuid, name, student_id, created_at) VALUES (?, ?, ?, ?)",
            (user_uuid, name, student_id, created),
        )
        # Insert answers
        conn.execute(
            """
            INSERT INTO responses (user_uuid, q1, q2, q3, q4, q5, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (user_uuid, answers["q1"], answers["q2"], answers["q3"], answers["q4"], answers["q5"], created),
        )
    return user_uuid

def fetch_recent(limit: int = 5):
    """Read back the most recent saved responses for sanity-check in the UI."""
    with get_conn() as conn:
        cur = conn.execute(
            """
            SELECT u.user_uuid, u.name, u.student_id, r.q1, r.q2, r.q3, r.q4, r.q5, r.created_at
            FROM responses r
            JOIN users u ON u.user_uuid = r.user_uuid
            ORDER BY r.id DESC
            LIMIT ?
            """,
            (limit,),
        )
        cols = [c[0] for c in cur.description]
        return [dict(zip(cols, row)) for row in cur.fetchall()]

# ---------- Streamlit UI ----------
st.set_page_config(page_title="RoomEase+ MVP", page_icon="🏡", layout="centered")
st.title("🏡 RoomEase+ — MVP")
st.caption("Step 2: Save your inputs to a local SQLite database")

init_db()

with st.form("onboarding_form"):
    st.subheader("1) Your info")
    name = st.text_input("Name")
    student_id = st.text_input("Student ID")

    st.subheader("2) Personality (free-text answers)")
    q1 = st.text_area("1) Describe your ideal weekday.", height=80, placeholder="e.g., wake early, study blocks, gym…")
    q2 = st.text_area("2) How do you like to keep your living space?", height=80, placeholder="e.g., very tidy, somewhat relaxed…")
    q3 = st.text_area("3) Evenings: quiet time or social time?", height=80, placeholder="e.g., quiet reading, friends over sometimes…")
    q4 = st.text_area("4) Routines vs. spontaneity—what fits you?", height=80, placeholder="e.g., I follow a schedule / I go with the flow…")
    q5 = st.text_area("5) Anything important for a roommate/apartment match?", height=80, placeholder="noise, guests, budget, neighborhood…")

    submitted = st.form_submit_button("Save & see mock match")

if submitted:
    # Validate
    if not name or not student_id:
        st.error("Please fill name and student ID.")
        st.stop()
    answers = { "q1": q1.strip(), "q2": q2.strip(), "q3": q3.strip(), "q4": q4.strip(), "q5": q5.strip() }
    if not all(answers.values()):
        st.error("Please answer all five questions (free-text).")
        st.stop()

    # Persist to SQLite
    user_uuid = save_submission(name, student_id, answers)

    # Mock match (still UI-only for now)
    st.success("Saved! Here’s a mock match (we’ll make it real next).")
    st.markdown("### 3) Your Match")
    st.write("**Roommate:** Andrea")
    st.write("**Apartment:** Salamanca · €900")
    st.caption("(Next step: embed your answers + query Chroma)")

    with st.expander("Debug: recently saved (shows last 5)"):
        st.json(fetch_recent(5))
