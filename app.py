import streamlit as st

st.set_page_config(page_title="RoomEase+ MVP", page_icon="🏡", layout="centered")
st.title("🏡 RoomEase+ — MVP")
st.caption("Step 1: UI only (we'll connect storage next)")

with st.form("onboarding_form"):
    st.subheader("1) Your info")
    name = st.text_input("Name")
    student_id = st.text_input("Student ID")

    st.subheader("2) Personality (free-text answers)")
    q1 = st.text_area("1) Describe your ideal weekday.", height=80, placeholder="e.g., wake up early, study blocks, gym…")
    q2 = st.text_area("2) How do you like to keep your living space?", height=80, placeholder="e.g., very tidy, somewhat relaxed…")
    q3 = st.text_area("3) Evenings: quiet time or social time?", height=80, placeholder="e.g., quiet reading, friends over sometimes…")
    q4 = st.text_area("4) Routines vs. spontaneity—what fits you?", height=80, placeholder="e.g., I follow a schedule / I go with the flow…")
    q5 = st.text_area("5) Anything important for a roommate/apartment match?", height=80, placeholder="noise, guests, budget, neighborhood…")

    submitted = st.form_submit_button("See my match")

if submitted:
    if not name or not student_id:
        st.error("Please fill name and student ID.")
        st.stop()

    # Basic validation for empty answers (optional but helpful)
    answers = {"q1": q1.strip(), "q2": q2.strip(), "q3": q3.strip(), "q4": q4.strip(), "q5": q5.strip()}
    if not all(answers.values()):
        st.error("Please answer all five questions (free-text).")
        st.stop()

    # UI-only: mock a result to prove the flow works end-to-end
    st.success("Match found!")
    st.markdown("### 3) Your Match")
    st.write("**Roommate:** Andrea")
    st.write("**Apartment:** Salamanca · €900")
    st.caption("(This is a mock result — next step: connect SQLite + Chroma)")

    with st.expander("Debug: Your inputs"):
        st.json({
            "name": name,
            "student_id": student_id,
            "answers": answers
        })

