import streamlit as st

st.set_page_config(page_title="RoomEase+ MVP", page_icon="🏡", layout="centered")
st.title("🏡 RoomEase+ — MVP")
st.caption("Step 1: UI only (we'll connect storage next)")

with st.form("onboarding_form"):
    st.subheader("1) Your info")
    name = st.text_input("Name")
    student_id = st.text_input("Student ID")

    st.subheader("2) Personality (1–5)")
    q1 = st.slider("I enjoy social gatherings.", 1, 5, 3)
    q2 = st.slider("I keep a tidy, organized space.", 1, 5, 4)
    q3 = st.slider("I prefer quiet evenings over parties.", 1, 5, 4)
    q4 = st.slider("I stick to schedules and routines.", 1, 5, 4)
    q5 = st.slider("I like spontaneous activities.", 1, 5, 3)

    submitted = st.form_submit_button("See my match")

if submitted:
    if not name or not student_id:
        st.error("Please fill name and student ID.")
        st.stop()

    # UI-only: mock a result to prove the flow works end-to-end
    st.success("Match found!")
    st.markdown("### 3) Your Match")
    st.write("**Roommate:** Andrea")
    st.write("**Apartment:** Salamanca · €900")
    st.caption("(This is a mock result — next step: connect SQLite + Chroma)")

    with st.expander("Debug: Your answers"):
        st.json({
            "name": name,
            "student_id": student_id,
            "answers": {"q1": q1, "q2": q2, "q3": q3, "q4": q4, "q5": q5}
        })
