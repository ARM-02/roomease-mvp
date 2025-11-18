import streamlit as st
import time

# Page config
st.set_page_config(
    page_title="AI Roommate Matcher",
    page_icon="🏠",
    layout="centered"
)

# Custom CSS for the chat interface
st.markdown("""
    <style>
    /* General Styling - Fix for white background */
    html, body, [class*="st"] {
        color: #31333F !important; /* Dark charcoal for high readability */
    }

    /* Chat Message Styling */
    .stChatMessage {
        background-color: transparent; 
    }

    /* User Message Bubble */
    .stChatMessage[data-testid="stChatMessage"]:nth-child(odd) {
        background-color: #f0f2f6; /* Light grey bubble for user */
        border: 1px solid #e0e0e0;
        border-radius: 15px;
        padding: 15px;
        margin-bottom: 10px;
    }

    /* Assistant Message Bubble */
    .stChatMessage[data-testid="stChatMessage"]:nth-child(even) {
        background-color: transparent;
        padding-top: 10px;
        padding-bottom: 10px;
    }

    /* Header Styling */
    h1 {
        background: -webkit-linear-gradient(45deg, #4CAF50, #2E7D32);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-weight: 800 !important;
        text-align: center; /* Center the title */
        padding-bottom: 20px;
    }

    /* Highlight bold text in assistant messages */
    strong {
        color: #2E7D32;
        font-weight: 700;
    }

    /* Hide default hamburger menu and footer for cleaner look */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}

    </style>
""", unsafe_allow_html=True)

# --- Backend Import Handling ---
try:
    from backend.rag_backend import recommend_apartments, recommend_roommates

    backend_available = True
except Exception as e:
    backend_available = False
    st.error(f"⚠️ Backend unavailable. Check imports/keys. Error: {e}")

# --- Session State Initialization ---
if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant",
         "content": "Hi there! 👋 I'm your AI Roommate & Apartment Finder. To find your perfect match, I need to get to know you a little better. \n\nFirst things first, what is your **name**?"}
    ]

if "current_step" not in st.session_state:
    st.session_state.current_step = 0

if "user_data" not in st.session_state:
    st.session_state.user_data = {}

if "finished" not in st.session_state:
    st.session_state.finished = False

# --- Conversation Flow Definition ---
# Each step maps to a key in user_data and the NEXT question to ask
conversation_steps = [
    {"key": "name",
     "next_q": "Nice to meet you! 🏢 Now, tell me about your **ideal apartment**. \n\nInclude details like location, budget per person, pet-friendliness, amenities and how many roommates you want (e.g., 'Barrio Salamanca, under €900, needs a balcony, max 2 roommates')."},
    {"key": "apartment_desc",
     "next_q": "Got it. Now let's talk personality to find compatible roommates. \n\n1. Do you like trying **new experiences** and foods, or do you prefer sticking to routines you enjoy?"},
    {"key": "q1", "next_q": "2. How do you feel about **spontaneous plans** or surprises?"},
    {"key": "q2", "next_q": "3. Are you super **organized** and plan ahead, or do you prefer to go with the flow?"},
    {"key": "q3", "next_q": "4. How **punctual** are you usually for classes or appointments?"},
    {"key": "q4",
     "next_q": "5. After a long day, do you prefer **chatting** with roommates or having **quiet time** alone?"},
    {"key": "q5", "next_q": "6. How do you feel about hosting **gatherings/parties** at the apartment?"},
    {"key": "q6", "next_q": "7. When **conflicts** arise, do you stand your ground or look for a compromise?"},
    {"key": "q7", "next_q": "8. Would you describe yourself as **patient** and easygoing?"},
    {"key": "q8", "next_q": "9. How do you handle **stress** (like exams or noisy neighbors)?"},
    {"key": "q9", "next_q": "10. Finally, how much **alone time** do you need to recharge?"},
    {"key": "q10", "next_q": "Thanks! I have everything I need. Type 'Ready' to generate your matches! 🚀"}
]

# --- UI Layout ---

st.title("🏠 Roommate Finder Chat")

# 1. Display Chat History
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# 2. Handle User Input
if not st.session_state.finished:
    if prompt := st.chat_input("Type your answer here..."):
        # Display user message immediately
        with st.chat_message("user"):
            st.markdown(prompt)
        st.session_state.messages.append({"role": "user", "content": prompt})

        # Process current step
        current_idx = st.session_state.current_step

        # If we are within the question range
        if current_idx < len(conversation_steps):
            step_info = conversation_steps[current_idx]

            # Store the answer
            st.session_state.user_data[step_info["key"]] = prompt

            # Prepare next assistant message
            next_question = step_info["next_q"]

            # Simulate "typing" delay for realism
            time.sleep(0.4)

            st.session_state.messages.append({"role": "assistant", "content": next_question})
            st.session_state.current_step += 1
            st.rerun()

        # If we are at the end (Trigger generation)
        else:
            st.session_state.finished = True
            st.rerun()

# 3. Logic for Results Generation
if st.session_state.finished:
    if not backend_available:
        st.error("Backend functionality is missing. Cannot generate matches.")
    else:
        # Only run this once
        if "results_apt" not in st.session_state:
            with st.chat_message("assistant"):
                with st.spinner("🧠 Analyzing your personality profile and apartment database..."):

                    # Construct Queries (Matching original logic)
                    data = st.session_state.user_data

                    apartment_query = f"""
                    My name is {data.get('name')}.
                    Apartment preferences: {data.get('apartment_desc')} 
                    """

                    rm_query = f"""
                    Personality profile:
                    1. New experiences: {data.get('q1')}
                    2. Spontaneity: {data.get('q2')}
                    3. Organization: {data.get('q3')}
                    4. Punctuality: {data.get('q4')}
                    5. Social energy: {data.get('q5')}
                    6. Hosting: {data.get('q6')}
                    7. Conflict: {data.get('q7')}
                    8. Patience: {data.get('q8')}
                    9. Stress: {data.get('q9')}
                    10. Alone time: {data.get('q10')}
                    """

                    try:
                        # Fetch recommendations
                        res_apt = recommend_apartments(apartment_query, top_k=3)
                        res_rm = recommend_roommates(rm_query, top_k=3)

                        st.session_state.results_apt = res_apt
                        st.session_state.results_rm = res_rm

                    except Exception as e:
                        st.error(f"Error during recommendation: {e}")

        # Display Results if they exist
        if "results_apt" in st.session_state:
            st.markdown("---")
            st.success("✅ Matches Found!")

            tab1, tab2 = st.tabs(["🏢 Apartments", "👥 Roommates"])

            with tab1:
                st.markdown(st.session_state.results_apt)

            with tab2:
                st.markdown(st.session_state.results_rm)

            # Restart Button
            if st.button("🔄 Start New Chat"):
                for key in list(st.session_state.keys()):
                    del st.session_state[key]
                st.rerun()