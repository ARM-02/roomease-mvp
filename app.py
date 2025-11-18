import streamlit as st
import sys
import os

# Add current directory to path
sys.path.insert(0, os.path.dirname(__file__))

# Page config
st.set_page_config(
    page_title="Roommate & Apartment Finder",
    page_icon="🏠",
    layout="wide"
)

# Custom CSS for better styling
st.markdown("""
    <style>
    html, body, [class*="st"] {
        color: white !important;
        background-color: #0e1117;
    }

    .section-header {
        font-size: 2rem;
        font-weight: 700;
        margin-top: 2rem;
        margin-bottom: 1rem;
        color: white;
    }

    label, p, .stMarkdown, .stTextInput label {
        font-size: 1.2rem !important;
        color: white !important;
        font-weight: 500 !important;
        line-height: 1.6;
    }

    .stTextInput > div > div > input,
    .stTextArea > div > div > textarea {
        border-radius: 8px;
        border: 2px solid #e0e0e0;
        padding: 0.75rem;
        font-size: 1.1rem;
        color: white;
        background-color: #262730;
    }

    .stButton>button {
        width: 100%;
        background-color: #4CAF50;
        color: white;
        padding: 1rem;
        font-size: 1.2rem;
        border-radius: 10px;
        margin-top: 2rem;
        font-weight: 600;
    }

    .stButton>button:hover {
        background-color: #45a049;
    }

    .element-container {
        margin-bottom: 1.5rem;
    }
    </style>
""", unsafe_allow_html=True)

# Try to import the recommend function
try:
    from rag_backend1 import recommend

    import_success = True
except Exception as e:
    import_success = False
    st.error(f"""
    ⚠️ **Cannot import the recommendation function!**

    Error: {str(e)}

    **Please make sure:**
    1. `rag_backend1.py` is in the same folder as `app.py`
    2. Your ChromaDB folder `chroma_store` exists
    3. All required packages are installed
    4. Your Gemini API key is set

    **Try running this in terminal:**
    ```
    export GEMINI_API_KEY="your_api_key_here"
    pip install chromadb sentence-transformers google-generativeai
    ```
    """)
    st.stop()

# Initialize session state
if 'submitted' not in st.session_state:
    st.session_state.submitted = False
if 'recommendation' not in st.session_state:
    st.session_state.recommendation = None

# Title
st.title("🏠 Roommate & Apartment Finder")
st.markdown("### Find your perfect living situation based on your personality and preferences")
st.markdown("---")

# Name Section
st.markdown('<div class="section-header">👤 Tell us about yourself</div>', unsafe_allow_html=True)
name = st.text_input("What's your name?", placeholder="Enter your name", key="name_input")

st.markdown("<br>", unsafe_allow_html=True)

# Apartment Section
st.markdown('<div class="section-header">🏢 Your Ideal Apartment</div>', unsafe_allow_html=True)
st.markdown(
    "Tell us what you're looking for - vibe, location, price range, dealbreakers, or anything that matters to you.")
apartment_desc = st.text_area(
    "My ideal apartment...",
    placeholder="Example: I want a quiet, dog-friendly apartment in Barrio Salamanca with a budget of €3,300 max. I need my own private bathroom and would love sunset views.",
    height=120,
    key="apartment_input",
    label_visibility="collapsed"
)

st.markdown("<br><br>", unsafe_allow_html=True)

# Personality Questions Section
st.markdown('<div class="section-header">💭 Personality Questions</div>', unsafe_allow_html=True)
st.markdown("Answer these questions so we can match you with compatible roommates. Be honest and specific!")

st.markdown("<br>", unsafe_allow_html=True)

# Question 1
q1 = st.text_area(
    "1️⃣ Do you like trying new experiences, foods, or activities — or do you prefer routines you already enjoy?",
    placeholder="Your answer...",
    height=80,
    key="q1"
)

# Question 2
q2 = st.text_area(
    "2️⃣ How do you feel about spontaneous plans or surprises?",
    placeholder="Your answer...",
    height=80,
    key="q2"
)

# Question 3
q3 = st.text_area(
    "3️⃣Do you like to keep your space organized and plan your day ahead, or do you go with the flow?",
    placeholder="Your answer...",
    height=80,
    key="q3"
)

# Question 4
q4 = st.text_area(
    "4️⃣How punctual are you for classes or appointments?",
    placeholder="Your answer...",
    height=80,
    key="q4"
)

# Question 5
q5 = st.text_area(
    "5️⃣When you come home after a long day, do you enjoy chatting with roommates or prefer quiet time alone?",
    placeholder="Your answer...",
    height=80,
    key="q5"
)

# Question 6
q6 = st.text_area(
    "6️⃣Would you enjoy hosting small gatherings or prefer a calm apartment?",
    placeholder="Your answer...",
    height=80,
    key="q6"
)

# Question 7
q7 = st.text_area(
    "7️⃣When conflicts arise, do you usually try to find compromise or prefer to stand your ground?",
    placeholder="Your answer...",
    height=80,
    key="q7"
)

# Question 8
q8 = st.text_area(
    "8️⃣Would you describe yourself as easygoing and patient?",
    placeholder="Your answer...",
    height=80,
    key="q8"
)

# Question 9
q9 = st.text_area(
    "9️⃣When things go wrong (like a noisy neighbor or exam stress), do you get anxious easily or stay calm?",
    placeholder="Your answer...",
    height=80,
    key="q9"
)

# Question 10
q10 = st.text_area(
    "1️⃣0️⃣How often do you need alone time to recharge?",
    placeholder="Your answer...",
    height=80,
    key="q10"
)

# Submit button
st.markdown("---")
submit_button = st.button("🔍 Find My Perfect Match!")

if submit_button:
    # Validate inputs
    if not name or not name.strip():
        st.error("⚠️ Please enter your name!")
    elif not apartment_desc or not apartment_desc.strip():
        st.error("⚠️ Please describe what you're looking for in an apartment!")
    elif not all([q1, q2, q3, q4, q5, q6, q7, q8, q9, q10]):
        st.error("⚠️ Please answer all 10 personality questions!")
    elif any(len(q.strip()) < 10 for q in [q1, q2, q3, q4, q5, q6, q7, q8, q9, q10]):
        st.error("⚠️ Please provide more detailed answers to the questions (at least a sentence each)!")
    else:
        # Construct the query from all inputs
        query = f"""
My name is {name}.

Apartment preferences: {apartment_desc}

Personality profile:
1. New experiences and routines: {q1}
2. Spontaneity and surprises: {q2}
3. Organization and planning: {q3}
4. Punctuality: {q4}
5. Social energy after work: {q5}
6. Hosting and gatherings: {q6}
7. Conflict resolution style: {q7}
8. Patience and temperament: {q8}
9. Stress response: {q9}
10. Alone time needs: {q10}
"""

        with st.spinner("🔄 Analyzing your profile and finding the best matches... This may take a moment."):
            try:
                # Call your recommendation function
                recommendation = recommend(query)
                st.session_state.recommendation = recommendation
                st.session_state.submitted = True
                st.rerun()
            except Exception as e:
                st.error(f"❌ An error occurred while getting recommendations: {str(e)}")
                st.error("Please check that your ChromaDB database is properly set up and your API key is configured.")

# Display results
if st.session_state.submitted and st.session_state.recommendation:
    st.markdown("---")
    st.markdown('<div class="section-header">✨ Your Personalized Recommendations</div>', unsafe_allow_html=True)

    # Display the recommendation in a nice box
    st.markdown("""
        <div style='background-color: #f8f9fa; padding: 2rem; border-radius: 12px; margin-top: 1.5rem; border: 2px solid #e0e0e0;'>
    """, unsafe_allow_html=True)

    st.markdown(st.session_state.recommendation)

    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # Reset button
    col1, col2, col3 = st.columns([1, 1, 1])
    with col2:
        if st.button("🔄 Start Over"):
            st.session_state.submitted = False
            st.session_state.recommendation = None
            st.rerun()
