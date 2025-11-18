from __future__ import annotations
from typing import List, Dict, Any, Optional
import os
import json
import re


import chromadb
from sentence_transformers import SentenceTransformer
import google.generativeai as genai
from apartment_description_summarizer import summarize_description





#Helper json function

def clean_json(text: str) -> str:
    # Remove newlines
    t = text.strip().replace("\n", " ")

    # Remove trailing commas before } or ]
    t = re.sub(r",\s*}", "}", t)
    t = re.sub(r",\s*]", "]", t)
    t = t.replace("```json", "").replace("```", "")

    # Remove leading garbage before a JSON object
    first_brace = t.find("{")
    if first_brace > 0:
        t = t[first_brace:]

    return t


# ======================================================
# 0. GEMINI CONFIG
# ======================================================

GEMINI_MODEL = "gemini-2.5-flash-preview-09-2025"

api_key = os.environ.get("GEMINI_API_KEY")
if not api_key:
    print("=" * 60)
    print("ERROR: GEMINI_API_KEY environment variable not set.")
    print("Export it first, e.g.:")
    print('  export GEMINI_API_KEY="your_api_key_here"')
    print("=" * 60)
    raise SystemExit(1)

genai.configure(api_key=api_key)


def run_gemini(prompt: str, json_schema: Optional[Dict] = None) -> str:
    """
    Safe wrapper around Gemini generate_content.
    If json_schema is provided → ask for JSON and return response.text.
    Otherwise → return plain text.
    """
    try:
        gen_config: Dict[str, Any] = {
            "temperature": 0.2,
            "max_output_tokens": 2048,
        }

        if json_schema is not None:
            gen_config["response_mime_type"] = "application/json"
            gen_config["response_schema"] = json_schema

        model = genai.GenerativeModel(GEMINI_MODEL)
        response = model.generate_content(
            prompt,
            generation_config=gen_config,
        )

        if not response.candidates:
            return "ERROR: Gemini returned no candidates."

        if json_schema is not None:
            if getattr(response, "text", None):
                return response.text
            # Fallback: concat text parts
            cand = response.candidates[0]
            txt = ""
            for p in cand.content.parts:
                if hasattr(p, "text"):
                    txt += p.text
            return txt or "ERROR: Empty JSON response."

        # Free-text mode
        cand = response.candidates[0]
        out = ""
        for p in cand.content.parts:
            if hasattr(p, "text"):
                out += p.text
        return out.strip() or "(empty response)"

    except Exception as e:
        return f"ERROR: Gemini API failed: {e}"


# ======================================================
# 1. EMBEDDINGS & CHROMA
# ======================================================

print("[INFO] Loading embedding model...")
EMBED_MODEL = "all-MiniLM-L6-v2"
embedder = SentenceTransformer(EMBED_MODEL)


def embed_query(text: str) -> List[List[float]]:
    return embedder.encode([text], normalize_embeddings=True).tolist()


print("[INFO] Connecting to ChromaDB...")
DB_PATH = "./chroma_store"
client = chromadb.PersistentClient(path=DB_PATH)

APTS_COLLECTION = "apartments"
STUDENTS_COLLECTION = "students"

apts_collection = client.get_collection(APTS_COLLECTION)
students_collection = client.get_collection(STUDENTS_COLLECTION)
print("[INFO] Collections loaded.")


# ======================================================
# 2. STUDENT FILTERING & CONTEXT
# ======================================================

def filter_students(docs, metas):
    filtered = []

    structured_patterns = [
        r"\bhighly conscientious\b",
        r"\bhigh conscientiousness\b",
        r"\bvery conscientious\b",
        r"\bhigh in conscientiousness\b",
        r"\bstrict routine\b",
        r"\bvery strict\b",
        r"\bhighly structured\b",
        r"\bvery structured\b",
        r"\bstructured routine\b",
        r"\bregimented\b",
        r"\bstick(s)? to routine\b",
    ]

    for doc, meta in zip(docs, metas):
        dog = (meta.get("dog_friendliness") or "").lower()
        lifestyle = (meta.get("lifestyle_summary") or "").lower()
        sleep = (meta.get("sleep_schedule") or "").lower()
        noise = (meta.get("noise_tolerance") or "").lower()

        # 🚫 DOG FILTER
        if "allergic" in dog or "pet allergic" in dog or "no dogs" in dog:
            continue

        # 🚫 STRUCTURE FILTER (strong only, via regex)
        if any(re.search(p, lifestyle) for p in structured_patterns):
            continue

        # 🚫 NOISE-SENSITIVE EARLY SLEEPER
        # Example: "10:30pm - wake up early", "noise sensitive"
        if "noise sensitive" in lifestyle or "noise sensitive" in noise:
            # Sleep times earlier than 11:00 → reject
            if "10" in sleep or "10:" in sleep:
                continue

        raw_name = meta.get("name")
        if raw_name in [None, "", "not specified"]:
            name = f"Candidate_{meta.get('chunk_index')}"
        else:
            name = raw_name

        # OTHERWISE → Accept
        filtered.append({
            "name": name,
            "personality": meta.get("personality"),
            "lifestyle": lifestyle,
            "sleep": sleep,
            "noise": noise,
            "dog": dog,
            "cleanliness": meta.get("cleanliness"),
            "study": meta.get("study_habits"),
            "raw": doc,
            "meta": meta,
        })

    return filtered



def build_student_context(students: List[Dict[str, Any]]) -> str:
    if not students:
        return "No candidates after filtering."

    lines = []
    for s in students:
        lines.append(
            f"- Name: {s['name']}\n"
            f"  Personality: {s['personality']}\n"
            f"  Lifestyle: {s['lifestyle']}\n"
            f"  Sleep: {s['sleep']}\n"
            f"  Noise tolerance: {s['noise']}\n"
            f"  Dog friendliness: {s['dog']}\n"
            f"  Cleanliness: {s['cleanliness']}\n"
            f"  Study habits: {s['study']}\n"
        )
    return "\n\n".join(lines)


# ======================================================
# 3. APARTMENT CONTEXT BUILDER
# ======================================================

def build_apartment_context(docs: List[str], metas: List[Dict[str, Any]]) -> str:
    """
    We do NOT filter here. We just format all retrieved apartments so Gemini
    can score them against structured + unstructured preferences.
    """
    if not docs:
        return "No apartments retrieved."

    lines = []
    for doc, meta in zip(docs, metas):
        prop_id = meta.get("propertyCode")
        neighborhood = meta.get("neighborhood") or ""
        district = meta.get("district") or ""
        price = meta.get("price")
        rooms = meta.get("rooms")
        bathrooms = meta.get("bathrooms")
        size = meta.get("size")
        exterior = meta.get("exterior")
        has_lift = meta.get("hasLift")
        url = meta.get("url")
        property_type = meta.get("propertyType") or meta.get("type")

        title = f"Property {prop_id}"
        stxt = meta.get("suggestedTexts")

        if stxt:
            try:
                import ast
                stxt_dict = ast.literal_eval(stxt)
                if isinstance(stxt_dict, dict):
                    t = stxt_dict.get("title")
                    if t:
                        title = t
            except:
                m = re.search(r"'title':\s*'([^']+)'", stxt)
                if m:
                    title = m.group(1)

        header = (
            f"PROPERTY_CODE: {prop_id} | {title} | "
            f"District: {district} | Neighborhood: {neighborhood} | "
            f"Rooms: {rooms} | Bathrooms: {bathrooms} | Size: {size} m2 | "
            f"Exterior: {exterior} | Lift: {has_lift} | "
            f"Price: €{price} | Type: {property_type} | URL: {url}"
        )
        lines.append(f"- {header}\n  Description: {doc}")
    return "\n\n".join(lines)


# ======================================================
# 4. JSON SCHEMAS
# ======================================================

# Apartment query parsing (structured + unstructured)
apartment_parse_schema: Dict[str, Any] = {
    "type": "object",
    "properties": {
        "apartment_query": {"type": "string"},
        "structured_filters": {
            "type": "object",
            "properties": {
                "district": {"type": "string"},
                "neighborhood": {"type": "string"},
                "min_rooms": {"type": "integer"},
                "min_bathrooms": {"type": "integer"},
                "min_size": {"type": "number"},
                "must_be_exterior": {"type": "boolean"},
                "must_have_lift": {"type": "boolean"},
            },
            "required": [],
        },
        "unstructured_preferences": {
            "type": "array",
            "items": {"type": "string"},
        },
        "roommates": {"type": "integer"},
        "budget": {"type": "number"},
    },
    "required": ["apartment_query"],
}

# Apartment scoring
apartment_score_schema: Dict[str, Any] = {
    "type": "object",
    "properties": {
        "apartments": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "property_code": {"type": "string"},
                    "total_score": {"type": "number"},
                    "reasoning": {"type": "string"},
                },
                "required": ["property_code", "total_score", "reasoning"],
            },
        }
    },
    "required": ["apartments"],
}

# Roommate scoring
student_score_schema: Dict[str, Any] = {
    "type": "object",
    "properties": {
        "students": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "name": {"type": "string"},
                    "score": {"type": "number"},
                    "reasoning": {"type": "string"},
                },
                "required": ["name", "score", "reasoning"],
            },
        }
    },
    "required": ["students"],
}


# ======================================================
# 5. APARTMENT RECOMMENDATION FLOW
# ======================================================

def recommend_apartments(user_query: str, top_k: int = 3) -> str:
    """
    Flow:
      1) Parse user_query into structured_filters + unstructured_preferences + roommates + budget.
      2) Retrieve apartments from Chroma (semantic).
      3) Let Gemini score apartments based on:
         - structured_filters (soft constraints, ignore if empty)
         - unstructured_preferences (views, quiet, etc., ignore if empty)
         - effective budget (budget * (roommates + 1) if available)
      4) Return top_k apartments with best scores.
    """
    print("[APT] Step 0: Parsing apartment query with Gemini...")

    parse_prompt = f"""
    You are given a user request about apartments.

    Extract:
    1. apartment_query: short cleaned version of their apartment request.

    2. structured_filters:
       - Only include fields that appear explicitly in the user request.
       - Valid fields: district, neighborhood, min_rooms, min_bathrooms, min_size, must_be_exterior, must_have_lift.
       - If the user names multiple neighborhoods or districts (e.g., Salamanca, Retiro, Chamberí), include ALL inside "neighborhood" as a single comma-separated string.

    3. unstructured_preferences:
       - Soft wishes like “quiet”, “sunset views”, “nice views”, “quiet street”.
       - Do NOT place neighborhoods here.

    4. roommates: number of roommates (not counting the user).
    5. budget: monthly budget per person.

    If the user does NOT specify something, leave it null.
    Do NOT invent constraints.

    User message:
    \"\"\"{user_query}\"\"\"
    """


    parsed_raw = run_gemini(parse_prompt, json_schema=apartment_parse_schema)
    if parsed_raw.startswith("ERROR"):
        return parsed_raw

    try:
        parsed_clean = clean_json(parsed_raw)
        parsed = json.loads(parsed_clean)
    except Exception as e:
        return f"ERROR: Failed to parse apartment parse JSON: {e}\nRaw: {parsed_raw}"

    apt_query = parsed.get("apartment_query") or user_query
    structured = parsed.get("structured_filters") or {}
    prefs = parsed.get("unstructured_preferences") or []
    roommates = parsed.get("roommates")
    budget_per_person = parsed.get("budget")

    # effective total budget
    effective_budget = None
    total_people = (roommates + 1) if isinstance(roommates, int) and roommates > 0 else None

    if isinstance(budget_per_person, (int, float)) and total_people:
        effective_budget = budget_per_person * total_people
    # else: keep None (means "no budget constraint")
    print(f"[APT] Parsed structured: {structured}")
    print(f"[APT] Parsed prefs: {prefs}")
    print(f"[APT] Roommates: {roommates}, budget_per_person: {budget_per_person}, effective_budget: {effective_budget}")

    # 1) Retrieve from Chroma (semantic)
    print("[APT] Step 1: Retrieving apartments from Chroma...")
    try:
        apt_res = apts_collection.query(
            query_embeddings=embed_query(apt_query),
            n_results=5,  # more candidates, LLM will rank
        )
    except Exception as e:
        return f"ERROR: Chroma query failed: {e}"

    if not apt_res["documents"]:
        return "No apartments found in database."

    apt_docs = apt_res["documents"][0]
    apt_metas = apt_res["metadatas"][0]

    # NEW: summarize the descriptions using your local LM Studio model
    summarized_docs = []
    for d in apt_docs:
        try:
            summarized_docs.append(summarize_description(d))
        except Exception as e:
            print(f"[WARN] Summarization failed for one doc: {e}")
            summarized_docs.append(d[:200])  # fallback to short truncation

    # Build context with summarized texts
    apt_context = build_apartment_context(summarized_docs, apt_metas)

    # --- Build dynamic clauses for scoring prompt ---
    neigh_raw = structured.get("neighborhood")
    if neigh_raw:
        requested_neighs = [n.strip().lower() for n in neigh_raw.split(",")]
        neighborhood_clause = f"- Preferred neighborhoods: {', '.join(requested_neighs)}"
    else:
        requested_neighs = []
        neighborhood_clause = ""

    exterior_clause = (
        "- Must be exterior"
        if structured.get("must_be_exterior") else ""
    )

    lift_clause = (
        "- Must have lift"
        if structured.get("must_have_lift") else ""
    )

    soft_prefs_list = (
        "- Soft preferences: " + ", ".join(prefs)
        if prefs else ""
    )

    print("[APT] Step 2: Scoring apartments with Gemini...")

    # Important: this scoring prompt is GENERIC, not hard-coded to Salamanca, etc.
    score_prompt = f"""
    You are evaluating a list of apartments for the user.

USER REQUIREMENTS:
Structured:
- Max total budget: €{effective_budget if effective_budget else "None"}
{neighborhood_clause}
{exterior_clause}
{lift_clause}


Unstructured preferences (soft filters — never treat as hard constraints):
{soft_prefs_list}

INSTRUCTIONS:
You MUST:
- Only use the apartments listed below.
- Score each apartment strictly from its provided metadata and description.
- NEVER assume missing features.
- NEVER add features that are not explicitly written.
- Penalize missing information instead of inventing.
- Output only valid JSON following the required schema.

SSCORING RULES (0–10):

NEIGHBORHOOD MATCHING:
- The user may specify one or multiple neighborhoods/districts.
- +2 if the apartment's neighborhood matches ANY of them (case-insensitive).
- +1 if the district matches but neighborhood does not explicitly match.
- 0 if unrelated.

OTHER RULES:
- +2 fits budget
- +2 exterior / good light / views
- +2 lift if requested
- +2 if the description contains any soft preferences ("quiet", "nice views", etc.)


OUTPUT FORMAT (strict JSON):
Return an object with a key "apartments" containing an array of objects.
Each object must have:
- "property_code" (string, exactly as shown in the context)
- "total_score" (number)
- "reasoning" (short explanation, 1–2 sentences)


APARTMENTS TO EVALUATE:
{apt_context}
"""

    score_raw = run_gemini(score_prompt, json_schema=apartment_score_schema)
    if score_raw.startswith("ERROR"):
        return score_raw

    try:
        score_raw = clean_json(score_raw)
        scores = json.loads(score_raw)
    except Exception as e:
        return f"ERROR: Failed to parse apartment scores JSON: {e}\nRaw: {score_raw}"

    scored_list = scores.get("apartments") or []
    if not scored_list:
        return "No apartment scores returned by LLM."

    # 3) Rank and take top_k
    scored_list.sort(key=lambda x: x.get("total_score", 0), reverse=True)
    top = scored_list[:top_k]

    # Build lookup for URLs (always use strings for keys)
    meta_by_code = {
        str(m.get("propertyCode")): m
        for m in apt_metas
        if m.get("propertyCode") is not None
    }

    lines = []
    for item in top:
        code = str(item["property_code"])
        meta = meta_by_code.get(code, {})
        url = meta.get("url", "None")
        price = meta.get("price", "N/A")
        district = meta.get("district", "")
        neighborhood = meta.get("neighborhood", "")
        lines.append(
            f"- PROPERTY_CODE: {code} | District: {district} | Neighborhood: {neighborhood} | "
            f"Price: €{price} | URL: {url}\n"
            f"  Score: {item['total_score']:.2f} | Reason: {item['reasoning']}"
        )

    return "Top apartment matches:\n" + "\n\n".join(lines)


# ======================================================
# 6. ROOMMATE RECOMMENDATION FLOW
# ======================================================

def recommend_roommates(user_profile_query: str, top_k: int = 3) -> str:
    """
    Flow for roommate recommendation:
      1) Retrieve candidates using user_profile_query.
      2) Filter out clearly incompatible students (dog-allergic, highly structured).
      3) Let Gemini score compatibility and return top_k.
    """
    print("[RM] Retrieving roommate candidates from Chroma...")

    try:
        stu_res = students_collection.query(
            query_embeddings=embed_query(user_profile_query),
            n_results=10,
        )
    except Exception as e:
        return f"ERROR: Chroma query failed: {e}"

    if not stu_res["documents"]:
        return "No roommate candidates found in database."

    stu_docs = stu_res["documents"][0]
    stu_metas = stu_res["metadatas"][0]

    students = filter_students(stu_docs, stu_metas)
    if not students:
        return "No compatible roommate candidates after basic filtering."

    stu_context = build_student_context(students)

    print("[RM] Scoring roommates with Gemini...")

    rm_prompt = f"""
You are matching roommates.

User profile:
\"\"\"{user_profile_query}\"\"\"

Below are candidate students with their traits and lifestyles.

Score each candidate from 0 to 10 based on:
- social and personality fit,
- tolerance for spontaneity / mess if relevant,
- dog-friendliness if relevant,
- sleep and noise compatibility if described in the profile.

If the user profile does not specify a constraint (e.g., doesn't mention dogs),
ignore that dimension (do not penalize or reward).

OUTPUT FORMAT (PLAIN TEXT, NO JSON):

Write ONE line per candidate exactly as follows:
NAME | SCORE | REASON

Where:
- NAME is the candidate's name exactly as shown
- SCORE is a number from 0 to 10
- REASON is a short justification (1 sentence)

Example:
Alex | 9 | Very compatible lifestyle, social schedule aligns.
Maria | 5.5 | Some compatibility but more structured than user.

Do NOT add headers.
Do NOT wrap in JSON.
Do NOT add extra text.





CANDIDATES:
{stu_context}
"""

    def parse_student_scores(text: str) -> List[Dict[str, Any]]:
        results = []
        for line in text.splitlines():
            line = line.strip()
            if not line or "|" not in line:
                continue
            parts = [p.strip() for p in line.split("|", 2)]
            if len(parts) < 3:
                continue
            name, score_str, reason = parts
            try:
                score = float(score_str)
            except ValueError:
                continue
            results.append({
                "name": name,
                "score": score,
                "reasoning": reason
            })
        return results

    rm_raw = run_gemini(rm_prompt)
    if rm_raw.startswith("ERROR"):
        return rm_raw

    scored_students = parse_student_scores(rm_raw)
    if not scored_students:
        return f"No roommate scores returned.\nRaw output:\n{rm_raw}"


    scored_students.sort(key=lambda x: x.get("score", 0), reverse=True)
    top = scored_students[:top_k]

    lines = []
    for s in top:
        lines.append(
            f"- Name: {s['name']} | Score: {s['score']:.2f}\n"
            f"  Reason: {s['reasoning']}"
        )

    return "Top roommate matches:\n" + "\n\n".join(lines)


# ======================================================
# 7. MANUAL TESTS
# ======================================================

if __name__ == "__main__":
    apt_q = (
        "I want a flat to rent in Madrid, ideally in Salamanca or Retiro, "
        "around 1500€ per person with 2 roommates (so 3 people total). "
        "I care about quiet streets and nice views, and I prefer exterior with lift."
    )
    print("--- Testing apartment recommendation ---")
    print(recommend_apartments(apt_q, top_k=3))

    rm_q = (
        "I am very extroverted, spontaneous, not very organized, and I can't stand "
        "people who are extremely structured or uptight. I like some socializing at home, "
        "and I am okay with dogs."
    )
    print("\n--- Testing roommate recommendation ---")
    print(recommend_roommates(rm_q, top_k=3))
