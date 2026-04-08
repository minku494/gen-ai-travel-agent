import os
import json
import streamlit as st
from dotenv import load_dotenv
from typing import List

from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser, JsonOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_core.messages import HumanMessage, AIMessage
from langchain_google_genai import ChatGoogleGenerativeAI

from langchain_text_splitters import RecursiveCharacterTextSplitter

from pydantic import BaseModel, Field

load_dotenv()

class Hotel(BaseModel):
    name: str = Field(description="Hotel name")
    stars: str = Field(description="Star rating e.g. ⭐⭐")
    price_per_night: str = Field(description="Price per night in given currency")
    location: str = Field(description="Area/neighbourhood of the hotel")
    highlight: str = Field(description="One standout feature of the hotel")

class Restaurant(BaseModel):
    name: str = Field(description="Restaurant name")
    cuisine: str = Field(description="Type of cuisine")
    avg_cost_per_person: str = Field(description="Average cost per person")
    must_try_dish: str = Field(description="Signature dish to try")
    vibe: str = Field(description="Ambience description e.g. rooftop, cozy, street-side")

class Attraction(BaseModel):
    name: str = Field(description="Place or attraction name")
    category: str = Field(description="Category e.g. Museum, Temple, Park, Market")
    entry_fee: str = Field(description="Entry fee or 'Free'")
    best_time_to_visit: str = Field(description="Best time of day or season to visit")
    insider_tip: str = Field(description="Useful insider tip for visitors")

class DayPlan(BaseModel):
    day_number: int = Field(description="Day number starting from 1")
    theme: str = Field(description="Theme for the day e.g. 'Heritage Walk & Culinary Delights'")
    morning: str = Field(description="Morning activities")
    afternoon: str = Field(description="Afternoon activities")
    evening: str = Field(description="Evening activities and dinner")

class BudgetBreakdown(BaseModel):
    accommodation: str = Field(description="Total accommodation cost")
    food: str = Field(description="Total food & dining cost")
    transport: str = Field(description="Total transport cost")
    activities: str = Field(description="Total activities & entry fees")
    miscellaneous: str = Field(description="Miscellaneous / shopping buffer")
    total: str = Field(description="Grand total estimate")

class TravelPlan(BaseModel):
    destination: str = Field(description="Full destination name with country")
    trip_tagline: str = Field(description="A catchy one-liner tagline for this trip")
    duration: str = Field(description="e.g. 7 Days / 6 Nights")
    best_season: str = Field(description="Best time/season to visit")
    summary: str = Field(description="2-3 sentence overview of the trip experience")
    budget_breakdown: BudgetBreakdown
    hotels: List[Hotel] = Field(description="3 hotel recommendations", min_length=3, max_length=3)
    restaurants: List[Restaurant] = Field(description="4 restaurant recommendations", min_length=4, max_length=4)
    places_to_visit: List[Attraction] = Field(description="5 must-visit attractions", min_length=5, max_length=5)
    day_plan: List[DayPlan] = Field(description="Day-by-day itinerary")
    travel_tips: List[str] = Field(description="5 practical travel tips", min_length=5, max_length=5)
    packing_essentials: List[str] = Field(description="5-7 packing essentials", min_length=5, max_length=7)
    emergency_contacts: List[str] = Field(description="3 useful emergency/helpline numbers or info", min_length=3, max_length=3)

def build_travel_chain():
    llm = ChatGoogleGenerativeAI(
        model="gemini-2.5-flash",
        google_api_key=os.getenv("GOOGLE_API_KEY"),
        temperature=0.7,
    )

    parser = JsonOutputParser(pydantic_object=TravelPlan)
    format_instructions = parser.get_format_instructions()

    system_prompt = """You are an elite AI travel agent with encyclopedic knowledge of global destinations.
You craft hyper-personalised, realistic travel plans with REAL hotel names, restaurant names, and attraction names.
You always suggest options that fit within the given budget.
Always include authentic local experiences aligned with the travellers' interests.

{format_instructions}

CRITICAL: Your ENTIRE response must be ONLY the JSON object. No markdown fences. No preamble. No explanation. Start with {{ and end with }}."""

    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", """\
Plan a {days}-day trip to {destination} for {people} traveller(s).
Total budget: {budget} {currency} (for all people combined).
Primary interests: {interests}.
Travel style: {style}.
Additional preferences: {preferences}.

Create a comprehensive, realistic travel plan with:
- 3 hotel options across different budget tiers (budget / mid-range / luxury within budget)
- 4 local restaurant picks (mix of street food to upscale)
- 5 must-visit places tailored to their interests
- Full day-by-day itinerary for all {days} days
- Detailed budget breakdown in {currency}
- Practical tips, packing list, and emergency contacts for {destination}

Remember: Use REAL, well-known establishment names. Be specific about areas, timings, and costs."""),
    ])

    chain =(
        RunnablePassthrough.assign(
            format_instructions=lambda _: format_instructions
        ) | prompt | llm | StrOutputParser()
    )

    return chain, parser

def build_chat_chain():
    llm = ChatGoogleGenerativeAI(
        model="gemini-2.5-flash",
        google_api_key=os.getenv("GOOGLE_API_KEY"),
        temperature=0.7,
    )

    system_prompt = """You are an expert AI travel agent. The traveller has already received the travel plan shown below. Your job is to answer questions, suggest modifications, swap hotels/restaurants/attractions,
adjust the itinerary, or refine any part of the plan based on the traveller's requests.

--- CURRENT TRAVEL PLAN (JSON) ---
{plan_json}
--- END OF PLAN ---

Guidelines:
- Always refer back to the plan above when the user asks about it.
- When the user asks to change something, clearly describe what the updated section would look like.
- Keep suggestions realistic, within the original budget unless the user asks to change it.
- Be conversational, friendly, and concise."""

    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{user_message}"),
    ])

    chain = prompt | llm | StrOutputParser()
    return chain

def chunk_and_parse(raw_output: str, parser: JsonOutputParser) -> dict:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=100,
        separators=["\n\n", "\n", "}", "{", ","],
    )
    chunks = splitter.split_text(raw_output)

    full_text = raw_output.strip()
    if full_text.startswith("```"):
        full_text = full_text.split("```")[1]
        if full_text.startswith("json"):
            full_text = full_text[4:]
    full_text = full_text.strip().rstrip("```").strip()

    return json.loads(full_text), chunks

st.set_page_config(
    page_title="GenAI Travel Manager",
    page_icon="✈️",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
</style>
""", unsafe_allow_html=True)

st.markdown("""
<div class="main-header">
  <h1>GenAI Travel Manager</h1>
</div>
""", unsafe_allow_html=True)

with st.sidebar:
    st.markdown("### 🗺️ Plan Your Trip")
    st.markdown("---")
    destination = st.text_input("📍 Destination")

    col1, col2 = st.columns(2)
    with col1:
        days = st.number_input("📅 Days", min_value=1, max_value=30, value=5)
    with col2:
        people = st.number_input("👥 People", min_value=1, max_value=20, value=2)

    col3, col4 = st.columns(2)
    with col3:
        budget = st.number_input("💰 Budget", min_value=100, value=2000, step=100)
    with col4:
        currency = st.selectbox("💱 Currency", ["USD", "EUR", "GBP", "INR", "JPY", "AUD", "CAD"])

    selected_interests = ["General sightseeing"]

    style = st.selectbox(
        "🎒 Travel Style",
        ["Backpacker", "Budget Traveller", "Mid-range Explorer", "Luxury Traveller", "Family-friendly"],
    )

    preferences = st.text_area("📝 Extra Preferences", height=80)
    generate_btn = st.button("🚀 Generate My Travel Plan")


if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "plan_data" not in st.session_state:
    st.session_state.plan_data = None
if "chunks_info" not in st.session_state:
    st.session_state.chunks_info = None
if "chat_active" not in st.session_state:    
    st.session_state.chat_active = False
if "plan_chat_history" not in st.session_state:
    st.session_state.plan_chat_history = []
    
def enforce_lengths(plan_dict):
    def fix_list(lst, min_len, max_len, default_item):
        lst = lst or []

        if len(lst) > max_len:
            lst = lst[:max_len]

        while len(lst) < min_len:
            lst.append(default_item.copy() if isinstance(default_item, dict) else default_item)

        return lst

    plan_dict["hotels"] = fix_list(plan_dict.get("hotels"), 3, 3, {
        "name": "TBD Hotel",
        "stars": "N/A",
        "price_per_night": "N/A",
        "location": "N/A",
        "highlight": "Auto-filled"
    })

    plan_dict["restaurants"] = fix_list(plan_dict.get("restaurants"), 4, 4, {
        "name": "TBD Restaurant",
        "cuisine": "N/A",
        "avg_cost_per_person": "N/A",
        "must_try_dish": "N/A",
        "vibe": "N/A"
    })

    plan_dict["places_to_visit"] = fix_list(plan_dict.get("places_to_visit"), 5, 5, {
        "name": "TBD Attraction",
        "category": "General",
        "entry_fee": "N/A",
        "best_time_to_visit": "Anytime",
        "insider_tip": "Auto-filled"
    })

    plan_dict["travel_tips"] = fix_list(plan_dict.get("travel_tips"), 5, 5, "General travel tip")
    plan_dict["packing_essentials"] = fix_list(plan_dict.get("packing_essentials"), 5, 7, "Basic essential")
    plan_dict["emergency_contacts"] = fix_list(plan_dict.get("emergency_contacts"), 3, 3, "Local emergency number")

    return plan_dict
if generate_btn:
    st.session_state.chat_active = False
    st.session_state.plan_chat_history = []

    chain, parser = build_travel_chain()

    chain_input = {
        "destination": destination,
        "days": days,
        "people": people,
        "budget": budget,
        "currency": currency,
        "interests": ", ".join(selected_interests),
        "style": style,
        "preferences": preferences or "None",
        "chat_history": st.session_state.chat_history,
    }

    raw_output = chain.invoke(chain_input)
    plan_dict, chunks = chunk_and_parse(raw_output, parser)

    plan_dict = enforce_lengths(plan_dict)

    for d in plan_dict.get("day_plan", []):
        if "morning" not in d: d["morning"] = ""
        if "afternoon" not in d: d["afternoon"] = ""
        if "evening" not in d: d["evening"] = ""

    try:
        plan = TravelPlan(**plan_dict)
    except Exception as e:
        st.error(f"Validation failed: {e}")
        st.stop()

    st.session_state.plan_data = plan
    st.session_state.chunks_info = (len(chunks), len(raw_output))

plan = st.session_state.plan_data

if plan:
    st.markdown(f"## ✈️ {plan.trip_tagline}")
    st.markdown(f"**Destination:** {plan.destination}")
    st.markdown(f"**Duration:** {plan.duration}")
    st.markdown(f"**Best Season:** {plan.best_season}\n")

    st.markdown(f"### 🌍 Overview")
    st.markdown(plan.summary)

    bd = plan.budget_breakdown
    st.markdown(f"""
### 💰 Budget Breakdown  
Accommodation: {bd.accommodation}  
Food: {bd.food}  
Transport: {bd.transport}  
Activities: {bd.activities}  
Miscellaneous: {bd.miscellaneous}  
**Total: {bd.total}**
""")

    st.markdown("### 🏨 Hotels")
    for h in plan.hotels:
        st.markdown(f"- **{h.name}** ({h.stars}) in {h.location}, {h.price_per_night}/night — {h.highlight}")

    st.markdown("### 🍽️ Restaurants")
    for r in plan.restaurants:
        st.markdown(f"- **{r.name}** ({r.cuisine}) — Avg: {r.avg_cost_per_person}, Must try: {r.must_try_dish}, {r.vibe}")

    st.markdown("### 🗺️ Places to Visit")
    for p in plan.places_to_visit:
        st.markdown(f"- **{p.name}** ({p.category}) — {p.entry_fee}, Best time: {p.best_time_to_visit}. Tip: {p.insider_tip}")

    st.markdown("### 📅 Itinerary")
    for d in plan.day_plan:
        st.markdown(f"""
**Day {d.day_number}: {d.theme}**  
Morning: {d.morning}  
Afternoon: {d.afternoon}  
Evening: {d.evening}
""")

    st.markdown("### 💡 Travel Tips")
    for t in plan.travel_tips:
        st.markdown(f"- {t}")

    st.markdown("### 🎒 Packing Essentials")
    st.markdown(", ".join(plan.packing_essentials))

    st.markdown("---")

    if not st.session_state.chat_active:
        if st.button("💬 Chat with your Travel Agent", type="primary"):
            st.session_state.chat_active = True
            st.rerun()
    else:
        st.markdown("### 💬 Travel Agent Chat")
        st.caption("Ask me to tweak hotels, swap restaurants, adjust your itinerary, or anything else about your plan!")

        for msg in st.session_state.plan_chat_history:
            if isinstance(msg, HumanMessage):
                st.markdown(f"**You**")
                st.markdown(msg.content)
            else:
                st.markdown(f"**AI Agent**")
                st.markdown(msg.content)
            st.markdown("---")


        user_input = st.chat_input("e.g. Can you increase the budget by...?")

        if user_input:
            st.markdown(f"**You**")
            st.markdown(user_input)
            st.markdown("---")

            chat_chain = build_chat_chain()
            plan_json = plan.model_dump_json(indent=2)

            with st.spinner("Thinking..."):
                response = chat_chain.invoke({
                    "plan_json": plan_json,
                    "chat_history": st.session_state.plan_chat_history,
                    "user_message": user_input,
                })

            st.markdown(f"**AI Agent**")
            st.markdown(response)
            st.markdown("---")

            st.session_state.plan_chat_history.append(HumanMessage(content=user_input))
            st.session_state.plan_chat_history.append(AIMessage(content=response))
            st.rerun()