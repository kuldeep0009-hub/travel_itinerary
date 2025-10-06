import streamlit as st
from typing import TypedDict, List
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq

# ---- TypedDict for state ----
class PlannerState(TypedDict):
    messages: List[HumanMessage | AIMessage]
    city: str
    interests: List[str]
    itinerary: str

# ---- Functions ----
def input_city(city: str, state: PlannerState) -> PlannerState:
    return {
        **state,
        "city": city,
        "messages": state["messages"] + [HumanMessage(content=city)],
    }

def input_interests(interests: str, state: PlannerState) -> PlannerState:
    return {
        **state,
        "interests": [interest.strip() for interest in interests.split(',')],
        "messages": state["messages"] + [HumanMessage(content=interests)],
    }

def create_itinerary(state: PlannerState, llm: ChatGroq) -> str:
    itinerary_prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a helpful travel assistant. Create a day trip itinerary for {city} based on the user's interests: {interests}. Provide a brief, bulleted itinerary."),
        ("human", "Create an itinerary for my day trip."),
    ])
    
    response = llm.invoke(itinerary_prompt.format_messages(
        city=state["city"],
        interests=", ".join(state["interests"])
    ))
    state["itinerary"] = response.content
    state["messages"] += [AIMessage(content=response.content)]
    return response.content


st.set_page_config(page_title="Travel Itinerary Planner")

st.title("🌏 Travel Itinerary Planner")
st.write("Enter your API key, the city you want to visit, and your interests to generate a personalized day trip itinerary.")


groq_api_key = st.text_input("Enter your GROQ API Key:", type="password")


if groq_api_key:
    #  input for city and interests
    city = st.text_input("Enter the city for your day trip")
    interests = st.text_input("Enter your interests (comma-separated)")
    
    if st.button("Generate Itinerary"):
        if not city or not interests:
            st.warning("Please enter both city and interests.")
        else:
            
            state = {
                "messages": [],
                "city": "",
                "interests": [],
                "itinerary": "",
            }

            
            llm = ChatGroq(
                temperature=0,
                groq_api_key=groq_api_key,
                model_name="llama-3.3-70b-versatile"
            )

           
            state = input_city(city, state)
            state = input_interests(interests, state)

            
            itinerary = create_itinerary(state, llm)

            # Display
            st.subheader("📝 Generated Itinerary")
            st.text(itinerary)
else:
    st.info("Please enter your GROQ API Key to continue.")
