import streamlit as st
from dotenv import load_dotenv
from pydantic import BaseModel
from langchain_openai import ChatOpenAI
from langchain.agents import create_agent
from tools import search_tool
from classifier import predict, explain

load_dotenv()

class ResponseSearch(BaseModel):
    topic: str
    summary: str
    sources: list[str]
    toolsused: list[str]

llm = ChatOpenAI(model="gpt-4o-mini")

system_prompt = """
You are an assistant doctor that will help newly graduated practitioner doctors.
You will help them establish diagnosis with given symptoms and create a roadmap 
to solve their problems. Be very careful with info you find online, really think 
about it, but still remind the user that you cannot replace a real doctor.
Wrap the output in this JSON format:
{
    "topic": "...",
    "summary": "...",
    "sources": ["..."],
    "toolsused": ["..."]
}
Do not provide any extra text outside the JSON.
"""

agent = create_agent(
    model=llm,
    tools=[search_tool],
    system_prompt=system_prompt,
    response_format=ResponseSearch,
)

st.title("OpenAI Medical Asistant for Practicians")
st.write("Enter symptoms for diagnosis and roadmap.")

query = st.text_area("Symptoms", placeholder="e.g. Patient has fever, sore throat and fatigue for 3 days.")

if st.button("Analyze"):
    if query:
        with st.spinner("Analyzing..."):

            # Step 1: Deep Learning
            predicted_disease, features = predict(query)
            st.subheader("🔬 Deep Learning Prediction")
            st.write(f"Predicted disease: **{predicted_disease.upper()}**")

            # Step 2: XAI
            _, explanation = explain(query)
            st.subheader("📊 XAI Explanation (SHAP)")
            st.text(explanation)

            # Step 3: Agent
            response = agent.invoke({"query": query})
            st.subheader("🤖 AI Agent Diagnosis Report")
            st.json(response)
    else:
        st.warning("Please enter symptoms first!")