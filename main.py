from dotenv import load_dotenv
from pydantic import BaseModel
from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic
from langchain_core.output_parsers import PydanticOutputParser
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

tools = [search_tool]

agent = create_agent(
    model=llm,
    tools=tools, 
    system_prompt=system_prompt,
    response_format=ResponseSearch,
)

query = "Patient has fever, sore throat and fatigue for 3 days."

#Derin Öğrenme
predicted_disease, features = predict(query)
print(f"DL Prediction: {predicted_disease}")

#XAI
_, explanation = explain(query)
print(explanation)

#Agent/Agentic AI
response = agent.invoke({"query": query})
print(response)