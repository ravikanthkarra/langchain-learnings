from dotenv import load_dotenv
from typing import List
from langchain_classic import hub
from langchain_classic.agents.agent import AgentExecutor
from langchain_classic.agents import create_react_agent
 


load_dotenv()

from langchain.tools import tool
from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI
from langchain_tavily import TavilySearch



llm = ChatOpenAI(model="gpt-4")
tools = [TavilySearch()]
react_prompt = hub.pull("hwchase17/react")

agent = create_react_agent(
        llm=llm,
        tools=tools,
        prompt=react_prompt,
)
agent_executor = AgentExecutor(agent=agent, tools=tools, verbose = True)
chain = agent_executor

def main():
    # print("Hello from search-agent!")
    query_content = "Search 3 job postings for an ai engineer using langchain in the bay area on linkedin and list their details"
    result = chain.invoke(
                input={
                    "input" : query_content
                }
    )
    print(result)


if __name__ == "__main__":
    main()