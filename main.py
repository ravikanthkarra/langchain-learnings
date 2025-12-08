from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain.agents.structured_output import ProviderStrategy
from langchain_openai import ChatOpenAI
from langchain_tavily import TavilySearch

from schemas import AgentResponse

load_dotenv()


llm = ChatOpenAI(model="gpt-4")
tools = [TavilySearch()]

# Create agent with structured output using ProviderStrategy (recommended for OpenAI)
agent = create_agent(
    model=llm,
    tools=tools,
    response_format=AgentResponse,
)


def main():
    query_content = "Search 3 job postings for an ai engineer using langchain in the bay area on linkedin and list their details"
    result = agent.invoke(
        {"messages": [{"role": "user", "content": query_content}]}
    )
    # Access structured response from the result
    structured_response = result["structured_response"]
    print(structured_response)

    # Access structured response from the agent
    # structured = result.get("structured_response", None)
    # print(structured if structured is not None else result)


if __name__ == "__main__":
    main()
