from dotenv import load_dotenv
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage
from typing import List
from callbacks import AgentCallbackHandler


load_dotenv()

@tool
def get_text_length(text: str) -> int:
    """Returns the length of the text by characters"""
    text = text.strip("'\n").strip('"')
    # stripping away non-alphanumeric characters just in case
    return len(text)

def find_tool_by_name(tools: List, tool_name: str):
    for tool in tools:
        if tool.name == tool_name:
            return tool
    raise ValueError(f"Tool with name {tool_name} not found")

if __name__ == '__main__':
    print('Hello Krishna')
    tools = [get_text_length]
    
    llm = ChatOpenAI(temperature=0, callbacks=[AgentCallbackHandler()])
    # Bind tools to the LLM
    llm_with_tools = llm.bind_tools(tools)
    
    messages = [
        HumanMessage(content="What is the length of 'DOG' in characters?")
    ]

    while True:
    
        # First agent step
        response = llm_with_tools.invoke(messages)
        print(f"Agent response: {response}")
        
        # Check if the response contains tool calls
        if len(response.tool_calls) > 0:
            messages.append(response)
            for tool_call in response.tool_calls:
            # tool_call = response.tool_calls[0]
                tool_name = tool_call["name"]
                tool_to_use = find_tool_by_name(tools, tool_name)
                tool_input = tool_call["args"]
                observation = tool_to_use.invoke(tool_input)
                print(f'{observation=}')
                messages.append(ToolMessage(content=str(observation), tool_call_id=tool_call["id"]))
            continue # go back to the beginning of the loop
            # final_response = llm_with_tools.invoke(messages)
        
        print(f"Response: {response.content}")
        break