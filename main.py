import os
from turtle import mode

from dotenv import load_dotenv
load_dotenv()

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore

print("....Initializing components....")

embeddings = OpenAIEmbeddings()
llm = ChatOpenAI(model="gpt-5.2")

vectorstore = PineconeVectorStore(index_name=os.environ["PINECONE_INDEX_NAME"], embedding=embeddings)

retriever = vectorstore.as_retriever(search_kwargs={"k":3})

prompt_template = ChatPromptTemplate.from_template(
    """Answer the question based only on the following context:

{context}

Question: {question}

Provide a detailed answer:"""
)

def format_docs(docs):
    """Format retrieved documents into a single string"""
    return "\n\n".join(doc.page_content for doc in docs)


def retrieval_chain_without_lcel(query:str):
    """
    Simple retrieval chain without LCEL.
    Manually retrieves documents, formats them, and generates a response.

    Limitations:
    - Manual step-by-step execution
    - No built-in streaming support
    - No async support without additional code
    - Harder to compose with other chains
    - More verbose and error-prone
    """
    # Step 1: Retrieve relevant chunks related to the query from the Vector database
    chunks = retriever.invoke(query)

    # Step 2: Convert the chunks into a single text using format_docs method above
    context = format_docs(chunks)

    # Step 3: Format the prompt with context and original user query
    final_messages = prompt_template.format_messages(context=context, question=query)

    # Step 4: Invoke LLM with the formatted messages
    response = llm.invoke(final_messages)

    # Step 5: Return response.content
    return response.content


if __name__ == "__main__":
    print(".....Retrieving.....")

    query = "What is pinecone in machine learning?"

    # ========================================================================
    # Option 0: Raw invocation without RAG
    # ========================================================================
    print("\n" + "=" * 70)
    print("IMPLEMENTATION 0: Raw LLM Invocation (No RAG)")
    print("=" * 70)
    result_raw = llm.invoke([HumanMessage(content=query)])
    print("\nAnswer:")
    print(result_raw.content)

        # ========================================================================
    # Option 1: RAG - Without LCEL
    # ========================================================================
    print("\n" + "=" * 70)
    print("IMPLEMENTATION 1: Rag - Without LCEL")
    print("=" * 70)
    result_rag_wo_lcel = retrieval_chain_without_lcel(query)
    print("\nAnswer:")
    print(result_rag_wo_lcel)