#pip install langgraph langchain-openai langchain-chroma chromadb
#export OPENAI_API_KEY="your-api-key"

from typing import TypedDict

from langgraph.graph import StateGraph, START, END

from langchain_openai import OpenAIEmbeddings
from langchain_openai import ChatOpenAI

from langchain_chroma import Chroma
from langchain.schema import HumanMessage


# -----------------------------------
# Documents
# -----------------------------------

documents = [
    "Docker is a containerization platform.",
    "Kubernetes is used for container orchestration.",
    "LangChain is a framework for building LLM applications.",
    "Playwright is a browser automation framework."
]

# -----------------------------------
# Embeddings + Vector Store
# -----------------------------------

embedding_model = OpenAIEmbeddings(
    model="text-embedding-3-small"
)

vector_store = Chroma.from_texts(
    texts=documents,
    embedding=embedding_model
)

# -----------------------------------
# LLM
# -----------------------------------

llm = ChatOpenAI(
    model="gpt-4o-mini",
    temperature=0
)

# -----------------------------------
# Shared State
# -----------------------------------

class RagState(TypedDict):
    question: str
    context: str
    answer: str


# -----------------------------------
# Node 1 : Retrieve
# -----------------------------------

def retrieve(state: RagState):

    docs = vector_store.similarity_search(
        state["question"],
        k=3
    )

    context = "\n".join(
        doc.page_content
        for doc in docs
    )

    return {
        "context": context
    }


# -----------------------------------
# Node 2 : Generate Answer
# -----------------------------------

def generate(state: RagState):

    prompt = f"""
Use only the context below.

Context:
{state['context']}

Question:
{state['question']}
"""

    response = llm.invoke(
        [HumanMessage(content=prompt)]
    )

    return {
        "answer": response.content
    }


# -----------------------------------
# Build Graph
# -----------------------------------

graph = StateGraph(RagState)

graph.add_node("retrieve", retrieve)
graph.add_node("generate", generate)

graph.add_edge(START, "retrieve")
graph.add_edge("retrieve", "generate")
graph.add_edge("generate", END)

app = graph.compile()

# -----------------------------------
# Run
# -----------------------------------

result = app.invoke({
    "question": "What is Docker?"
})

print(result["answer"])