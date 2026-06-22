#pip install langchain langchain-openai langchain-chroma chromadb
#export OPENAI_API_KEY="your-api-key"

from langchain_openai import OpenAIEmbeddings
from langchain_openai import ChatOpenAI
from langchain_chroma import Chroma
from langchain.schema import HumanMessage

# ----------------------------------
# Step 1: Documents
# ----------------------------------

documents = [
    "Docker is a containerization platform.",
    "Kubernetes is used for container orchestration.",
    "LangChain is a framework for building LLM applications.",
    "Playwright is used for browser automation testing.",
    "Selenium is a popular web automation framework."
]

# ----------------------------------
# Step 2: Create Embedding Model
# ----------------------------------

embedding_model = OpenAIEmbeddings(
    model="text-embedding-3-small"
)

# ----------------------------------
# Step 3: Create Vector Store
# ----------------------------------

vector_store = Chroma.from_texts(
    texts=documents,
    embedding=embedding_model
)

# ----------------------------------
# Step 4: User Query
# ----------------------------------

query = "What is Docker?"

# ----------------------------------
# Step 5: Similarity Search
# ----------------------------------

retrieved_docs = vector_store.similarity_search(
    query=query,
    k=3
)

# ----------------------------------
# Step 6: Build Context
# ----------------------------------

context = "\n".join(
    [doc.page_content for doc in retrieved_docs]
)

# ----------------------------------
# Step 7: Create Prompt
# ----------------------------------

prompt = f"""
You are a helpful assistant.

Use ONLY the context below to answer.

Context:
{context}

Question:
{query}
"""

# ----------------------------------
# Step 8: Call LLM
# ----------------------------------

llm = ChatOpenAI(
    model="gpt-4o-mini",
    temperature=0
)

response = llm.invoke(
    [HumanMessage(content=prompt)]
)

# ----------------------------------
# Step 9: Print Answer
# ----------------------------------

print("Retrieved Context:")
print("-" * 50)
print(context)

print("\nAnswer:")
print("-" * 50)
print(response.content)