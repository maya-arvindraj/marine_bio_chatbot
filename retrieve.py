from operator import itemgetter

from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory

# --- Vector store / retriever ---
embedding_model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

vector_store = FAISS.load_local(
    "vector_store", embedding_model, allow_dangerous_deserialization=True
)
retriever = vector_store.as_retriever(search_kwargs={"k": 5})
# print("done")
# docs = retriever.invoke("what are the different type of species")
# for i, doc in enumerate(docs):
#     print(f"--- Doc {i} ---")
#     print(doc.page_content)
#     print(doc.metadata)

# from langchain_ollama import ChatOllama

# llm = ChatOllama(
#     model="gemma4:12b",
#     temperature=0.3,
# )

# prompt = """
# You are a helpful marine biology teaching assistant. Answer the question using ONLY the context below.
#  If you don't know the answer from the context, say "I don't have that information."

#  Context:
#  these animals and preserve biodiversity

# turtles, octopuses, jellyfish, crabs, and countless other species

# . They come in a wide variety of shapes, sizes, and colors, ranging from tiny

# Marine animals are fascinating creatures that live in oceans, seas, and other

# . Marine ecosystems are home to fish, dolphins, sharks, sea turtles, octopuses

# Question: Where do marine animals live?
# """
# response = llm.invoke(prompt)
# print(response.content)

# --- LLM ---
llm = ChatOllama(
    model="gemma4:12b",  # double-check this matches `ollama list` exactly
    temperature=0.3,
)

# --- Prompt (with chat history support) ---
prompt = ChatPromptTemplate.from_messages([
    ("system", """You are a helpful marine biology teaching assistant. Answer the question using ONLY the context below.
If you don't know the answer from the context, say "I don't have that information."

Context:
{context}"""),
    MessagesPlaceholder("chat_history"),
    ("human", "{question}"),
])


def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)


# --- RAG chain ---
rag_chain = (
    {
        "context": itemgetter("question") | retriever | format_docs,
        "question": itemgetter("question"),
        "chat_history": itemgetter("chat_history"),
    }
    | prompt
    | llm
    | StrOutputParser()
)

# --- Message history wiring ---
store = {}


def get_session_history(session_id):
    if session_id not in store:
        store[session_id] = ChatMessageHistory()
    return store[session_id]


conversational_chain = RunnableWithMessageHistory(
    rag_chain,
    get_session_history,
    input_messages_key="question",
    history_messages_key="chat_history",
)

session_id = "user_001"

print("Loading Gemma locally via Ollama — first response may be slow...")
print("Chatbot ready! Type 'quit' to exit.\n")

while True:
    user_input = input("You: ")
    if user_input.lower() == "quit":
        break

    response = conversational_chain.invoke(
        {"question": user_input},
        config={"configurable": {"session_id": session_id}},
    )
    print(f"Bot: {response}\n")