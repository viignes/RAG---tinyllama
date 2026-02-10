import os
from langchain_ollama import ChatOllama, OllamaEmbeddings # <--- New Import
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

# 1. Setup: Use Local Ollama Models instead of OpenAI
# Ensure you have run 'ollama pull llama3' in your terminal first!
llm = ChatOllama(model="tinyllama")
embeddings = OllamaEmbeddings(model="tinyllama")

# 2. The Knowledge Base
documents = [
    "Gemini is a family of multimodal AI models developed by Google.",
    "RAG stands for Retrieval-Augmented Generation.",
    "LangChain is a framework for developing applications powered by LLMs."
]

# 3. Vector Store
print("--- Creating Vector Store (This may take a moment locally) ---")
vectorstore = Chroma.from_texts(
    texts=documents,
    embedding=embeddings
)
retriever = vectorstore.as_retriever()

# 4. The Chain
template = """Answer the question based only on the following context:
{context}

Question: {question}
"""
prompt = ChatPromptTemplate.from_template(template)

rag_chain = (
    {"context": retriever, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)

# 5. Run: Interactive Chat Loop
print("--- Chatbot Started! (Type 'exit' to quit) ---")

while True:
    user_input = input("\nYou: ")
    if user_input.lower() == "exit":
        break
    
    # Generate answer
    response = rag_chain.invoke(user_input)
    print(f"AI: {response}")