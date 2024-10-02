import ollama
import bs4
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import WebBaseLoader
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import OllamaEmbeddings
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

# 1. Load the data
loader = WebBaseLoader(
    web_paths=("https://alettertoyoubyvaishvikhandelwal.blogspot.com/2021/08/a-letter-to-you.html",),
    bs_kwargs=dict(
        parse_only=bs4.SoupStrainer(
            class_=("post-content", "post-title", "post-header")
        )
    ),
)
docs = loader.load()
text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=200)
splits = text_splitter.split_documents(docs)

# 2. Create Ollama embeddings and vector store
embeddings = OllamaEmbeddings(model="llama3")
vectorstore = Chroma.from_documents(documents=splits, embedding=embeddings)

# 3. Call Ollama Llama3 model
def ollama_llm(question, context):
    formatted_prompt = f"Question: {question}\n\nContext: {context}"
    print(f"Prompt sent to the model:\n{formatted_prompt}")  # Log the prompt to verify
    response = ollama.chat(model='llama3', messages=[{'role': 'user', 'content': formatted_prompt}])
    return response['message']['content']

# 4. RAG Setup
retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
def combine_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)
def rag_chain(question):
    retrieved_docs = retriever.invoke(question)
    # Checking if text generated is using the website or not 
    if retrieved_docs:
        print("Retrieved Docs from Vectorstore:", [doc.page_content for doc in retrieved_docs])
    else:
        print("No documents retrieved from vector store.")
    formatted_context = combine_docs(retrieved_docs)
    return ollama_llm(question, formatted_context)

# 5. Use the RAG App
# result = rag_chain("What is Task Decomposition?")
# result = rag_chain("What changes are expected to be observed with Biden?")
result = rag_chain("why is the first anniversary special")
print(result)