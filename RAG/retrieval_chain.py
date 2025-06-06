# This program is intended to demo the use of the following;
# 1. WebBaseLoader to read a webpage
# 2. RecursiveCharacterTextSplitter to split the text into chunks
# 3. Convert the documents into embeddidngs and store on FAISS DB
# 4. Cteate a Stuff document chain to retrieve the relevant chunks
# 5. Create a retrieval Chain using the FAISS retriever and document chain

from langchain_community.document_loaders import WebBaseLoader
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.prompts import ChatPromptTemplate
from langchain_google_genai import GoogleGenerativeAI
from langchain_core.documents import Document
from langchain.chains import create_retrieval_chain 

loader=WebBaseLoader("https://en.wikipedia.org/wiki/India")

docs=loader.load()

text_splitter = RecursiveCharacterTextSplitter()

documents= text_splitter.split_documents(docs)

llm=GoogleGenerativeAI(model="gemini-2.0-flash",temperature=0.5)

embeddings=GoogleGenerativeAIEmbeddings(model="models/embedding-001")

vector = FAISS.from_documents(documents, embeddings)

prompt = ChatPromptTemplate.from_template("""Answer the question based on the context: 
<context> {context} </context>
                                          
Question: {input}""")

document_chain = create_stuff_documents_chain(
    llm=llm,
    prompt=prompt
)

retriever = vector.as_retriever()
retrieval_chain = create_retrieval_chain(retriever,document_chain)

response = retrieval_chain.invoke({
    "input": "What are some major moments in the history of India?",
    "context": "You're an Indian tourist guide, and you are answering questions about India."
})
print(response['answer'])

"""
Output of the code:
--------------------
Based on the context provided, some major moments in the history of India include:

*   **1848:** Appointment of Lord Dalhousie as Governor General, leading to changes essential to a modern state.
*   **1857:** The Indian Rebellion, which led to the dissolution of the East India Company.
*   **1885:** Founding of the Indian National Congress.
*   **Post-World War I:** Emergence of Mahatma Gandhi and the non-cooperation movement.
*   **1947:** Independence and partition of India and Pakistan.
*   **1950:** Completion of the Indian Constitution.
*   **1980s:** Beginning of economic liberalization.

"""