from langchain_community.llms import LlamaCpp
from langchain import PromptTemplate, LLMChain
from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import (
    StreamingStdOutCallbackHandler
)
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceInstructEmbeddings
from langchain_community.document_loaders import UnstructuredFileLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader

SUB_DOC = './f_doc'
SUB_PDF = './f_pdf'

template = """Question: {question}

Answer:"""

prompt = PromptTemplate(template=template, input_variables=["question"])
callback_manager = CallbackManager([StreamingStdOutCallbackHandler()])

llm = LlamaCpp(
    model_path="./models/codellama-7b-python.Q2_K.gguf",
    n_ctx=6000,
    n_batch=30,
    callback_manager=callback_manager,
    temperature=0.9,
    max_tokens=4095,
    n_parts=1,
    verbose=True
)

llm_chain = LLMChain(prompt=prompt, llm=llm)

loader = UnstructuredFileLoader(SUB_DOC + "/Canonical_About_Me.docx")
documents = loader.load()

loader = PyPDFLoader(SUB_PDF + "/Canonical_About_Me.pdf")
pages = loader.load_and_split()
print(pages[0])

text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=50)
docs = text_splitter.split_documents(documents)

# embedding engine
hf_embedding = HuggingFaceInstructEmbeddings()

db = FAISS.from_documents(pages, hf_embedding)

# save embeddings in local directory
db.save_local("faiss_AiArticle")

# load from local
db = FAISS.load_local("faiss_AiArticle/", embeddings=hf_embedding, allow_dangerous_deserialization=True)

query = "Give me a summary of my canonical assessment"
search = db.similarity_search(query, k=2)

template = '''Context: {context}

Based on Context provide me answer for following question
Question: {question}

Tell me the information about the fact. The answer should be from context only
do not use general knowledge to answer the query'''

prompt = PromptTemplate(input_variables=["context", "question"], template= template)
final_prompt = prompt.format(question=query, context=search)

llm_chain.run(final_prompt)