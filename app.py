__import__('pysqlite3')
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')

import streamlit as st
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain_openai import ChatOpenAI
from langchain.chains.question_answering import load_qa_chain

OPENAI_AI_KEY = st.secrets.openai_key

st.title("💬 Ассистент") 

if "messages" not in st.session_state:
    st.session_state["messages"] = [{"role": "assistant", "content": "Cізге қалай көмектесе аламын ? / Как я могу Вам помочь?"}]

@st.cache_resource(show_spinner=False)
def load_data():
    with st.spinner(text="Жүйе қосылып жатыр / Идет загрузка системы ..."):
        embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_AI_KEY)
        llm = ChatOpenAI(temperature=0, openai_api_key=OPENAI_AI_KEY)
        chain = load_qa_chain(llm, chain_type="stuff")
        loader = PyPDFLoader("./content/doc_1_updated.pdf")
        data = loader.load()
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
        texts = text_splitter.split_documents(data)
        persist_directory = './content/'
        vectorstore = Chroma.from_documents(texts, embeddings, persist_directory=persist_directory)
        vectorstore.persist()
        vectorstore=None
        vectorstore=Chroma(persist_directory=persist_directory, embedding_function=embeddings)
        vectorstore.get()

        return vectorstore, chain


        

vectorstore, chain = load_data()

if prompt := st.chat_input("Қандай сұрағыңыз бар? / Ваш вопрос?"): # Prompt for user input and save to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])

if st.session_state.messages[-1]["role"] != "assistant":
    with st.chat_message("assistant"):
        with st.spinner("Сұрақтың анализі жүріп жатыр / Идет анализ вопроса ..."):
            docs = vectorstore.similarity_search(prompt)

            response = chain.run(input_documents=docs, question=prompt)
            st.write(response)
            message = {"role": "assistant", "content": response}
            st.session_state.messages.append(message)
            

