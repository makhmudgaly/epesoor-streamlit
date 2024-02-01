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
embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_AI_KEY)
llm = ChatOpenAI(temperature=0, openai_api_key=OPENAI_AI_KEY)
chain = load_qa_chain(llm, chain_type="stuff")
st.title("💬 Ассистент") 

@st.cache_resource(show_spinner=False)
def load_data():
    with st.spinner(text="Жүйе қосылып жатыр / Идет загрузка системы ..."):
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

        return vectorstore


        

vectorstore = load_data()


if "messages" not in st.session_state:
    st.session_state["messages"] = [{"role": "assistant", "content": "Cізге қалай көмектесе аламын ? / Как я могу Вам помочь?"}]

for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg["content"])

if prompt := st.chat_input("Қандай сұрағыңыз бар? / Ваш вопрос?"):

    st.session_state.messages.append({"role": "user", "content": prompt})
    st.chat_message("user").write(prompt)
    with st.chat_message("assistant"):
        with st.spinner("Сұрақтың анализі жүріп жатыр / Идет анализ вопроса ..."):
            docs = vectorstore.similarity_search(prompt)

            msg = chain.run(input_documents=docs, question=prompt)
            st.session_state.messages.append(msg)
            st.write(msg)