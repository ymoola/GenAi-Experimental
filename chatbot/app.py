from langchain_community.chat_models import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

import streamlit as st
import os
from dotenv import load_dotenv
load_dotenv()

os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")

##Prompt template

prompt = ChatPromptTemplate.from_messages(
    
    [
        ("system", "You are a helpful assistant. Please respond to the user queries"),
        ("user", "Question:{question}")
        
    ]
    
    
)

##streamlit framework
st.title("Langchain Demo with OPENAI")
input_text = st.text_input("Search topic:")

#LLM
llm = ChatOpenAI(model="gpt-3.5-turbo")
output_parser = StrOutputParser()
chain = prompt | llm | output_parser

if input_text:
    st.write(chain.invoke({'question':input_text}))