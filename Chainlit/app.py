import chainlit as cl      
from langchain.memory.buffer import ConversationBufferMemory      
from langchain.prompts import PromptTemplate
from langchain_community.llms import Ollama
from langchain.chains import LLMChain

conversation_memory = ConversationBufferMemory(memory_key="chat_history",
                                               max_len=50,
                                               return_messages=True,
                                                   )


F1_assistant_template = """
You are a motorsport assistant named "RaceBot". Your expertise 
is exclusively in providing information and advice about anything related to 
Formula One (F1). 

If a question is not about F1, respond with, "I'm sorry, I specialize 
only in F1 related queries."
Chat History: {chat_history}
Question: {question}
Answer: """

ice_cream_assistant_prompt_template = PromptTemplate(
    input_variables=["chat_history", "question"],
    template=F1_assistant_template
)

  
  
@cl.on_chat_start
def quey_llm():

    llm = Ollama(model = "Gemma")
    
    conversation_memory = ConversationBufferMemory(memory_key="chat_history",
                                                   max_len=50,
                                                   return_messages=True,
                                                   )
    llm_chain = LLMChain(llm=llm, 
                         prompt=ice_cream_assistant_prompt_template,
                         memory=conversation_memory)
    
    cl.user_session.set("llm_chain", llm_chain)   
    
            

@cl.on_message # decorator running on a loop
async def main(message: cl.Message):
    llm_chain = cl.user_session.get("llm_chain")
    
    response = await llm_chain.acall(message.content, 
                                     callbacks=[
                                         cl.AsyncLangchainCallbackHandler()])
    
    await cl.Message(response["text"]).send()
      