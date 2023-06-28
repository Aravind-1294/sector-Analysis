import streamlit as st
from streamlit_lottie import st_lottie
import requests
from langchain import OpenAI, ConversationChain, LLMChain, PromptTemplate
from langchain.chains import LLMChain
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS

st.set_page_config(page_title="SectorGPT",page_icon="📈",layout='wide')
st.subheader('Hello :wave:! Investor welcome to ')
st.title('SectorGPT')
st.subheader("Your Ultimate Indian Stock Market Sector Analysis Assistant!")
with st.container():
    st.write("---")
    query = st.text_area("Ask any question related to sector analysis",height=10)
    if st.button("Analyze"):
        with st.spinner('Analyzing Data ...'):
          embed = OpenAIEmbeddings(openai_api_key='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')
          new_db = FAISS.load_local("secor_analysis_index", embed)
          answer = new_db.similarity_search(query)
          template = """you are a financial advisor who gives the sector analysis of various sectors and companies in india. Finanical advisor gives their analysis with a vision of investments. 
    Financial advisor give  all the important points required for investments in indian stock markets from sector point of view.
    Overall, Financial advisor is a powerful tool that can help with a wide range of tasks and provide valuable insights and information on financial topics. Whether you need help with a specific question or just want to have a conversation about a particular topic, Financial advisor is here to help.
    Answer the follwing query only using the {answer}.if not tell "I cannot give such advise."
    Human : Hi
    financial advisor : Hi, Investor how can i help you?
    
    Human: {query}
    financial advisor:"""
          prompt = PromptTemplate(input_variables=["answer","query"], template=template)
          chain = LLMChain(
              llm=OpenAI(temperature=0,openai_api_key='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'),
              prompt=prompt,
              verbose=True,)
          output = chain.predict(query=query,answer=answer)
          st.subheader('Analysis:')
          st.write(output)
