import streamlit as st
from langchain_openai import ChatOpenAI
import openai
from openai import OpenAIError
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
import time

st.title("Sales Forecasting 📈")

openai_api_key = st.sidebar.text_input("OpenAI API Key", type="password")


# getting the API key from the user
if not openai_api_key:
    st.warning('Please provide an OpenAI API key.')

else:
    try:
        
        openai.api_key = openai_api_key
        openai.models.list()  
        
        st.sidebar.success("API Key Verified ✅")

    except OpenAIError:
        st.error("Invalid API Key ❌")
    except Exception as e:
        st.sidebar.error(f"Error: {str(e)}")

model = ChatOpenAI(
    streaming = True,
    model = 'gpt-4o-mini',
    temperature= 0.1,
    api_key = openai_api_key,
    max_tokens= 2048
)


# LLM response 
def get_response(model, query, chat_history):
    template = '''
    You are a helpful assistant. Answer the user queries in a concise way.
    ''' 

    prompt = ChatPromptTemplate([
        ('system', template),
        MessagesPlaceholder(variable_name = 'chat_history'), # pass the chat history to the bot
        ('user', '{question}')
    ])
    
    chain = prompt | model | StrOutputParser()

    
    response = chain.stream({'question' : query, 'chat_history' : chat_history})

    return response


# streamlit chat history part
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []

user_query = st.chat_input('What\'s up?')


for message in st.session_state.chat_history:
    if isinstance(message, HumanMessage):
        with st.chat_message('Human', avatar = ':material/person:'):
            st.markdown(message.content) 
    else:
        with st.chat_message('AI',avatar = ':material/query_stats:' ):
            st.markdown(message.content)

if user_query is not None and user_query != '':
    st.session_state.chat_history.append(HumanMessage(user_query))

    with st.chat_message('Human', avatar = ':material/person:'):
        st.markdown(user_query)
    
    with st.chat_message('AI',avatar = ':material/query_stats:' ): 

        ai_response = st.write_stream(get_response(model, user_query, st.session_state.chat_history))

        st.session_state.chat_history.append(AIMessage(content=ai_response))


