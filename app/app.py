import streamlit as st
from langchain_openai import ChatOpenAI
import openai
from openai import OpenAIError
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from langchain_experimental.agents import create_csv_agent
from pathlib import Path
import pandas as pd
from langchain_community.utilities import SQLDatabase
from sqlalchemy import create_engine
from pathlib import Path
import os
from langchain_community.agent_toolkits import create_sql_agent


db_path = Path(os.getcwd()) / 'datasets' / "testdb.db"
db_path = f"sqlite:///{str(db_path)}"

engine = create_engine(db_path)
db = SQLDatabase(engine = engine)


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
        st.sidebar.text(db_path)

    except OpenAIError:
        st.error("Invalid API Key ❌")
    except Exception as e:
        st.sidebar.error(f"Error: {str(e)}")




model = ChatOpenAI(
    model = 'gpt-4o-mini',
    temperature= 0.1,
    api_key = openai_api_key,
    max_tokens= 2048
)



agent = create_sql_agent(model, db = db, agent_type= 'openai-tools', verbose = True)
# st.markdown(agent_executor.invoke({"input" : "what are you?"}))


def get_response(agent, query, chat_history):
    """
    Handles interaction with the SQL agent by processing the user's query and incorporating chat history.

    Args:
        agent: The LangChain agent_executor instance.
        query (str): The user's query to be passed to the agent.
        chat_history (list): A list containing the history of chat interactions.

    Returns:
        dict: A dictionary with the agent's response and the updated chat history.
    """
    try:
        # Prepare the input payload with query and chat history
        input_payload = {
            "input": query,
            "chat_history": chat_history  # Optional, if the agent supports chat context
        }

        
        # Invoke the agent with the query
        response = agent.invoke(input_payload)
        
        # Return the response and updated history
        return {
            "response": response['output']  # The agent's reply
        }
    except Exception as e:
        return {
            "response": f"An error occurred while processing the query: {str(e)}"
        }






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

        ai_response = get_response(agent, user_query, st.session_state.chat_history)
        ai_response = ai_response['response']

        st.markdown(ai_response)

        st.session_state.chat_history.append(AIMessage(content=ai_response))