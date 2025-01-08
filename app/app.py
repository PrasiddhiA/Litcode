import streamlit as st
from langchain_openai import ChatOpenAI
import openai
from openai import OpenAIError
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from pathlib import Path
import pandas as pd
from langchain_community.utilities import SQLDatabase
from sqlalchemy import create_engine
import os
from langchain_community.agent_toolkits import create_sql_agent
from datetime import datetime
from langchain_core.tools import tool
from langgraph.prebuilt import create_react_agent
import pickle


# Path to the database where train data is stored
db_path = Path(os.getcwd()) / 'datasets' / "testdb.db"
db_path = f"sqlite:///{str(db_path)}"

# creating sql agent 
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

# Tool for predicting sales for 'x' number of days from now
@tool
def make_predictions(days: int):
    """
    Predict sales for the given number of days.
    """
    with open('app/model.pkl', 'rb') as f:
        m = pickle.load(f)

    # today's date
    date_today = datetime.today()

    # date at which training had stopped
    end_date = datetime(2024,4,30)

    days_since_end_date = (date_today - end_date).days

    total_pred_days = days_since_end_date + days

    future = m.make_future_dataframe(periods=total_pred_days, freq='D')
    forecast = m.predict(future)
    return forecast[['ds', 'yhat']].tail(days).to_dict(orient='records')

# tool for predicting sales on a specific date (has to invoke the model, doesnt work on past data!)
@tool
def predict_specific_date(target_date: str):
    """
    Predict sales for a specific date.
    
    Args:
        target_date (str): The date to predict sales for, in the format 'YYYY-MM-DD'.
    
    Returns:
        dict: The predicted sales value for the given date.
    """
    with open('app/model.pkl', 'rb') as f:
        m = pickle.load(f)

    # date which we want to predict
    target_date = datetime.strptime(target_date, '%Y-%m-%d')

    # date at which the training stopped in our model
    end_date = datetime(2024,4,30)

    if target_date > end_date:
        prediction_period = (target_date - end_date).days
    else:
        return 'Future prediction will not happen on dates before 30th of April'
    
    future = m.make_future_dataframe(periods = prediction_period, freq = 'D')
    forecast = m.predict(future)

    return forecast[['ds', 'yhat']].iloc[-1].to_dict()






sql_agent = create_sql_agent(model, db = db, agent_type= 'openai-tools', verbose = True)
prediction_tools = [make_predictions, predict_specific_date]
prediction_agent = create_react_agent(model, prediction_tools)

tools = [sql_agent,prediction_agent]
agent = create_react_agent(model, tools)






def get_response(agent, query, chat_history):
    """
    Handles interaction with the SQL agent and Prediction agent by processing the user's query and incorporating chat history.

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