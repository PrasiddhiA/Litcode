from datetime import datetime

# langchain stuff
from langchain.agents import AgentExecutor, create_react_agent  
from langchain_openai import ChatOpenAI                         
from langchain_core.prompts import PromptTemplate               
from langchain_core.tools import tool                           

# streamlit stuff
import streamlit as st
from langchain_community.callbacks.streamlit import StreamlitCallbackHandler
from langchain_community.chat_message_histories import StreamlitChatMessageHistory

# open ai stuff
import openai
from openai import OpenAIError

# for database query part
from langchain_community.agent_toolkits.sql.toolkit import SQLDatabaseToolkit
from langchain_community.utilities import SQLDatabase
from sqlalchemy import create_engine

# for model
import pickle
from pathlib import Path
import os

# making schema for a tool
from pydantic import BaseModel
from typing import Union




# getting the API key from the user in the sidebar
openai_api_key = st.sidebar.text_input("OpenAI API Key", type="password")

if not openai_api_key:
    st.warning('Please provide an OpenAI API key.')

else:
    try:
        openai.api_key = openai_api_key
        openai.models.list()                            # checking if api key is valid  
        st.sidebar.success("API Key Verified ✅")
    except OpenAIError:
        st.error("Invalid API Key ❌")
    except Exception as e:
        st.sidebar.error(f"Error: {str(e)}")


# initializing path to the database
db_path = Path(os.getcwd()) / 'datasets' / "traindb.db"
db_path = f"sqlite:///{str(db_path)}"

# creating sql engine
engine = create_engine(db_path)
db = SQLDatabase(engine = engine)

# llm model used
llm = ChatOpenAI(
    streaming = True,
    model = 'gpt-4o-mini',
    temperature= 0.1,
    api_key = openai_api_key,
    max_tokens= 2048
)

# input schema for predict range tool, LLM cant pass more than one variable into the function without it i dont know why
class PredictRangeInput(BaseModel):
    start_date: str
    end_date: str

# only works with JSON input :sob:
@tool
def predict_range(input_data: Union[str,PredictRangeInput]):
    '''
    Predict sales in a range of dates. Provide input as a valid JSON with the following keys 
    ["start_date", "end_date"] for example {"start_date": "2025-01-07", "end_date": "2025-01-22"}
    Call only when dates are beyond April 30 2024.

    Args:
        input_date(str): The date at the beginning of the range to be predicted
        end_date(str): The date at the end of the range to be predicted
    Returns:
        dict: Predicted sales between start and end dates.
    '''
    
    with open('app/model.pkl', 'rb') as f:
        m = pickle.load(f)

    if isinstance(input_data,str):
        input_data=PredictRangeInput.model_validate_json(input_data)
    elif isinstance(input_data, dict):
        input_data = PredictRangeInput(**input_data)

    start_date = datetime.strptime(input_data.start_date, "%Y-%m-%d")
    end_date = datetime.strptime(input_data.end_date, "%Y-%m-%d")

    train_end_date = datetime(2024,4,30)

    prediction_period = (end_date - train_end_date).days

    future = m.make_future_dataframe(periods = prediction_period, freq = 'D')
    forecast = m.predict(future)

    filtered_forecast = forecast[(forecast['ds'] >= start_date) & (forecast['ds'] <= end_date)]

    return filtered_forecast[['ds', 'yhat']].to_dict(orient = "records")

@tool
def predict_specific_date(target_date: str):
    """
    Predict sales for a specific date. Call only when the date is beyond the 30th of April 2024.
    
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
    train_end_date = datetime(2024,4,30)

    if target_date > train_end_date:
        prediction_period = (target_date - train_end_date).days
    else:
        return 'Future prediction will not happen on dates before 30th of April, use SQL tools instead.'
    
    future = m.make_future_dataframe(periods = prediction_period, freq = 'D')
    forecast = m.predict(future)

    return forecast[['ds', 'yhat']].iloc[-1].to_dict()

@tool
def date_now():
    '''
    Returns the current date. Use when information about current day, week, month or year is asked.
    Keywords like 'this', 'now', 'current' should result in the use of this tool
    Always call this tool without the parenthesis! (like date_now and not date_now())
    '''

    return datetime.now().strftime("%Y-%m-%d")

# PROCESS TO CREATE THE AGENT
sql_toolkit = SQLDatabaseToolkit(db = db, llm = llm)                # exisiting SQL toolkit
sql_tools = sql_toolkit.get_tools()
tools = sql_tools + [predict_specific_date,predict_range,date_now]  # combine custom tools with pre-existing toolkit

# prompt, taken off of langchain hub (hub/hwchase17/react-chat) and modified a bit to suit the app's needs
prompt = PromptTemplate.from_template('''Assistant is a large language model trained by OpenAI.

Assistant is designed to be able to assist with a wide range of tasks, from answering simple questions to providing in-depth explanations and discussions on a wide range of topics. As a language model, Assistant is able to generate human-like text based on the input it receives, allowing it to engage in natural-sounding conversations and provide responses that are coherent and relevant to the topic at hand.

Assistant is constantly learning and improving, and its capabilities are constantly evolving. It is able to process and understand large amounts of text, and can use this knowledge to provide accurate and informative responses to a wide range of questions. Additionally, Assistant is able to generate its own text based on the input it receives, allowing it to engage in discussions and provide explanations and descriptions on a wide range of topics.

Overall, Assistant is a powerful tool that can help with a wide range of tasks and provide valuable insights and information on a wide range of topics. Whether you need help with a specific question or just want to have a conversation about a particular topic, Assistant is here to assist.

TOOLS:
------
Assistant has access to the following tools:

{tools}
To use a tool, please use the following format:

Thought: Do I need to use a tool? Yes
Action: the action to take, should be one of [{tool_names}]
Action Input: the input to the action
Observation: the result of the action

When you have a response to say to the Human, or if you do not need to use a tool, you MUST use the format:

Thought: Do I need to use a tool? No
Final Answer: [your response here]

Begin!
Remember, the trained prediction model you have access to has been trained from 1st January 2020, up till the 30th of April 2024, so for any human questions falling under that range should be answered by querying the database, and for dates after the specified range use the prediction tools at your disposal. If the question has overlapping dates, seamlessly integrate both the answers. For any dates before 1st Jan 2020, say that you dont have data on that.

Previous conversation history:
{chat_history}

New input: {input}
{agent_scratchpad}
''')

# creating the agent + agent executor (used to show thought process)
agent = create_react_agent(llm, tools, prompt)
agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)

# chat history part
history = StreamlitChatMessageHistory(key="chat_messages")
if len(history.messages) == 0:
    history.add_ai_message("How can I help you today?")             # initial message 


for msg in history.messages:
    st.chat_message(msg.type).write(msg.content)                    # displaying previous messages

if prompt := st.chat_input():
    st.chat_message("user").write(prompt)                           # user message
    history.add_user_message(prompt)                                # adding it to history
    
    with st.chat_message("assistant"):  
        st_callback = StreamlitCallbackHandler(st.container())      # callback handler for thought process
        response = agent_executor.invoke(
            {"input": prompt,"chat_history":history}, {"callbacks": [st_callback]}      # invoke with st_callback for thought process display
        )
        st.write(response["output"])
        history.add_ai_message(response["output"])                  # add ai message to history