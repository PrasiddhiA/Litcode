from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage
from langgraph.prebuilt import create_react_agent
import pickle
from langchain_core.tools import tool
from langchain_core.prompts import ChatPromptTemplate
from prophet import Prophet
from datetime import datetime, date



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
    
    


# Initialize the chat model
openai_api_key = '-'  # DONT PUSH AN OPEN AI API KEY TO GIT JUST CHANGE THIS

llm = ChatOpenAI(
    model='gpt-4o-mini',
    temperature=0.1,
    api_key=openai_api_key,
    max_tokens=2048
)

# Add both tools to the agent
tools = [make_predictions, predict_specific_date]
agent_executor = create_react_agent(llm, tools)

# Example input to the agent for predictions
input_message = {'messages': [HumanMessage(content="Predict Sales 15 days from today")]}

# Execute the agent and stream the results
for s in agent_executor.stream(input_message, stream_mode='values'):
    message = s['messages'][-1]
    if isinstance(message, tuple):
        print(message)
    else:
        message.pretty_print()

# FOR EACH TOOL PROVIDE DESCRIPTION LIKE DONE
# MAKE A TOOL THAT PREDICTS A CERTAIN DATE (12TH OF APRIL 2025, OUR TEST DATA ENDS ON APRIL 30TH) KA SALES VALUE
# SUBTRACT THE DATE TO BE PREDICTED FROM APRIL 30TH, THAT MANY DAYS KA PREDICTION PERIOD PASS INTO PREDICT FUCNTION OF OUR MODEL AND THEN JUST OUTPUT THE LAST 1 RECORD AS 
# IS ALREADY DONE
# MAKE THIS IN AGENT.PY FILE ONLY, INTEGRATION INTO APP.PY WILL HAPPEN BADME