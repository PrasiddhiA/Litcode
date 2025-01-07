from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage
from langgraph.prebuilt import create_react_agent
import pickle
from langchain_core.tools import tool
from langchain_core.prompts import ChatPromptTemplate





@tool
def make_predictions(days: int):
    '''Predict sales for the given number of days.'''
    with open('app/model.pkl', 'rb') as f:
        m = pickle.load(f)
    future = m.make_future_dataframe(periods=days, freq='D')
    forecast = m.predict(future)
    return forecast[['ds', 'yhat']].tail(days).to_dict(orient='records')



openai_api_key = '_' # DONT PUSH AN OPEN AI API KEY TO GIT JUST CHANGE THIS 


llm = ChatOpenAI(
    model = 'gpt-4o-mini',
    temperature= 0.1,
    api_key = openai_api_key,
    max_tokens= 2048
)

tools = [make_predictions]
agent_executor = create_react_agent(llm,tools)

input_message = {'messages' : [HumanMessage(content = "Predict sales for the next 10 days")]}

for s in agent_executor.stream(input_message, stream_mode = 'values'):
    message = s['messages'][-1]
    if isinstance(message,tuple):
        print(message)
    else:
        message.pretty_print()


# FOR EACH TOOL PROVIDE DESCRIPTION LIKE DONE
# MAKE A TOOL THAT PREDICTS A CERTAIN DATE (12TH OF APRIL 2025, OUR TEST DATA ENDS ON APRIL 30TH) KA SALES VALUE
# SUBTRACT THE DATE TO BE PREDICTED FROM APRIL 30TH, THAT MANY DAYS KA PREDICTION PERIOD PASS INTO PREDICT FUCNTION OF OUR MODEL AND THEN JUST OUTPUT THE LAST 1 RECORD AS 
# IS ALREADY DONE
# MAKE THIS IN AGENT.PY FILE ONLY, INTEGRATION INTO APP.PY WILL HAPPEN BADME