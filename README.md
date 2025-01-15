# AI-Powered Predictive Chatbot with Time-Series Forecasting 


### Objective 

Develop an AI-based chatbot that takes user input and generates time-series predictions using a Prophet model. The project uses Streamlit for the chatbot interface, Python for parsing inputs, and Docker for deployment. 


### Technologies 

- **Chatbot Interface**: Streamlit 
- **Language Model Integration**: LangChain with OpenAI APIs 
- **Forecasting Model**: Prophet 
- **Programming Language**: Python 
- **Deployment**: Docker 


### Setup Instructions 

1. Clone the repository. 
2. Navigate to the project directory.
3. Build the Docker image: 
   ```bash 
   	docker build -t chatbot_forecast . 
4. Run the container: 
   ```bash 
      docker run -p 8501:8501 chatbot_forecast 
5. Access the application at http://localhost:8501. 


### Usage 

Enter time-series data or forecasting queries into the chatbot interface. 

The model will process the input and display predictions along with visual plots. 


### Example 

Input: 

"What will the sales trend be for the next 3 months?" 

Output: 

Predicted values showing future trends. 