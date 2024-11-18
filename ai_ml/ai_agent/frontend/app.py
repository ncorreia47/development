import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# from bigquery.bigquery_loader import BigQueryLoader
from data_quality.contract import Insurance
from dotenv import load_dotenv as env
from langchain_google_genai import GoogleGenerativeAI
from langchain.agents.agent_types import AgentType
from langchain_experimental.agents.agent_toolkits import create_pandas_dataframe_agent
import streamlit as st
import pandas as pd

    
def chat_gemini(dataframe=None):
    env(dotenv_path='variables.env')

    agent = create_pandas_dataframe_agent(GoogleGenerativeAI(google_api_key=os.getenv('API_KEY_GEMINI_TEST'),
                                                             model="gemini-pro",
                                                             temperature=0.5,
                                                             allow_dangerous_code=True),
                                                             dataframe,
                                                             verbose=True)
    
    if dataframe is not None or len(dataframe) > 0:
        query = st.text_input('Enter your question:')

        if st.button('Search answers about the question...'):
            if query:
                try:
                    st.spinner('Processing the best answers...')
                    result = agent.invoke({"input": {"query": query}})
                    output = result.get("output")
                    st.write(output)
                except Exception as error:
                    st.error('An error occurred!', error)
            else:
                st.write('Enter your question: ')
    else:
        st.write('Please upload a valid file: ')


def read_file(uploaded_file, sheet_index=0):
    if uploaded_file is not None:
        file_name , file_extension = uploaded_file.name.split('.')[-2], uploaded_file.name.split('.')[-1]
        file_name_w_extension = f'{file_name}.{file_extension}'

        if file_extension not in ['csv', 'xlsx', 'xls']:
            st.error("File extension not supported.")
            return None
        else:
            df = pd.read_csv(uploaded_file) if file_extension == 'csv' else pd.read_excel(uploaded_file, sheet_name=sheet_index)

        return file_name_w_extension, df


def main():
    st.title('IA Agent - Insurance Report')
    
    # Creating variables in frontend:
    username = st.text_input('Username: ')
    insurance_plan = st.text_input('Insurance Plan: ')
    initial_date = st.date_input('Initial Date: ')
    final_date = st.date_input('Final Date: ')

    # Search in table Group Name -- SELECT DISTINC GROUP_NAME FROM TABLE_NAME
    group_name = st.selectbox('Group name:', ['Group name 1', 'Group name 2', 'Group name 3'])
    uploaded_file = st.file_uploader('Import your file:', type=['csv', 'xlsx', 'xls'])


    if st.button('Salvar'):

        if uploaded_file or uploaded_file is not None:
            file_name_w_extension, df = read_file(uploaded_file=uploaded_file)
            
            st.write(f'File {file_name_w_extension} was successful imported!\nCount rows imported: {len(df)}\n')
            st.write(f'Sample Data:')
            st.write(df.head(20))

            chat_gemini(dataframe=df)

        
        else:

            try:
                insurance = Insurance(username=username, 
                                    insurance_plan=insurance_plan, 
                                    initial_date=initial_date, 
                                    final_date=final_date,
                                    group_name=group_name)
                
                if insurance:
                    print('ok')
                    

            except Exception as error:
                st.write(f'Error during validation: {error}')

if __name__ == '__main__':
    main()
