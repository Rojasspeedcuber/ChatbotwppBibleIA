import os
import streamlit as st
from decouple import config
from langchain import hub
from langchain_openai import ChatOpenAI
from langchain.agents import create_react_agent, AgentExecutor
from langchain.prompts import PromptTemplate
from langchain_community.utilities.sql_database import SQLDatabase
from langchain_community.agent_toolkits.sql.toolkit import SQLDatabaseToolkit

os.environ['OPENAI_API_KEY'] = config('OPENAI_API_KEY')

st.set_page_config(
    page_title='Bible AI',
    page_icon='biblia.png'
)

st.header('Chatbot Gênesis')

model_options = [
    'gpt-3.5-turbo',
    'gpt-4',
    'gpt-4-turbo',
    'gpt-4o-mini',
    'gpt-4o',
]

bible_options = [
    'ACF',
    'ARA',
    'ARC',
    'AS21',
    'KJA',
    'NAA',
    'NTLH',
    'NVI',
    'NVT',
]

selected_box = st.sidebar.selectbox(
    label='Selecione o modelo LLM',
    options=model_options,
)

selected_bible = st.sidebar.selectbox(
    label='Selecione a versão da base de dados',
    options=bible_options,
)

st.sidebar.markdown('### Sobre')
st.sidebar.markdown('Sou o ChatBot Gênesis. Fui criado pela inspiração de Deus na vida de um estudante de Ciência da Computação. Utilizo Inteligência Artificial para ajudá-lo a conhecer os ensinamentos bíblicos.')
st.write('Faça perguntas sobre a Bíblia')

if 'messages' not in st.session_state:
    st.session_state['messages'] = []

user_question = st.chat_input('O que deseja saber sobre a Bíblia?')

model = ChatOpenAI(
    model=selected_box,
    max_completion_tokens=400,
)


db = SQLDatabase.from_uri(
    f'sqlite:///wpp_bot_ai/databases/{selected_bible}.db')

toolkit = SQLDatabaseToolkit(
    db=db,
    llm=model,
)

system_message = hub.pull('hwchase17/react')

agent = create_react_agent(
    llm=model,
    tools=toolkit.get_tools(),
    prompt=system_message,
)

agent_executor = AgentExecutor(
    agent=agent,
    tools=toolkit.get_tools(),
    verbose=True,
)

prompt = '''
    Você é um chatbot especializado na Bíblia Sagrada, capaz de responder perguntas sobre seu conteúdo, interpretação e contexto histórico, cultural e espiritual.
    Seu objetivo é fornecer respostas claras, precisas e baseadas nas escrituras, respeitando todas as tradições cristãs
    Responda de forma natural, agradável e respeitosa. Seja objetivo nas respostas, com 
    informações claras e diretas. Foque em ser natural e humanizado, como um diálogo comum
    Use como base a Bíblia Sagrada disponibilizada no banco de dados.
    Sempre use os versículos contidos na base de dados para responder as perguntas.
    A resposta final deve ter uma formatação amigável(markdown) de vizualização para o usuário.
    Responda sempre em português brasileiro.
    Pergunta: {q}
    '''
prompt_template = PromptTemplate.from_template(prompt)


if user_question:
    for message in st.session_state.messages:
        st.chat_message(message.get('role')).write(message.get('content'))

    st.chat_message('user').write(user_question)
    st.session_state.messages.append(
        {'role': 'user', 'content': user_question})
    with st.spinner('Buscando resposta...'):
        formatted_prompt = prompt_template.format(q=user_question)
        output = agent_executor.invoke({'input': formatted_prompt})
        st.markdown(output.get('output'))
