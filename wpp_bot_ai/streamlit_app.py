import os
import streamlit as st
from langchain import hub
from decouple import config
from langchain_community.llms import HuggingFaceHub
from langchain_huggingface import ChatHuggingFace
from langchain.agents import create_react_agent, AgentExecutor
from langchain.prompts import PromptTemplate
from langchain_community.utilities.sql_database import SQLDatabase
from langchain_community.agent_toolkits.sql.toolkit import SQLDatabaseToolkit


os.environ['HUGGINGFACE_API_KEY'] = config('HUGGINGFACE_API_KEY')

st.set_page_config(
    page_title='Bible AI',
    page_icon='biblia.png'
)

st.header('Chatbot Gênesis')

model_options = [
    'meta-llama/Llama-3.3-70B-Instruct',
    'meta-llama/Llama-3.2-3B-Instruct',
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

llm = HuggingFaceHub(
    repo_id=f'{selected_box}',
    huggingfacehub_api_token=os.environ['HUGGINGFACE_API_KEY'],
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

model = ChatHuggingFace(
    llm=llm
)


db = SQLDatabase.from_uri(f'sqlite:///databases/{selected_bible}.sqlite')

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
    Não mostre os scripts utilizados na busca nos dados da base.
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
