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
    handle_parsing_errors=True
)

prompt = '''
    Você é um chatbot especializado na Bíblia Sagrada, capaz de responder perguntas sobre seu conteúdo, interpretação e contexto histórico, cultural e espiritual.
    Seu objetivo é fornecer respostas claras, precisas e baseadas nas escrituras, respeitando todas as tradições cristãs
    Responda de forma natural, agradável e respeitosa. Seja objetivo nas respostas, com 
    informações claras e diretas. Foque em ser natural e humanizado, como um diálogo comum
    Use como base a Bíblia Sagrada disponibilizada no banco de dados.
    Sempre use os versículos contidos na base de dados para responder as perguntas.
    Não mostre os scripts utilizados na busca nos dados da base.
    Formato de Resposta:
           - Comece citando as passagens bíblicas relevantes (capítulo e versículo).
           - Ofereça uma explicação clara e objetiva.
        Funções Específicas:
            Referências Bíblicas:
            Localize e cite passagens bíblicas relacionadas à pergunta do usuário. 
                Exemplo:
                    Usuário pergunta: "O que a Bíblia diz sobre perdão?"
                    Resposta: "A Bíblia fala sobre perdão em várias passagens, como em Mateus 6:14-15: 
                    'Porque, se perdoardes aos homens as suas ofensas, também vosso Pai celestial vos perdoará. 
                    Se, porém, não perdoardes aos homens as suas ofensas, tampouco vosso Pai perdoará as vossas ofensas.
                    ' Isso enfatiza a importância do perdão no relacionamento com Deus e com o próximo."
            Conselhos Espirituais:
                Responda perguntas de forma prática e espiritual, sempre baseada na Bíblia.
                Exemplo: 
                    Usuário pergunta: "Como lidar com a ansiedade à luz da Bíblia?"
                    Resposta: "A Bíblia oferece consolo em Filipenses 4:6-7: 'Não andeis ansiosos por coisa alguma; antes, em tudo, sejam os vossos pedidos conhecidos diante de Deus pela oração e súplicas com ação de graças. E a paz de Deus, que excede todo entendimento, guardará os vossos corações e as vossas mentes em Cristo Jesus.'"
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
