# ChatbotwppBibleIA

Bot de WhatsApp que responde, em português, a perguntas sobre a Bíblia. Uma API Flask recebe mensagens por webhook, consulta o histórico na WAHA e envia ao modelo Groq uma pergunta acompanhada de trechos recuperados do Chroma. O texto gerado é enviado de volta pelo WhatsApp via WAHA. As referências bíblicas são solicitadas no prompt, mas o código não confere as citações da resposta.

## Como está organizado

- `wpp_bot_ai/app.py`: endpoint `POST /chatbot/webhook/`. Lê `payload.from` e `payload.body`, ignora chats com `@g.us`, consulta até 10 mensagens do histórico, gera uma resposta e a envia. A resposta HTTP é `{"status":"success"}` quando o fluxo termina.
- `wpp_bot_ai/services/waha.py`: cliente HTTP da WAHA em `http://waha:4000`, com sessão fixa `default`. Usa os endpoints de histórico, digitação e envio de texto.
- `wpp_bot_ai/bot/ai_bot.py`: usa `llama-3.3-70b-versatile` via Groq, embeddings Hugging Face e Chroma em `/app/chroma_data`. Recupera até 30 documentos por pergunta e inclui o histórico no prompt.
- `wpp_bot_ai/rag/rag.py` e `wpp_bot_ai/rag/data/biblia_13V.sql`: tentativa de carregar a base bíblica, dividir documentos em trechos e gravar embeddings no Chroma. O arquivo SQL é um dump MySQL; o script, como está, não oferece um procedimento de ingestão reproduzível (veja **Limitações**).
- `wpp_bot_ai/docker-compose.yml` e `wpp_bot_ai/Dockerfile.api`: dois serviços, `waha` e `api`, publicados nas portas 4000 e 5000. O diretório `wpp_bot_ai` é montado em `/app`, inclusive `chroma_data`.

## Pré-requisitos e configuração

- Docker com o plugin Compose, para o caminho descrito abaixo. A imagem da API usa Python 3.11; as dependências estão fixadas em `wpp_bot_ai/requirements.txt`.
- Acesso à WAHA para conectar uma conta de WhatsApp à sessão `default` e configurar o encaminhamento dos eventos de mensagem ao webhook. O Compose não cria a sessão nem registra o webhook.
- Credenciais próprias para Groq e Hugging Face. `python-decouple` lê `GROQ_API_KEY` e `HUGGINGFACE_API_KEY` do ambiente ou de um arquivo `.env` no diretório da aplicação. Não inclua esse arquivo no commit; configure também como as credenciais chegarão ao contêiner `api` (o Compose atual não declara `env_file` nem repassa essas variáveis).
- Um índice Chroma populado e compatível em `wpp_bot_ai/chroma_data`. O código consulta `/app/chroma_data`, mas a inicialização da API não gera o índice.

A WAHA é alcançada pelo nome de serviço `waha` dentro da rede do Compose; executar somente `python app.py` fora dela não resolverá esse endereço sem adaptar a configuração do código ou da rede. Configure no lado da WAHA o destino do webhook como `http://api:5000/chatbot/webhook/` quando ambos os serviços estiverem na mesma rede do Compose. O handler pressupõe um JSON com `payload.from` (identificador do chat) e `payload.body` (texto); não há tratamento para outros formatos de evento.

## Execução com Compose

A partir da raiz do repositório:

```sh
cd wpp_bot_ai
docker compose up --build
```

O Compose publica a WAHA na porta 4000 e a API na porta 5000 do host. Depois de configurar a sessão `default` e o encaminhamento na WAHA, mensagens privadas entregues ao webhook seguem o fluxo descrito acima. Para encerrar os contêineres, execute `docker compose down` no mesmo diretório. Não exponha essas portas à internet sem proteção adicional: o servidor Flask inicia em modo debug e o webhook não autentica requisições.

## Verificação e testes

Não há suíte de testes do projeto neste repositório. Para conferir o arquivo de configuração, use `docker compose config` dentro de `wpp_bot_ai` em um ambiente com Docker Compose instalado. Um teste de ponta a ponta requer as credenciais, a sessão WAHA conectada, o webhook registrado e um índice Chroma utilizável; `docker compose up --build` por si só não verifica nem provisiona esses recursos. Evite usar uma mensagem real como teste sem considerar os efeitos de envio pelo WhatsApp e de chamadas ao Groq.

## Limitações conhecidas

- `rag/rag.py` usa o caminho Windows `rag\\data\\biblia_13V.sql`, enquanto a imagem Docker é Linux. Além disso, passa o caminho de um dump MySQL para `SQLDatabaseLoader` sem configurar uma conexão e consultas SQL; não há etapa automática de ingestão no Compose. O arquivo `chroma_data` versionado não substitui uma validação do índice.
- O tratamento do histórico em `ai_bot.py` classifica `fromMe` como mensagem humana e as demais como mensagem do assistente. Isso pode inverter os papéis da conversa.
- O webhook não valida assinatura/origem, não filtra tipos de evento além de grupos, não trata falhas de rede ou respostas inválidas da WAHA/Groq e não impede eventuais mensagens de saída de retornarem ao fluxo. O cliente WAHA não verifica os códigos HTTP de envio. Use apenas em ambiente controlado até corrigir esses pontos.
- O modelo pode produzir respostas inexatas, inclusive citações bíblicas. Confirme passagens e interpretações antes de usá-las como referência.
