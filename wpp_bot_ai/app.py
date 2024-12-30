from flask import Flask, request, jsonify
import time
from bot.ai_bot import AIBot
from services.waha import Waha

app = Flask(__name__)


@app.route('/chatbot/webhook/', methods=['POST'])
def webhook():

    data = request.json
    chat_id = data['payload']['from']
    received_message = data['payload']['body']

    is_group = '@g.us' in chat_id
    if is_group:
        return jsonify({'status': 'success', 'message': 'Mensagem de grupo ignorada.'}), 200

    waha = Waha()

    waha.start_typing(chat_id=chat_id)

    waha.send_message(
        chat_id=chat_id,
        message='Olá! Sou o ChatBot Gênesis. Fui criado pela inspiração de Deus na vida de um estudante de Ciência da Computação. Utilizo Inteligência Artificial para ajudá-lo a conhecer os ensinamentos bíblicos. Você tem alguma pergunta específica ou gostaria de explorar um tema em particular? Estou aqui para ajudar!'
    )

    waha.stop_typing(chat_id=chat_id)

    time.sleep(2.5)

    ai_bot = AIBot()

    waha.start_typing(chat_id=chat_id)

    history_messages = waha.get_history_messages(
        chat_id=chat_id,
        limit=10,
    )

    response_message = ai_bot.invoke(
        history_messages=history_messages,
        question=received_message,
    )

    waha.send_message(
        chat_id=chat_id,
        message=response_message,
    )

    waha.stop_typing(chat_id=chat_id)

    return jsonify({'status': 'success'}), 200


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
