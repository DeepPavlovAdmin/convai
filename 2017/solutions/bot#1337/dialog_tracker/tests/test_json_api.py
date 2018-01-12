import requests
import json
from sys import argv


def start_chat(url, text=''):
    url = url + '/start'
    r = requests.post(url, json={'text': text})
    return r.json()


def message(url, chat_id, text):
    url = url + '/message'
    r = requests.post(url, json={'text': text, 'chat_id': chat_id})
    return r.json()


def message_with_print(url, chat_id, text):
    print('HUMAN: {}'.format(text))
    res = message(url, chat_id, text)
    print('BOT: {}'.format(res['text']))


def end_chat(url, chat_id):
    url = url + '/end'
    r = requests.post(url, json={'chat_id': chat_id})
    return r.json()


def conversation_example(url):
    text = ("The name-letter effect is the tendency of people to prefer the letters"
           " in their name over other letters in the alphabet. Discovered in 1985"
           " by the Belgian psychologist Jozef Nuttin, the effect has been replicated "
           " in dozens of studies.")

    chat_id = start_chat(url, text)['chat_id']

    message_with_print(url, chat_id, 'Hello dear friend!')
    message_with_print(url, chat_id, 'How are you?')
    message_with_print(url, chat_id, 'What is your name?')
    message_with_print(url, chat_id, 'What was discovered in 1985?')
    message_with_print(url, chat_id, 'I have to go. Bye!')

    end_chat(url, chat_id)


def conversation_example_without_text(url):
    text = ""

    chat_id = start_chat(url, text)['chat_id']

    message_with_print(url, chat_id, 'Hello dear friend!')
    message_with_print(url, chat_id, 'How are you?')
    message_with_print(url, chat_id, 'What is your name?')
    try:
        message_with_print(url, chat_id, 'What was discovered in 1985?')
    except json.decoder.JSONDecodeError as e:
        print(e)
    message_with_print(url, chat_id, 'You dont know?')
    message_with_print(url, chat_id, 'I have to go. Bye!')

    end_chat(url, chat_id)


if __name__ == '__main__':
    url = argv[1]
    conversation_example(url)
    print("")
    conversation_example_without_text(url)
