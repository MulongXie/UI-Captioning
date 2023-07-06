import openai
import tiktoken


class OpenAI:
    def __init__(self, role=None, model='gpt-4'):
        openai.api_key = open('C:/Mulong/Code/UI-Captioning/utils/llm/openaikey.txt', 'r').readline()
        self.role = role if role else 'You are a mobile virtual assistant that understands and interacts with the user interface to complete a given task.'
        self.model = model

    def ask_openai_prompt(self, prompt, role, printlog=False):
        if role is None:
            conversation = [{'role': 'user', 'content': prompt}]
        else:
            conversation = [{'role': 'system', 'content': role}, {'role': 'user', 'content': prompt}]
        if printlog:
            print('*** Asking ***\n', conversation)
        resp = openai.ChatCompletion.create(
                model=self.model,
                messages=conversation
            )
        if printlog:
            print('\n*** Answer ***\n', resp['choices'][0].message, '\n')
        return dict(resp['choices'][0].message)

    def ask_openai_conversation(self, conversation, printlog=False):
        if printlog:
            print('*** Asking ***\n', conversation)
        resp = openai.ChatCompletion.create(
                model=self.model,
                messages=conversation
            )
        if printlog:
            print('\n*** Answer ***\n', resp['choices'][0].message, '\n')
        return dict(resp['choices'][0].message)

    def count_token_size(self, string):
        enc = tiktoken.encoding_for_model("gpt-3.5-turbo")
        return len(enc.encode(string))
