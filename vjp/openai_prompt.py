from functools import lru_cache
import openai
import tiktoken
import vjp.text as text

MAX_TOKEN_FOR_PROMPT = 2000

@lru_cache
def send_prompt(prompt, model_name):
    while True:
        try:
            res = openai.ChatCompletion.create(
                model=model_name,
                messages=prompt,
                max_tokens=MAX_TOKEN_FOR_PROMPT,
                temperature=0
                )
            return res['choices'][0]['message']['content']
        except Exception as e:
            print(str(e))

class Prompt:
    def __init__(self, dataset: list[dict], model_name: str="gpt-3.5-turbo", few_shot_data: list=None) -> None:
        self.dataset = dataset
        self.model_name = model_name
        self.tokenizer = tiktoken.encoding_for_model(model_name)
        self.few_shot_data = few_shot_data
        self.cache = {}
        
    def create_prompt(self, with_mot: bool=False) -> list[list[dict]]:
        ret = []
        for process in self.dataset:
            messages = [
                {"role": "system", "content": "You are a system that decide, given some informations, if a giuridical appeal is upheld or rejected in the italian laws. The format for the output must be: \"Result: uphold\" or \"Result: reject\""},
            ]
            if self.few_shot_data is not None:
                messages.extend(self._get_few_shots())
            messages.append({"role": "user", "content": self._process_message(process, with_mot)})
            ret.append(messages)

        return ret

    def send_prompt(self, prompt: list[dict]) -> str:
        res = send_prompt(prompt, self.model_name)

        if self.model_name not in self.cache:
            self.cache[self.model_name] = []
        
        self.cache[self.model_name].append((prompt, res))
        return res
    
    def print_nice_prompt(self, prompt: list[dict]):
        for message in prompt:
            role, content = message['role'], message['content']
            print(f"###{role}")
            print(content, "\n")

    def _get_few_shots():
        #TODO
        return []
    
    def _process_message(self, process: dict, with_mot: bool=False):
        if with_mot:
            return text.shot_normalize_whites_pipeline(process['preliminaries'])\
                  + text.shot_normalize_whites_pipeline(process['decisions'])
        return text.shot_normalize_whites_pipeline(process['preliminaries'])
