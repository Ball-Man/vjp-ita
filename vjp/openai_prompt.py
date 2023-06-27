from typing import Tuple
import openai
import tiktoken

MAX_TOKEN_FOR_PROMPT = 5

class Prompt:
    def __init__(self,
                 template: str,
                 verbalizer: Tuple[str, str],
                 model_short_name: str="gpt-3.5-turbo",
                 model_long_name: str="gpt-3.5-turbo-16k",
                 few_shot_data: list=None) -> None:
        self.template = template
        self.verbalizer = verbalizer

        self.model_short_name = model_short_name
        self.model_long_name = model_long_name

        self.short_tokenizer = tiktoken.encoding_for_model(model_short_name)

        self.few_shot_data = few_shot_data
        self.cache = {}

    def create_prompt(self, process: dict,
                      with_mot: bool=False) -> list[dict]:
        messages = [
            {"role": "system",
             "content": self.template.format(*self.verbalizer)},
        ]
        if self.few_shot_data is not None:
            messages.extend(self._get_few_shots())

        text = self._process_message(process, with_mot)
        messages.append({"role": "user",
                         "content": f"###START OF USER TEXT\n{text}\n###END OF USER TEXT"})

        return messages

    def send_prompt(self, prompt: list[dict], mockup: bool=False) -> str:
        num_token = len(self.short_tokenizer.encode(self.get_nice_prompt(prompt)))

        if num_token > 4090:
            model_name = self.model_long_name
            print(f"WARNING: Context size for message too long({num_token})")
        else:
            model_name = self.model_short_name

        hashable_prompt = tuple(tuple(sorted(turn.items())) for turn in prompt)
        if model_name in self.cache and hashable_prompt in self.cache[model_name]:
            return self.cache[model_name][hashable_prompt]

        def _actually_send(prompt):
            if mockup:
                return self.verbalizer[1]
            import time
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
                    time.sleep(0.5)
        res = _actually_send(prompt)

        if model_name not in self.cache:
            self.cache[model_name] = {}

        self.cache[model_name][hashable_prompt] = res
        return res

    def interpret_response(self, response: str):
        response = response.lower()
        if self.verbalizer[0] in response:
            return 1
        elif self.verbalizer[1] in response:
            return 0
        else:
            raise Exception("Result not in the correct format")

    def get_nice_prompt(self, prompt: list[dict]):
        res = ""
        for message in prompt:
            role, content = message['role'], message['content']
            res += f"###{role}\n"
            res += f'{content}\n'
        return res

    def _get_few_shots():
        #TODO
        return []

    def _process_message(self, process: dict, with_mot: bool=False):
        message = process['preliminaries']
        if with_mot:
            message += process['decisions']

        return message
