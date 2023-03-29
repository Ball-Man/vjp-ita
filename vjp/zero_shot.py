from typing import Sequence
import openai
import pandas as pd
from sklearn.base import BaseEstimator
import tiktoken
from tqdm.auto import tqdm
import re

MAX_TOKEN_FOR_PROMPT = 1980

# words returned from davinci model
REJECTED_WORDS = "reject", "respin", "infondat", "yes", "sì", "riget"
UPHELD_WORDS = "accolt", "upheld", "no", 

# word that mark the answer as useless
USELESS_WORDS = ('?',)


class OpenAiClassifier(BaseEstimator):
    def __init__(self, api_key, features, engine: str="davinci", max_competation_token: int=20) -> None:
        super().__init__()
        openai.api_key = api_key
        self.tokenizer = tiktoken.encoding_for_model(engine)
        self.engine = engine
        self.max_competation_token = max_competation_token

        self.remaining_size = MAX_TOKEN_FOR_PROMPT -len(self.tokenizer.encode(
                self.write_prompt_for_sample({
                    k: "" for k in features }
        )))

    def fit(self, X, y=None):
        return self

    def predict(self, X: Sequence[dict]) -> list[bool]:
        predictions = []
        for text in tqdm(X):
            prompt = self.write_prompt_for_sample(text, self.remaining_size)
            while True:
                while True:
                    responses = self.send_prompt(prompt)

                    responses = self.filter_useless_responses(responses)

                    if len(responses) >= 1:
                        break

                count_rejected = self.count_rejected(responses)
                count_upheld = self.count_upheld(responses)
                if count_upheld != count_rejected:
                    break
            
            is_upheld = count_upheld > count_rejected

            # print(count_upheld, count_rejected, responses)
            predictions.append(is_upheld)
        return predictions

    def calculate_best_threshold(self, sample: dict, max_length: int, 
                                 used_features=['fact', 'req', 'arg',
                                                'claim', 'mot', 'dec']) -> int:
        if sum(len(sample[k])
               for k in used_features
               if k in sample) > max_length:
            def argmin(x):
                return min(range(len(x)), key=lambda i: x[i])
            # chose a threshold that only cut features that makes the prompt
            # exceed the limit of tokens
            return argmin([
                abs(
                    sum(min(th, len(self.tokenizer.encode(sample[k])))
                        for k in used_features)
                    - max_length)
                for th in range(0, max_length)]) - 1
        return max_length

    def write_prompt_for_sample(self, sample: dict, max_length=2049) -> str:
        threshold = self.calculate_best_threshold(sample, max_length)

        def cut_feature(feature):
            return self.tokenizer.decode(self.tokenizer.encode(feature)[:threshold])
        
        # TODO Try to write in english the request 
        prompt = "I present you some data about an italian process and you " \
                 "have to decide if the request is uphold or reject.\n\n"
        if 'fact' in sample:
            prompt += f"The facts are:\n" \
                      f"{cut_feature(sample['fact'])}\n\n"
        if 'req' in sample:
            prompt += f"The request is:\n" \
                      f"{cut_feature(sample['req'])}\n\n"
        if 'arg' in sample:
            prompt += f"The argumentation or additional info are:\n" \
                      f"{cut_feature(sample['arg'])}\n\n"
        if 'claim' in sample:
            prompt += f"Other useful information are:\n" \
                      f"{cut_feature(sample['claim'])}\n\n"
        if 'mot' in sample:
            prompt += f"The reason for the final decision are:\n" \
                      f"{cut_feature(sample['mot'])}\n\n"
        if 'dec' in sample:
            prompt += f"The final decision is:\n" \
                      f"{cut_feature(sample['dec'])}\n\n"
            
        prompt += "Is the appeal rejected?"
                
        return prompt
    
    def filter_useless_responses(self, responses: list[str]) -> list[str] :

        
        ret = []
        for r in responses:
            if len(r.split(" ")) < 3:
                continue

            # if there are both rejected and upheld word
            # or it there is none of both
            if (any(rw in r for rw in REJECTED_WORDS) ==
                any(uw in r for uw in UPHELD_WORDS)):
                continue 
            
            if any(uw in r for uw in USELESS_WORDS):
                continue

            ret.append(r)
        return ret
    
    def count_rejected(self, responses: list[str]) -> bool:
        return sum(any(w in r for r in responses) for w in REJECTED_WORDS)
    
    def count_upheld(self, responses: list[str]) -> bool:
        return sum(any(w in r for r in responses) for w in UPHELD_WORDS)

    def send_prompt(self, prompt: str, temperature=0.7) -> bool:
        response = openai.Completion.create(
            engine=self.engine,
            prompt=prompt,
            max_tokens=self.max_competation_token,
            temperature=temperature,
            n=5,
            stop=None,
        )
        return [re.sub(r'\s+', ' ', c.text).lower() for c in response.choices]