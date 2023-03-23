from typing import Sequence
import openai
import pandas as pd
from sklearn.base import BaseEstimator
import tiktoken
from tqdm.auto import tqdm

MAX_TOKEN_FOR_PROMPT = 2000

class OpenAiClassifier(BaseEstimator):
    def __init__(self, api_key, features, engine: str="davinci", max_competation_token: int=8) -> None:
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
                response = self.send_prompt(prompt)
                if not self.retry_prompt(response):
                    break
            predictions.append(self.is_response_rejected(response))
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
            
        prompt += "Is the proces rejected? respond with true or false"
                
        return prompt
    
    def retry_prompt(self, response: str) -> bool:
        # TODO is ok to write this rules? This is in case the response is
        # a repetition of the prompt
        if "true or false" in response:
            return True
        if len(response) < 5:
            return True
        return False
    
    def is_response_rejected(self, response: str) -> bool:
        # TODO is ok to write this rules?
        rejected_words = "rejected", "respinge", "true", "infondata"
        return any(w in response for w in rejected_words)

    def send_prompt(self, prompt: str, temperature=0.7) -> bool:
        response = openai.Completion.create(
            engine=self.engine,
            prompt=prompt,
            max_tokens=self.max_competation_token,
            temperature=temperature,
            n=1,
            stop=None,
        )
        return response.choices[0].text