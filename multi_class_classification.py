# %%
from utils.console import Console

import openai
from openai import OpenAI
from tqdm import tqdm
from time import sleep
from dotenv import load_dotenv
from math import exp
from random import choice
from typing import Union, Any
from multiprocessing import Pool
import os
import argparse

# Set argparse
parser = argparse.ArgumentParser(description='Demonstration of Multi Class Classification of Construction Accident Prediction from Scenarios.')
parser.add_argument('--print-log', 
    default='correct',
    help='Level of classes that will be printed on console.')
parser.add_argument('--log-level', 
    default='debug',
    help='Level of logs that will be printed on console. (default: debug)')
parser.add_argument('--scenario', 
    default='',
    help='A line or lines of script of construction accident scenario.')
args = parser.parse_args()
# set up logging
logger = Console(level=args.log_level)
# 개인 api key
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")


class ModelHandler():
    global OPENAI_API_KEY
    
    def __init__(self,
    model : str, 
    name : str = "",
    ans_word : str = "",
    api_key : str = OPENAI_API_KEY,
    temperature=.2,
    max_tokens=256,
    top_p=1,
    frequency_penalty=0,
    presence_penalty=0,
    **kwargs) -> None:
        """
        :param api_key: OpenAI API key.
        :type api_key: str
        :param model: Binary class model.
        :type model: str
        :param name: Class name(en).
        :type name: str
        :param ans_word: Expected answer.
        :type ans_word: str
        :param temperature: GPT's response temperature.
        :type temperature: float
        :param max_token: GPT's response max tokens.
        :type max_token: int
        :param top_p: GPT's top prediction.
        :type top_p: int
        :param frequency_penalty: GPT's frequency penalty.
        :type frequency_penalty: float
        :param presence_penalty: GPT's presence penalty.
        :type presence_penalty: float
        :return: None
        """

        self.name = name  # = kwargs["name"] if kwargs.get("name") else None
        self._model = model
        self._api_key = api_key
        self._temperature=temperature
        self._max_tokens=max_tokens
        self._top_p=top_p
        self._frequency_penalty=frequency_penalty
        self._presence_penalty=presence_penalty
        self.ans_word = ans_word
        """
        for _k, _v in kwargs.items():
            self._k = _v
        """

    def set_model(self, new_model) -> None:
        old_model = self._model
        self._model = new_model
        logger.debug(f"Changed model from {old_model} to {new_model}")
        return None

    def set_api_key(self, new_api_key) -> None:
        old_api_key = self._api_key
        self._api_key = new_api_key
        logger.debug(f"Changed api key from {old_api_key} to {new_api_key}")
        return None

    def set_temperature(self, new_temperature) -> None:
        old_temperature = self._temperature
        self._temperature = new_temperature
        logger.debug(f"Changed temperature from {old_temperature} to {new_temperature}")
        return None

    def set_max_tokens(self, new_max_tokens) -> None:
        old_max_tokens = self._max_tokens
        self._max_tokens = new_max_tokens
        logger.debug(f"Changed max tokens from {old_max_tokens} to {new_max_tokens}")
        return None

    def set_top_p(self, new_top_p) -> None:
        old_top_p = self._top_p
        self._top_p = new_top_p
        logger.debug(f"Changed top p from {old_top_p} to {new_top_p}")
        return None

    def set_frequency_penalty(self, new_frequency_penalty) -> None:
        old_frequency_penalty = self._frequency_penalty
        self._frequency_penalty = new_frequency_penalty
        logger.debug(f"Changed frequency penalty from {old_frequency_penalty} to {new_frequency_penalty}")
        return None

    def set_presence_penalty(self, new_presence_penalty) -> None:
        old_presence_penalty = self._presence_penalty
        self._presence_penalty = new_presence_penalty
        logger.debug(f"Changed presence penalty from {old_presence_penalty} to {new_presence_penalty}")
        return None

    def __call__(self, prompt : str) -> tuple:
        openai.api = self._api_key
        client = OpenAI()
        response = client.completions.create(
            model=self._model,
            prompt=prompt,
            temperature=self._temperature,
            max_tokens=self._max_tokens,
            top_p=self._top_p,
            frequency_penalty=self._frequency_penalty,
            presence_penalty=self._presence_penalty,
            logprobs=1
        )
        """
        :param prompt: A input text of construction accident scenario.
        :type prompt: str
        :return: A tuple containing the result.
        :rtype: tuple
        """
        return response.choices[0].text, exp(list(response.choices[0].logprobs.top_logprobs[0].values()).pop())


class MCS():
    __db = [
        {
            "index" : 0,
            "name" : "trip",
            "ans_word" : "넘어짐",
            "model" : "ft:babbage-002:personal::8nglTQdt"
        }, {
            "index" : 1,
            "name" : "cut",
            "ans_word" : "절단, 베임, 찔림",
            "model" : "ft:babbage-002:personal::8njvRyLA"
        }, {
            "index" : 2,
            "name" : "hit",
            "ans_word" : "부딪힘",
            "model" : "ft:babbage-002:personal::8nkoSL6W"
        }, {
            "index" : 3,
            "name" : "fall",
            "ans_word" : "추락",
            "model" : "ft:babbage-002:personal::8nkzOBRX"
        }, {
            "index" : 4,
            "name" : "caught-in-between",
            "ans_word" : "협착",
            "model" : "ft:babbage-002:personal::8nlAcGXN"
        }, {
            "index" : 5,
            "name" : "etc",
            "ans_word" : "기타",
            "model" : "ft:babbage-002:personal::8o5ufMJV"
        }
    ]

    def __init__(self, db = None) -> None:
        self.db = db if db else self.__db

    def _find_model_key_by_name(self, name) -> Union[str, None]:
        for item in self.db:
            if item['name'] == name:
                return item['model']
        return None

    def _is_rst_true(self, answer : dict) -> bool:
        if answer["rst"][0] == answer["word"][0]: 
            return True
        return False

    def run(self, prompt, model) -> dict:      
        """
        :param prompt: A input text of construction accident scenario.
        :type prompt: str
        :param model: A model of interest.
        :type model: Any
        :return: A dictionary containing the result.
        :rtype: dict
        """
        _a, _p = model(prompt=prompt)
        return {
            "name" : model.name,
            "word" : model.ans_word,
            "rst" : _a,
            "prob" : _p
        }

    def _parallel_map_helper(self, to_map : Any, target_map : list) -> list:
        L = []
        for target in target_map:
            L.append((to_map, target))
        return L

    def _parallel_run(self, prompt : str, models : list) -> list:
        pool = Pool(len(models))
        answer_list = pool.starmap(self.run, self._parallel_map_helper(prompt, models))
        pool.close()
        pool.join()
        return answer_list

    def __call__(self, prompt : str) -> tuple:
        """
        :param prompt: A input text of construction accident scenario.
        :type prompt: str
        :return: A tuple containing the result.
        :rtype: tuple
        """
        model_list = []
        for item in self.db: 
            model_list.append(ModelHandler(self._find_model_key_by_name(item["name"]), name=item["name"], ans_word = item["ans_word"]))
        answer_list = self._parallel_run(prompt, model_list)
        for answer in answer_list: 
            answer["correct"] = True if self._is_rst_true(answer) else False
        return sorted(answer_list, key=lambda x: x["prob"], reverse=True)


class ICCEPM(ModelHandler):
    __class_dict = {
        "trip" : "넘어짐",
        "fall" : "떨어짐",
        "collisions" : "깔림",
        "caught-in-between" : "끼임",
        "cut" : "베임",
        "hit" : "부딪힘",
        "impact" : "물체에 맞음",
        "piercing" : "찔림",
        "disease" : "질병"
    }

    def __init__(self, model, temperature=.0, ):
        super().__init__(model=model)
        self._classes = list(self.__class_dict.values())

    def __call__(self, prompt : str) -> tuple:
        openai.api = self._api_key
        client = OpenAI()
        response = client.completions.create(
            model=self._model,
            prompt=prompt,
            temperature=self._temperature,
            max_tokens=self._max_tokens,
            top_p=self._top_p,
            frequency_penalty=self._frequency_penalty,
            presence_penalty=self._presence_penalty,
            logprobs=9
        )
        """
        :param prompt: A input text of construction accident scenario.
        :type prompt: str
        :return: A tuple containing the result.
        :rtype: tuple
        """
        return response.choices[0], response.choices[0].text, exp(list(response.choices[0].logprobs.top_logprobs[0].values()).pop())

    def eva(self, prompt: str) -> str:
        """
        A method to support evaluation of the model.

        :param prompt: A input text of construction accident scenario.
        :type prompt: str
        :return: A string containing the result.
        :rtype: str
        """
        full, text, prob = self.__call__(prompt)
        for idx, cls in enumerate(self._classes):
            if text.startswith(cls[0]): return cls
            elif idx+1 < len(self._classes): continue
            else:
                rdm_rst = choice(self._classes)
                logger.debug(f"No match. Returning random output. ({rdm_rst})")
                return rdm_rst

def main1():
    model = MCS()
    prompt = input("사고 사례: ") if not args.scenario else args.scenario
    A = model(prompt)
    logger.info(f"[Input] {prompt}")
    correct_cnt = 0
    for elem in A:
        correct_cnt += 1 if elem.get("correct") else 0
    cnt = 0
    for idx, elem in enumerate(A): 
        if args.print_log == "correct" and elem.get("correct"): 
            cnt += 1
            logger.info(f'[Output] {elem.get("name")} : {round(elem.get("prob")*100, 2)}%\n' if cnt >= correct_cnt else f'[Output] {elem.get("name")} : {round(elem.get("prob")*100, 2)}%')
        elif args.print_log == "all":
            for k, v in elem.items(): 
                logger.info("[Output] " + k + " : " + str(v) + "\n" if idx + 1 >= len(A) and k == "correct" else "[Output] " + k + " : " + str(v))

def main2():
    model = ICCEPM("ft:babbage-002:personal::8wPXwLA5")
    prompt = input("사고 사례: ") if not args.scenario else args.scenario
    logger.info(f"[Input] {prompt}")
    full, compl, prob = model(prompt+"\n\n###\n\n")
    logger.info(f"[Output] {compl}, {prob}")
    logger.debug(f"len(full): {len(full)}")
    logger.debug(f"len(text): {len(full[0].text)}")
    logger.debug(f"len(text_offset): {len(full[0].logprobs.text_offset)}")
    tmp_md = f"# ICCEPM 2024\n\n```\n{prompt}\n```\n\n```\n{full[0].text}\n```\n\n| text offset | token logprobs (probs) | tokens | top logprobs | text |\n| :--: | :--: | :--: | :--: | :--: |\n"
    logger.debug(str(full[0].logprobs.text_offset[-1] - full[0].logprobs.text_offset[0]))
    str_idx = 0
    for i in range(len(full[0].logprobs.text_offset)):
        tmp_md += f"| {full[0].logprobs.text_offset[i]} | {full[0].logprobs.token_logprobs[i]} ({round(exp(full[0].logprobs.token_logprobs[i]), 2)}%) | {full[0].logprobs.tokens[i]} | {full[0].logprobs.top_logprobs[i]} | <b>{full[0].text[full[0].logprobs.text_offset[i]-full[0].logprobs.text_offset[0]]}</b> |\n"
        str_idx += 1
    with open("out/tmp.md", 'w', encoding="utf-8") as f:
        f.writelines(tmp_md)
    logger.debug("MD file saved successfully. (out/tmp.md)")


if __name__ == "__main__":
    main2()
# %%
