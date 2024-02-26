# %%
from utils import  (get_time, 
                    read_file, 
                    train_valid_split_save, 
                    dir_file_splitter, 
                    name_ext_splitter,
                    mkdir_if_dir_not_exists,
                    check_file_name)

import pandas as pd
import openai
from openai import OpenAI
import os
import json
from datetime import datetime
from dotenv import load_dotenv
from typing import Union
from collections import OrderedDict


load_dotenv()
FILTER = "협착"
SYSTEM_MSG = "입력되는 건설현장 데이터로부터 발생할 수 있는 사고를 차근차근 생각하여 추론합니다."
file = "data/train/trimmed5_train.xlsx"
style = "completion"

df = read_file(file)[["prompt", "completion"]]
if FILTER: 
    # 'completion' 열에서 'FILTER'이라는 값을 갖는 행들을 필터링합니다.
    filter_df = df[df['completion'] == FILTER]
    # 'completion' 열에서 'FILTER'이 아닌 행들을 필터링하고, 이 중 랜덤하게 900개를 선택합니다.
    not_filter_df = df[df['completion'] != FILTER].sample(n=900)
    not_filter_df['completion'] = '그외'
    # 두 DataFrame을 합칩니다.
    combined_df = pd.concat([filter_df, not_filter_df])# 합친 DataFrame을 랜덤하게 섞습니다.
    df = combined_df.sample(frac=1).reset_index(drop=True)

json_dir = name_ext_splitter(file)[0]
mkdir_if_dir_not_exists(json_dir + '/')
if FILTER: json_dir += f" ({FILTER})"
json_file = check_file_name(json_dir + ".jsonl")
df.to_json(json_file, orient='records', force_ascii=False, lines=True)


def conclusion_merge(situation : str, CoT : str, conclusion : str) -> str:
    abstract = CoT.replace("이 과정은 사고 발생의 원인부터 결과까지 순차적으로 분석하고 연결하는 논리적 사고 과정을 보여줍니다.",
    f"5. **사고 경위**: {situation}{conclusion}\n   - 사고 전: {situation}\n   - 사고 후: {conclusion}")
    return abstract

def show_result_file(json_file : str, style : str = "chat", print_status : bool = True) -> str:
    if style == "chat":
        with open(json_file) as f:
            new_data = [json.loads(line) for line in f]
            if print_status:
                for data in new_data:
                    print(f"{list(data.keys())[0]}:")
                    for msg in data["messages"]:
                        print('\t' + f'{msg["role"]}:')
                        print('\t\t' + f'{msg["content"]}')
                print(f"{len(new_data)} lines.")
            return new_data
    elif style == "completion":
        with open(json_file) as f:
            new_data = [json.loads(line) for line in f]
            if print_status:
                for data in new_data:
                    print(f"{list(data.keys())[0]}:")
                    for msg in data["messages"]:
                        print('\t' + f'{msg["role"]}:')
                        print('\t\t' + f'{msg["content"]}')
                print(f"{len(lines)} lines.")
            return new_data
    else: raise ValueError("The style parameter should one of 'chat' or 'completion'.")

def main(style : str = "chat"):
    global SYSTEM_MSG, json_file

    if style == "chat":
        with open(json_file) as f:
            lines = [json.loads(line) for line in f]
        new_lines = []
        for line in lines:
            data = OrderedDict()
            data["messages"] = [{"role" : "system", "content" : SYSTEM_MSG},
                {"role" : "user", "content" : line["사고 전"]},
                {"role" : "assistant", "content" : conclusion_merge(line["사고 전"], line["CoT"], line["사고 후"])}]
            new_lines.append(data)
        with open(json_file, "w", encoding="utf-8") as f:
            for line in new_lines:
                json.dump(line, f, ensure_ascii=False) # ensure_ascii로 한글이 깨지지 않게 저장
                f.write("\n") # json을 쓰는 것과 같지만, 여러 줄을 써주는 것이므로 "\n"을 붙여준다.
    elif style == "completion":
        with open(json_file) as f:
            lines = [json.loads(line) for line in f]
        new_lines = []
        return new_lines.append({"prompt" : line["prompt"], "completion" : line["completion"]} for line in lines)
        with open(json_file, "w", encoding="utf-8") as f:
            for line in new_lines:
                json.dump(line, f, ensure_ascii=False) # ensure_ascii로 한글이 깨지지 않게 저장
                f.write("\n") # json을 쓰는 것과 같지만, 여러 줄을 써주는 것이므로 "\n"을 붙여준다.
    show_result_file(json_file, style=style)


if __name__ == "__main__":
    main(style)
# %%
