# %%
from utils.osutil import get_time, check_file_name, drop_outliers
from utils.console import Console
from multi_class_classification import MCS, ICCEPM

import matplotlib.pyplot as plt
import pandas as pd
from random import randrange
from tqdm import tqdm
import os
from time import time
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report, accuracy_score
from math import exp
import pickle
from dotenv import load_dotenv
import json


logger = Console(level='info')
plt.rcParams['font.family'] = 'Malgun Gothic'
# 개인 api key
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
MAX_TRIAL = 100
NUM = 25
target_file = "data/test/CSI_gpt-3.5-turbo_1000_samples_frac_8_to_2_rest.xlsx"
"""
target_columns = ["인적사고"]  # , "물적사고"]
with open ("del_target.json", 'r', encoding='utf-8') as f:
    del_targets = json.load(f)
"""
target_column = "completion"
target_column = "인적사고"
df = pd.read_excel(target_file)
"""
df = pd.read_excel(target_file)
df.rename(columns = {"사고경위" : "prompt", target_column : "completion"}, inplace = True)
df = df[["prompt", "completion"]]
df = drop_outliers(df, 'completion', list(del_targets[target_column]))
df.to_excel("data/CSIPernalPropertyDamage.xlsx", index=False)
"""
# Confusion Matrix를 PNG 파일로 저장하는 함수
def save_confusion_matrix(cm, classes, file_path):
    fig, ax = plt.subplots(figsize=(8, 6))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=classes)
    disp.plot(ax=ax)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(file_path)
    plt.close()

def main_MCS():
    model = MCS()
    X_test, y_test, predictions = [], [], []
    start_time = time()
    completion_uniq = df['completion'].unique()
    for compl in completion_uniq:
        L = list(df[(df['completion'] == compl)].index)
        for i in tqdm(range(NUM), desc=compl):
            X_test.append(df['prompt'].iloc[L[i]])  # str(df.iloc[i, df.columns.get_loc('사고경위')]))
            y_test.append(df['completion'].iloc[L[i]])  # df.iloc[i,  df.columns.get_loc('인적사고')])
            candidate = model(df["prompt"][L[i]])
            for idx, output in enumerate(candidate):
                if output.get("correct"):
                    predictions.append(output.get("word"))
                    break
                elif idx+1 >= len(candidate): predictions.append("기타") 
                else: continue
    logger.debug(f"x test: {X_test}\ny test: {y_test}\npred: {predictions}")
    # Op time 
    op_time = time() - start_time
    classes = sorted(completion_uniq)
    # cm
    cm = confusion_matrix(y_test, predictions, labels=classes)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=classes)

    disp.plot()

    plt.xticks(rotation=45)
    plt.show()

    logger.info(f"""
    model : MCS
    accuracy : {(accuracy_score(y_test, predictions))*100:.3f}%
    classification report : \n{classification_report(y_test, predictions)}""")
    # pickle
    pickle_dict = {"X_test" : X_test,
                    "y_test" : y_test,
                    "predictions" : predictions,
                    "cm" : cm,
                    "acc" : f"{(accuracy_score(y_test, predictions))*100:.3f}%",
                    "classifiction_report" : f"\n{classification_report(y_test, predictions)}\n"}
    pickling_file_name = f"{target_file.split('/')[-1].split('.')[0]}_MCS_{target_column}_{len(completion_uniq)}_classes_6"
    # confusion matrix 저장
    save_confusion_matrix(cm, classes, check_file_name(f"out/{pickling_file_name}.png"))
    with open(check_file_name(f"pkls/{pickling_file_name}.pkl"), "wb") as f:
        pickle.dump(pickle_dict, f, protocol=pickle.HIGHEST_PROTOCOL)
    logger.info("Pickled succefully.")

def main_ICCEPM():
    model = ICCEPM("ft:babbage-002:personal::8wPXwLA5")
    X_test, y_test, predictions = [], [], []
    start_time = time()
    completion_uniq = df['인적사고'].unique()
    for compl in completion_uniq:
        L = list(df[(df['인적사고'] == compl)].index)
        for i in tqdm(range(len(L)), desc=compl):
            X_test.append(df['사고 전'].iloc[L[i]])  # str(df.iloc[i, df.columns.get_loc('사고경위')]))
            y_test.append(df['인적사고'].iloc[L[i]])  # df.iloc[i,  df.columns.get_loc('인적사고')])
            predictions.append(model.eva(df["사고 전"][L[i]]))
    logger.debug(f"x test: {X_test}\ny test: {y_test}\npred: {predictions}")
    # Op time 
    op_time = time() - start_time
    classes = sorted(completion_uniq)
    # cm
    cm = confusion_matrix(y_test, predictions, labels=classes)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=classes)

    disp.plot()

    plt.xticks(rotation=45)
    plt.show()

    logger.info(f"""
    model : ICCEPM
    accuracy : {(accuracy_score(y_test, predictions))*100:.3f}%
    classification report : \n{classification_report(y_test, predictions)}""")
    # pickle
    pickle_dict = {"X_test" : X_test,
                    "y_test" : y_test,
                    "predictions" : predictions,
                    "cm" : cm,
                    "acc" : f"{(accuracy_score(y_test, predictions))*100:.3f}%",
                    "classifiction_report" : f"\n{classification_report(y_test, predictions)}\n"}
    pickling_file_name = f"{target_file.split('/')[-1].split('.')[0]}_ICCEPM_{target_column}_{len(completion_uniq)}_classes_{str(len(completion_uniq))}_pre_accident"
    # confusion matrix 저장
    save_confusion_matrix(cm, classes, check_file_name(f"out/{pickling_file_name}.png"))
    with open(check_file_name(f"pkls/{pickling_file_name}.pkl"), "wb") as f:
        pickle.dump(pickle_dict, f, protocol=pickle.HIGHEST_PROTOCOL)
    logger.info("Pickled succefully.")


if __name__ == "__main__":
    main_ICCEPM()
# %%
