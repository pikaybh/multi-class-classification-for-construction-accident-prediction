# %%
from .console import Console

import os
import pandas as pd
import warnings
import time
from typing import Union
from random import random
from sklearn.model_selection import train_test_split


logger = Console()

random_state = 12883823

def get_time() -> str:
    return time.strftime('%Y-%m-%d %X', time.localtime())

def dir_file_splitter(location : str) -> Union[tuple, None]:
    if not '/' in location: logger.warning("Nothing to split.")
    elif location[-1] == '/': logger.warning("There's no file in given location")
    else: 
        file = location.split('/')[-1]
        return location[: -len(file)], file
    return None

def mkdir_if_dir_not_exists(location : str = "") -> None:
    if not '/' in location: logger.warning("No directory to make.")
    directory = ""
    if location[-1] == '/': directory = location
    else: 
        file = location.split('/')[-1]
        directory = location[: -len(file)]
    if not os.path.exists(directory):
        os.makedirs(directory)
        logger.debug(f'A directory is made. (location: "{directory}")')
    logger.debug(f'"{directory}" already exists.')
    return None

def name_ext_splitter(file : str) -> Union[str, tuple]:
    if not '.' in file:
        logger.warning(f'"{file}" has no extension.')
        return file, ''
    elif file.startswith('.') or file.endswith('.'):
        logger.warnings(f'The system can not extract file name from "{file}".')
        return file, ''
    else:
        ext = file.split('.')[-1]
        name = file[ : -(len(ext) + 1)]
        return name, ext

def check_file_name(file : str = "data.csv", num : int = 0) -> str:
    name, ext = name_ext_splitter(file)
    if num != 0:
        file = f"{name} ({num}).{ext}" if num == 1 else f"{name[ : - (len(str(num)) + 3)]} ({num}).{ext}"
    if os.path.isfile(file): return check_file_name(file, num+1)
    else: return file

def gcd(a, b):
    """
    Greatest Common Divisor
    """
    while b > 0:
        a, b = b, a % b
    return a

def drop_outliers(df : pd.DataFrame, column : str, targets : list, threshold : int = 100) -> pd.DataFrame:
    df_count = df[column].value_counts()
    for v in df[column].unique():
        if (df_count.get(v, 0) < threshold) or (v in targets):
            df.drop(df[(df[column] == v)].index, inplace=True)
    return df
    
def read_file(file : str) -> None:
    _, ext = name_ext_splitter(file)
    try: return pd.read_excel(file) if ext == "xlsx" else pd.to_csv(file)
    except Exception as e: 
        logger.error(e)
        raise ValueError(f"Can't read a file with {ext} extension.")

def to_file(df : pd.DataFrame, file : str) -> None:
    _, ext = name_ext_splitter(file)
    if ext == "xlsx": df.to_excel(check_file_name(file), index=False)
    elif ext == "csv": df.to_csv(check_file_name(file), indx=False)
    else: raise ValueError(f"Can't save a file with {ext} extension.")
    logger.debug(f'File saved successfully. ("{file}")')
    return None

def __name_ratio(ratio : float, type="str") -> Union[str, tuple]:
    len_ratio = int(len(str(ratio)) - 2)
    big, small = round(ratio*(10^len_ratio-1)), round((1-ratio)*(10^len_ratio-1))
    ratio_gcd = gcd(big, small)
    if type == "str": return f"{str(big // ratio_gcd)}_to_{str(small // ratio_gcd)}"
    elif type == "int": return big // ratio_gcd, small // ratio_gcd
    else: raise ValueError('The type must "str" or "int"')

def train_test_split_save(df : pd.DataFrame = None, file : str = "", ratio : int = .8, save_file : str = "") -> None:
    if (df and file) or (not df and not file): raise ValueError("file이나 df 둘 중 하나 혹은 하나는 꼭 필요함.")
    if not df: df = read_file(file)
    elif not save_file: raise ValueError("file이나 save_file 둘 중 하나 혹은 하나는 꼭 필요함.")
    if '/' in file: file = file.split('/')[-1]
    if not save_file: save_file = file
    name, ext = name_ext_splitter(save_file)
    ratio_name = __name_ratio(ratio)
    part_file, rest_file = f"data/train/{name}_frac_{ratio_name}_part.{ext}", f"data/test/{name}_frac_{ratio_name}_rest.{ext}"
    for i in [part_file, rest_file]:
        mkdir_if_dir_not_exists(i, print_status = False)
    part = df.sample(frac = ratio)
    to_file(df=part, file=part_file, print_status = True)
    to_file(df=df.drop(part.index), file=rest_file)

def train_valid_split_save(df : pd.DataFrame = None, 
file : str = "", 
ratio : int = .8, 
save_file : str = "", 
return_name : bool = False) -> tuple:
    if (df and file) or (not df and not file): raise ValueError("file이나 df 둘 중 하나 혹은 하나는 꼭 필요함.")
    if not df: df = read_file(file)
    elif not save_file: raise ValueError("file이나 save_file 둘 중 하나 혹은 하나는 꼭 필요함.")
    if '/' in file: file = file.split('/')[-1]
    if not save_file: save_file = file
    name, ext = name_ext_splitter(save_file)
    train_ratio, valid_ratio = __name_ratio(ratio, type="int")
    part_file, rest_file = f"data/train/{name}/train_({valid_ratio}).{ext}", f"data/train/test/{name}/valid_({valid_ratio}).{ext}"
    for i in [part_file, rest_file]:
        mkdir_if_dir_not_exists(i, print_status = False)
    part = df.sample(frac = ratio)
    rest = df.drop(part.index)
    to_file(df=part, file=part_file, print_status = True)
    to_file(df=rest, file=rest_file)
    return part, rest, part_file, rest_file if return_name else part, rest

def clean_text_file(file : str, target : str = '@') -> None:
    with open(file, 'r', encoding="UTF-8") as f:
        lines = f.readlines()
    new_lines = []
    for line in lines:
        new_lines.append(line[: line.index(target)] + '\n' if target in line else line)
    with open(file, 'w', encoding="UTF-8") as f:
        for line in new_lines:
            f.writelines(line)

# function nfc
def pass_file(file : str, files : list) -> bool:
    """List of files that just pass from searching..."""
    for i in files:
        if i in file:
            return True
    return False

def search_files(root_dir : str, _ext : str, exception_files : list) -> list:
    """Searching files with specitic extension in the given directory tree"""
    L = list()
    for (path, dir, files) in os.walk(root_dir):
        for filename in files:
            if pass_file(filename, exception_files):
                continue
            ext = os.path.splitext(filename)[-1]
            if ext == _ext and '$' not in filename:
                _file = f"{path}/{filename}"
                if "//" in _file:
                    L.append(f"{path}{filename}")
                else:
                    L.append(_file)
    return L

def split_dir_file_ext(full_name : str) -> tuple:
    """
    @param full_name: full name of the file.
    @type full_name: str
    @return: tuple
    """
    _dir, _file = os.path.split(full_name)
    _name, _ext = os.path.splitext(_file)
    return _dir, _name, _ext

def new_file_cnt(f : str, cnt : int) -> str:
    """It just add numbers to result file, it prevents some error..."""
    name, ext = f.split('.')
    # tmp = name + '.' + ext if cnt == 0 else name + '(' + str(cnt) + ')' + '.' + ext
    tmp = name + '.' + ext if cnt == 0 else name + str(cnt) + '.' + ext
    if os.path.isfile(tmp):
        cnt += 1
        return new_file_cnt(f, cnt)
    else:
        return tmp

if __name__ == "__main__":
    print(split_dir_file_ext("hello/world/.py.env"))

# %%
