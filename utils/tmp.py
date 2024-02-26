import os


def split_dir_file_ext(full_name : str) -> tuple:
    """
    @param full_name: full name of the file.
    @type full_name: str
    @return: tuple
    """
    _dir, _file = os.path.split(full_name)
    _name, _ext = os.path.splitext(_file)
    return _dir, _name, _ext