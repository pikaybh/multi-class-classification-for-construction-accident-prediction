# %%
try: from .osutil import split_dir_file_ext
except: from .tmp import split_dir_file_ext

import logging
import os
import sys
import inspect

class Console():
    _levels = {
        "debug" : logging.DEBUG,
        "info" : logging.INFO,
        "warning" : logging.WARNING,
        "error" : logging.ERROR,
        "critical" : logging.CRITICAL
    }

    def __init__(self, 
    filename : str = sys.argv[0], 
    level : str = "info", 
    directory : str = 'logs'):
        """
        @param level: `logging`의 level을 설정 (default: "info")
            - "debug" : logging.DEBUG,
            - "info" : logging.INFO,
            - "warning" : logging.WARNING,
            - "error" : logging.ERROR,
            - "critical" : logging.CRITICAL
        @param directory: Working directory, or save location. (default: "logs")
        @type filename: str
        @type level: str
        @type directory: str
        @return: None
        """
        self.filename = split_dir_file_ext(filename)[1]
        self.file = os.path.join(directory, self.filename)
        try:
            self.level = self._levels[level]
        except Exception as e:
            logging.error(e)
            raise e
        # logger
        self._logger = logging.getLogger(__name__)
        self._logger.setLevel(logging.DEBUG)
        # formatter
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        # 콘솔 핸들러 설정
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(self.level)
        console_handler.setFormatter(formatter)
        self._logger.addHandler(console_handler)
        # 파일 핸들러 설정
        file_handler = logging.FileHandler(f"{self.file}.log", encoding="utf-8")
        file_handler.setLevel(self.level)
        file_handler.setFormatter(formatter)
        self._logger.addHandler(file_handler)
        # Session log
        self._session_log = []
        # History log
        self._history = []

    def __call__(self) -> logging:
        return self._logger

    def _formatter(self, message : str) -> str:
        return f'{self.filename}:{inspect.stack()[1].lineno} - {message}'

    def debug(self, message : str = "") -> None:
        """
        @param message: Debbuging message.
        @type message: str
        @return: None
        """
        format_msg = self._formatter(message)
        self._logger.debug(format_msg)
        self._session_log.append(format_msg)
        return None

    def info(self, message : str = "") -> None:
        """
        @param message: Inform message.
        @type message: str
        @return: None
        """
        format_msg = self._formatter(message)
        self._logger.info(format_msg)
        self._session_log.append(format_msg)
        return None

    def warning(self, message : str = "") -> None:
        """
        @param message: Warning message.
        @type message: str
        @return: None
        """
        format_msg = self._formatter(message)
        self._logger.warning(format_msg)
        self._session_log.append(format_msg)
        return None

    def error(self, message : str = "") -> None:
        """
        @param message: Error message.
        @type message: str
        @return: None
        """
        format_msg = self._formatter(message)
        self._logger.error(format_msg)
        self._session_log.append(format_msg)
        return None

    def critical(self, message : str = "") -> None:
        """
        @param message: Critical message.
        @type message: str
        @return: None
        """
        format_msg = self._formatter(message)
        self._logger.critical(format_msg)
        self._session_log.append(format_msg)
        return None

    def change_logger(self, new_logger : logging) -> None:
        """
        @return: None
        """
        self._logger.info(f"Chaging the logger.")
        self._logger = new_logger
        self._logger.info(f"Logger changed successfully.")
        return None

    def get_a_log(self, index : int = -1) -> str:
        """
        @param index: Index of interest.
        @type index: int
        @return str
        """
        if not isinstance(index, int): 
            _error_msg = "Index should be integer."
            self.error(_error_msg)
            raise SyntaxError(_error_msg)
        return self._session_log[index]

    def get_log(self, index : int = 0) -> list:
        """
        @param index: Log starting index.
        @type index: int
        @return str
        """
        L = []
        if not index < len(self._session_log):
            L += self.get_log(index+1)
        return L + self.get_a_log(index)

    def _read_histroy(self) -> None:
        with open(f"{self.file}.log") as hist:
            L = hist.readlines()
        self._history = L
        self.info("Log history has get to read.")

    def get_a_history(self, index : int = -1) -> str:
        """
        @param index: Index of interest.
        @type index: int
        @return str
        """
        if not isinstance(index, int): 
            _error_msg = "Index should be integer."
            self.error(_error_msg)
            raise SyntaxError(_error_msg)
        if not self._history: self._read_histroy()
        return self._history[index]

    def get_history(self, index : int = 0) -> list:
        """
        @param index: History starting index.
        @type index: int
        @return str
        """
        L = []
        if not index < len(self._history):
            L += self.get_log(index+1)
        return L + self.get_a_history(index)


if __name__ == "__main__":
    log = Console() 
    log.info("hello")
# %%
