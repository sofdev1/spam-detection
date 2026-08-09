"""
Custom exception class that wraps the original exception with the
filename and line number it occurred on, for easier debugging in logs.
"""
import sys


def error_message_detail(error, error_detail):
    _, _, exc_tb = error_detail.exc_info()
    file_name = exc_tb.tb_frame.f_code.co_filename
    line_number = exc_tb.tb_lineno
    return (
        f"Error occurred in python script [{file_name}] "
        f"line number [{line_number}] error message [{str(error)}]"
    )


class SpamDetectionException(Exception):
    def __init__(self, error_message, error_detail):
        super().__init__(error_message)
        self.error_message = error_message_detail(error_message, error_detail)

    def __str__(self):
        return self.error_message
