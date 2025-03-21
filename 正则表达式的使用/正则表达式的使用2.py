content = """
    白日依2021/05/26山尽，黄河入2021.05.27海流
    欲穷05-28-2020千里目，更上5/29/2020一层楼
"""

import re

content = re.sub(r"(\d{4})/(\d{2})/(\d{2})", r"\1-\2-\3", content)
print(content)

content = re.sub(r"(\d{4})\.(\d{2})\.(\d{2})", r"\1-\2-\3", content)
print(content)

content = re.sub(r"(\d{2})-(\d{2})-(\d{4})", r"\3-\1-\2", content)
print(content)

content = re.sub(r"(\d{1})/(\d{2})/(\d{4})", r"\3-0\1-\2", content)
print(content)