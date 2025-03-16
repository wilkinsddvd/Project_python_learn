from urllib.parse import urlparse

result = urlparse('https://www.baidu.com/index.html#comment', allow_fragments=False)
print(result)
