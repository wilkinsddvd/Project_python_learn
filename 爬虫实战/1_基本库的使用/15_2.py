from urllib.parse import urlparse

result = urlparse('https://www.baidu.com/index.html;user?id=5#comment', scheme='https')

# result = urlparse('www.baidu.com/index.html;user?id=5#comment', scheme='https')

# 输出不一样 netloc的输出不一样
print(result)