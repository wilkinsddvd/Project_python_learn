from urllib.request import HTTPPasswordMgrWithDefaultRealm, HTTPBasicAuthHandler, build_opener
from urllib.error import URLError

username = 'admin'
password = 'admin'
url = 'https://ssr3.scrape/center/'

p = HTTPPasswordMgrWithDefaultRealm()
p.add_password(None, url, username, password)
auto_handler = HTTPBasicAuthHandler(p)
opener = build_opener(auto_handler)

try:
    result = opener.open(url)
    html = result.read().decode('utf-8')
    print(html)
except URLError as e:
    print(e.reason)

# 结果为[Errno 11001] getaddrinfo failed 找不到主机
# 源代码疑似需要更新
