import urllib.request, http.cookiejar

cookie = http.cookiejar.LWPCookieJar()
cookie.load('12cookie.txt', ignore_discard=True, ignore_expires=True)
handler = urllib.request.HTTPCookieProcessor(cookie)
opener = urllib.request.build_opener(handler)
response = opener.open('https://www.baidu.com')
print(response.read().decode('utf-8'))

# FileNotFoundError: [Errno 2] No such file or directory: '12cookie.txt'