import http.cookiejar, urllib.request

cookie = http.cookiejar.CookieJar()
handler = urllib.request.HTTPCookieProcessor(cookie)
opener = urllib.request.build_opener(handler)
response = opener.open('https://www.baidu.com')
for item in cookie:
    print(item.name + "=" + item.value)

# result:
# BAIDUID=82E102A829F16280F8AD1D5CD5726199:FG=1
# BIDUPSID=82E102A829F16280029D943B498C2735
# PSTM=1707824755
# BD_NOT_HTTPS=1