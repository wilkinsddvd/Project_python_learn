from urllib.request import urlopen

myURL = urlopen("http://ip.json-json.com/")
print(myURL.read())