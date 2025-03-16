import webbrowser
import requests
import sys

res = requests.get('https://marcobonzanini.com/category/big-data/')
res.raise_for_status()
