import requests
from bs4 import BeautifulSoup

# 发送 HTTP 请求，获取网页内容
url = 'https://example.com'  # 替换成要爬取的网页地址
response = requests.get(url)
html_content = response.text

# 解析 HTML 内容
soup = BeautifulSoup(html_content, 'html.parser')

# 提取数据
# 示例1：提取网页标题
title = soup.title.string
print("网页标题：", title)

# 示例2：提取所有的链接
links = soup.find_all('a')
for link in links:
    href = link.get('href')
    print("链接：", href)

# 示例3：提取特定的数据
data = soup.find('div', class_='class_name')
if data:
    print("提取的数据：", data.text)
else:
    print("未找到指定的数据")