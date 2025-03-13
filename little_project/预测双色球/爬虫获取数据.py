import requests
from bs4 import BeautifulSoup
import pandas as pd

# 定义目标URL
url = "https://www.zhcw.com/kjxx/ssq/"  # 将此替换为实际提供双色球历史数据的URL

# 发送HTTP请求
response = requests.get(url)

# 检查响应状态码
if response.status_code != 200:
    print(f"请求失败，状态码：{response.status_code}")
    exit()

# 解析HTML内容
soup = BeautifulSoup(response.content, 'html.parser')

# 查找表格元素
table = soup.find('table')

# 检查表格是否存在
if table is None:
    print("未找到表格，请检查URL和HTML结构。")
else:
    # 提取表格数据
    data = []
    headers = []

    # 找到表头行
    header_row = table.find('thead').find_all('tr')
    for tr in header_row:
        ths = tr.find_all('th')
        headers.extend([th.text.strip() for th in ths])

    # 检查是否成功提取到表头
    if not headers:
        print("未找到表头，请检查表格结构。")
    else:
        print(f"表头: {headers}")

    # 提取表格内容
    for row in table.find('tbody').find_all('tr'):
        cells = row.find_all('td')
        if len(cells) == len(headers):
            data.append([cell.text.strip() for cell in cells])
        else:
            print(f"跳过行：{[cell.text.strip() for cell in cells]}，因为列数不匹配。")

    # 创建DataFrame
    if data:
        df = pd.DataFrame(data, columns=headers)
        # 保存为CSV文件
        df.to_csv('ssq_history.csv', index=False)
        print("数据已保存到 ssq_history.csv 文件中")
    else:
        print("未提取到有效数据，请检查表格结构。")


# import requests
# from bs4 import BeautifulSoup
# import pandas as pd
#
# # 定义目标URL
# url = "https://www.zhcw.com/kjxx/ssq/"  # 将此替换为实际提供双色球历史数据的URL
#
# # 发送HTTP请求
# response = requests.get(url)
#
# # 检查响应状态码
# if response.status_code != 200:
#     print(f"请求失败，状态码：{response.status_code}")
#     exit()
#
# # 打印响应内容以调试
# print(response.content.decode('utf-8'))  # 使用decode将bytes转换为str
#
# # 解析HTML内容
# soup = BeautifulSoup(response.content, 'html.parser')
#
# # 查找表格元素
# table = soup.find('table')
#
# # 检查表格是否存在
# if table is None:
#     print("未找到表格，请检查URL和HTML结构。")
# else:
#     # 提取表格数据
#     data = []
#     headers = [header.text for header in table.find_all('th')]
#
#     for row in table.find_all('tr')[1:]:
#         cells = row.find_all('td')
#         data.append([cell.text for cell in cells])
#
#     # 创建DataFrame
#     df = pd.DataFrame(data, columns=headers)
#
#     # 保存为CSV文件
#     df.to_csv('ssq_history.csv', index=False)
#
#     print("数据已保存到 ssq_history.csv 文件中")
#

# 找不到数据

# import requests
# from bs4 import BeautifulSoup
# import pandas as pd
#
# # 定义目标URL
# url = "https://www.zhcw.com/kjxx/ssq/"  # 将此替换为实际提供双色球历史数据的URL
#
# # 发送HTTP请求
# response = requests.get(url)
#
# # 打印响应内容以调试
# print(response.content)
#
# # 解析HTML内容
# soup = BeautifulSoup(response.content, 'html.parser')
#
# # 查找表格元素
# table = soup.find('table')
#
# # 检查表格是否存在
# if table is None:
#     print("未找到表格，请检查URL和HTML结构。")
# else:
#     # 提取表格数据
#     data = []
#     headers = [header.text for header in table.find_all('th')]
#
#     for row in table.find_all('tr')[1:]:
#         cells = row.find_all('td')
#         data.append([cell.text for cell in cells])
#
#     # 创建DataFrame
#     df = pd.DataFrame(data, columns=headers)
#
#     # 保存为CSV文件
#     df.to_csv('ssq_history.csv', index=False)
#
#     print("数据已保存到 ssq_history.csv 文件中")

# import requests
# from bs4 import BeautifulSoup
# import pandas as pd
#
# # 定义目标URL
# url = "https://example.com/ssq-history"  # 将此替换为实际提供双色球历史数据的URL
#
# # 发送HTTP请求
# response = requests.get(url)
#
# # 解析HTML内容
# soup = BeautifulSoup(response.content, 'html.parser')
#
# # 假设数据在一个表格中
# table = soup.find('table')
#
# # 提取表格数据
# data = []
# headers = [header.text for header in table.find_all('th')]
#
# for row in table.find_all('tr')[1:]:
#     cells = row.find_all('td')
#     data.append([cell.text for cell in cells])
#
# # 创建DataFrame
# df = pd.DataFrame(data, columns=headers)
#
# # 保存为CSV文件
# df.to_csv('ssq_history.csv', index=False)
#
# print("数据已保存到 ssq_history.csv 文件中")