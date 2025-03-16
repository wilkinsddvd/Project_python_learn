import pandas as pd
import openpyxl

df = pd.read_excel('数据集12.xlsx', sheet_name='Sheet1')
# wb = openpyxl.load_workbook('数据集12.xlsx')
# wb['Sheet1']

# 4. 处理数据：将Excel中的数据存储在Pandas的DataFrame对象中，可以对DataFrame对象进行各种操作，如数据筛选、数据清洗、数据转换等。

print(df)
print(wb)