import pandas as pd
import matplotlib.pyplot as plt

from pylab import mpl
# 设置显示中文字体
mpl.rcParams["font.sans-serif"] = ["SimHei"]

# 读取Excel文件
file_path = 'target.xlsx'  # 替换为你的Excel文件路径
sheet_name = 'Sheet1'  # 替换为你的工作表名称
df = pd.read_excel(file_path, sheet_name=sheet_name)
''
# 假设'ColumnA'到'ColumnZ'是你要进行计算的列（即第1列到第26列）
columns_to_process = df.columns[11:62]  # 第12列到第64列（索引从0开始，所以第12列是索引11，第64列是索引63）

# 数据清洗：在每个原始数据的前提下加10
for column in columns_to_process:
    df[column] = df[column] + 10

# 计算结果：经过数据清洗后的数据之和
sum_of_cleaned_data = df[columns_to_process].sum().sum()

# 分析数据：根据计算结果绘制饼图
# 假设你想要根据每列的和来绘制饼图，而不是单一的总和
column_sums = df[columns_to_process].sum()
fig, ax = plt.subplots()
ax.pie(column_sums, labels=column_sums.index, autopct='%1.1f%%', startangle=90)

# 设置标题和等比例显示
ax.set_title('Pie Chart Based on Sums of Cleaned Data')
ax.axis('equal')

# 显示饼图
plt.show()
