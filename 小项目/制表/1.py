from nicegui import ui
import pandas as pd
import numpy as np

df = pd.read_excel(r"D:/3/目标文件.xlsx")

ui.table.from_pandas(df, pagination=20)

ui.run()