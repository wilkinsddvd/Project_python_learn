# coding: utf-8
# 程序文件Pex4_4.py
from scipy.stats import binom
n, p = 20, 0.8
mean, variance, skewness, kurtosis = binom.stats(n, p, moments='mvsk')
# 上述语句不显示，只为了说明数据顺序
print("所求的数字特征为：", binom.stats(n, p, moments='mvsk'))

n, p = 20, 0.8
mean, variance, skewness, kurtosis = binom.stats(n, p, moments='mvsk')

# 使用round函数对variance进行四舍五入
variance_rounded = round(variance, 1)

# 使用格式化字符串输出结果
print(f"所求的数字特征为：均值 {mean:.1f}，方差 {variance_rounded:.1f}，偏度 {round(skewness, 1):.1f}，峰度 {round(kurtosis, 1):.1f}")
