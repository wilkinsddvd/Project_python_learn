import time

# def timestamp(t):
#     timeArray = time.strptime(t, "%Y-%m-%d %H:%M:%S")
#     timeStamp = int(time.mktime(timeArray))
#     return timeStamp

'''
数据形式
'''
# 2017-06-10 17:24:56|京东|原始价格:8998.0|优惠券:满8000减800|到手价格:8198.0|价格趋势:-
# 2017-06-10 17:24:56|淘宝_A|原始价格:7890.0|运费:免运费|到手价格:7890.0|价格趋势:-
# 2017-06-10 17:24:56|淘宝_B|原始价格:7800.0|运费:39.0|到手价格:7839.0|价格趋势:-
# 2017-06-10 17:24:56|淘宝_C|原始价格:7800.0|运费:30.0|到手价格:7830.0|价格趋势:-
# 2017-06-10 17:24:56|淘宝_D|原始价格:7750.0|运费:免运费|到手价格:7750.0|价格趋势:-
# 2017-06-10 17:24:56|最优价格方:淘宝_D|目前最低到手价格:7750.0|历史最低价:7750.0|价格趋势:-


shop = {}
with open('D:/3/目标文件.xlsx') as f:
    for line in f:
        linesplit = line.split("|")
        if ':' in linesplit[1]:  # 剔除最优价格方的记录
            pass
        else:
            needline = str(linesplit[0]) + '|' + linesplit[-2].split(':')[-1]  # 获取时间戳和到手价格，毕竟分析的时候只需要这两个就行
            if linesplit[1] not in shop:  # 进行字典的构建，里面存储list
                shop[linesplit[1]] = []
            shop[linesplit[1]].append(needline)
            # 预处理-替换缺失值，向上替换，pandas里有直接的方法，这里选择字符串处理方法
            if shop[linesplit[1]][-1].split("|")[-1] == 'None':
                shop[linesplit[1]][-1] = shop[linesplit[1]][-1].split("|")[0] + '|' + shop[linesplit[1]][-2].split("|")[-1]
                t += 1
            else:
                pass
            # 预处理-波动异常点-由于抓取的时候店家对链接的修改，导致元素异位

            try:
                if ((float(shop[linesplit[1]][-1].split("|")[-1]) > float(
                        shop[linesplit[1]][-2].split("|")[-1]) + 500) or (
                            float(shop[linesplit[1]][-1].split("|")[-1]) < float(
                            shop[linesplit[1]][-2].split("|")[-1]) - 500)) and (linesplit[1] != '京东'):
                    shop[linesplit[1]][-1] = shop[linesplit[1]][-1].split("|")[0] + '|' + \
                                             shop[linesplit[1]][-2].split("|")[-1]
                else:
                    pass
            except Exception as ex:
                print(ex)


def find_change_time(shop):
    chang_ = []
    for shopname in shop.keys():
        pricedetail = shop[shopname]
        for i in range(len(pricedetail) - 2):  # 最后一个时刻抛弃
            split_f = pricedetail[i].split('|')
            split_s = pricedetail[i + 1].split('|')
            price_f = split_f[-1]
            time_f = split_f[0]
            price_s = split_s[-1]
            time_s = split_s[0]
            if price_f == price_s:
                pass
            else:
                chang_.append(time_s)
    return chang_


# 获取价格变更时间戳
changtime = sorted(find_change_time(shop))
print(changtime)

for shopname in shop.keys():
    satisfy_ = [k.split('|')[-1] for k in shop[shopname] if k.split('|')[0] in changtime]
    print(satisfy_)
