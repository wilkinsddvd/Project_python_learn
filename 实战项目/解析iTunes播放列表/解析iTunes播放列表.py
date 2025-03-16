import re
import argparse
import sys
from matplotlib import pyplot
import plistlib
import numpy as np

def findCommonTracks(fileNames):
    """
    在给定的播放列表文件中找到共同的音轨，
    并将它们保存到common.txt文件中。
    """
    # 一个包含音轨名称集合的列表
    trackNameSets = []
    for fileNames in fileNames:
        # 创建一个新的集合
        trackNames = set()
        # 读取播放列表
        plist = plistlib.readPlist(fileNames)
        # 获取音轨
        tracks = plist['Tracks']
        # 遍历音轨
        for trackId, tracks in tracks.items():
            try:
                # 将音轨名称添加到集合中
                trackNames.add(tracks['Name'])
            except:
                # 忽略异常
                pass
        # 添加到列表中
        trackNameSets.append(trackNames)
    # 获取共同的音轨集合
    commonTracks = set.intersection(*trackNameSets)
    # 写入文件
    if len(commonTracks) > 0:
        f = open("common.txt", 'w')
        for val in commonTracks:
            s = "%s\n" % val
            f.write(s.encode("UTF-8"))
        f.close()
        print("找到%d个共同的音轨。音轨名称已写入common.txt文件。"%len(commonTracks))
    else:
        print("没有共同的音轨！")

def plotStats(fileName):
    """
    通过读取播放列表中的音轨信息，绘制一些统计图表。
    """
    # 读取播放列表
    plist = plistlib.readPlist(fileName)
    # 获取音轨
    tracks = plist['Tracks']
    # 创建歌曲评分和音轨时长的列表
    ratings = []
    durations = []
    # 遍历音轨
    for trackId, track in tracks.items():
        try:
            ratings.append(track['Album Rating'])
            durations.append(track['Total Time'])
        except:
            # 忽略异常
            pass
    # 确保已收集到有效的数据
    if ratings == [] or durations == []:
        print("在%s中没有找到有效的专辑评分/音轨时长数据。"%fileName)
        return

    # 散点图
    x = np.array(durations, np.int32)
    # 转换为分钟
    x = x/60000.0
    y = np.array(ratings, np.int32)
    pyplot.subplots(2,1,1)
    pyplot.plot(x, y, 'o')
    pyplot.axis([0, 1.05*np.max(x), -1, 110])
    pyplot.xlabel('音轨时长')
    pyplot.ylabel('音轨评分')
    # 绘制直方图
    pyplot.subplots(2, 1, 2)
    pyplot.hist(x, bins=20)
    pyplot.xlabel('音轨时长')
    pyplot.ylabel('计数')
    # 显示图表
    pyplot.show()

def findDuplicates(fileName):
    """
    在给定的播放列表中找到重复的音轨。
    """
    print('在%s中查找重复的音轨...'%fileName)
    # 读取播放列表
    plist = plistlib.readPlist(fileName)
    # 获取音轨
    tracks = plist['Tracks']
    # 创建一个音轨名称字典
    trackNames = {}
    # 遍历音轨
    for trackId, track in tracks.items():
        try:
            name = track['Name']
            duration = track['Total Time']
            # 查找现有的条目
            if name in trackNames:
                # 如果名称和时长匹配，则增加计数
                # 将音轨时长舍入到最近的秒
                if duration//1000 == trackNames[name][0]//1000:
                    count = trackNames[name][1] + 1
                    trackNames[name] = (duration, count)
                else:
                    # 添加字典条目，形式为元组（时长，计数）
                    trackNames[name] = (duration, 1)
        except:
            # 忽略异常
            pass
    # 将重复的音轨保存为（名称，计数）元组
    dups = []
    for k,v in trackNames.items():
        if v[1] > 1:
            dups.append((v[1], k))
    # 将重复的音轨保存到文件中
    if len(dups) > 0:
        print("找到%d个重复的音轨。音轨名称已保存到dups.txt文件。"%len(dups))
        f = open("dups.txt", 'w')
        for val in dups:
            f.write("[%d] %s\n" %(val[0],val[1]))
        f.close()
    else:
        print("没有找到重复的音轨！")

# 主函数
def main():
    # 创建解析器
    descStr = """
    此程序分析从iTunes中导出的播放列表文件(.xml)。
    """
    parser = argparse.ArgumentParser(description=descStr)
    # 添加互斥的参数组
    group = parser.add_mutually_exclusive_group()

    # 添加导出的参数
    group.add_argument('--common', nargs='*', dest='plFiles', required=False)
    group.add_argument('--stats', dest='plFile', required=False)
    group.add_argument('--dup', dest='plFileD', required=False)

    # 解析参数
    args = parser.parse_args()

    if args.plFiles:
        # 找到共同的音轨
        findCommonTracks(args.plFiles)
    elif args.plFile:
        # 绘制统计图表
        plotStats(args.plFiles)
    elif args.plFileD:
        # 找到重复的音轨
        findDuplicates(args.plFileD)
    else:
        print("这不是你要找的音轨。")

# 调用主函数
if __name__ == '__main__':
    main()