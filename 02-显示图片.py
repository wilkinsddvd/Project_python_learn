import os
import pygame

pygame.init()

window =pygame.display.set_mode((1024,1024))
pygame.display.set_caption('显示图片')

#设置背景颜色
window.fill((255,255,255))

#=============游戏开始页面静态效果=======
#显示图片
#加载图片
image1 = pygame.image.load('D:\好看的.jpg')
#渲染图片
#blit（渲染对象,坐标）

window.blit(image1,(0,0))

#刷新显示页面
#第一次刷新
pygame.display.flip()

#第一次以后的刷新 pygame.display.update()
while True:
    #==========游戏帧数刷新=========
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            exit()