import pygame
pygame.init()

#创建窗口
window = pygame.display.set_mode((400,600))

#设置游戏标题
pygame.display.set_caption('mygame')



#保持运行
#game loop(游戏循环)
while True:
    #检测事件
    for event in pygame.event.get():
        #检测关闭按钮被点击的事件
        if event.type == pygame.QUIT:
            #退出
            exit()
        pass