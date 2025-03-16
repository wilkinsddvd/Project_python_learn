# 1引入需要的模块
import pygame
import random

# 1配置图片地址
IMAGE_PATH = '0.jpg'
# 1设置页面宽高
scrrr_width = 800
scrrr_height = 560
# 1创建控制游戏结束的状态
GAMEOVER = False


# 1主程序
class MainGame():
    # 1加载游戏窗口
    def init_window(self):
        # 1调用显示模块的初始化
        pygame.display.init()
        # 1创建窗口
        MainGame.window = pygame.display.set_mode([scrrr_width, scrrr_height])  #

    # 1开始游戏
    def start_game(self):
        # 1初始化窗口
        self.init_window()
        # 1只要游戏没结束，就一直循环
        while not GAMEOVER:
            # 1渲染白色背景
            MainGame.window.fill((255, 255, 255))
            # 1实时更新
            pygame.display.update()

#2 创建关数，得分，剩余分数，钱数
shaoguan = 1
score = 0
remnant_score = 100
money = 200
#2 文本绘制
def draw_text(self, content, size, color):
    pygame.font.init()
    font = pygame.font.SysFont('kaiti', size)
    text = font.render(content, True, color)
    return text

#2 加载帮助提示
def load_help_text(self):
    text1 = self.draw_text('1.按左键创建向日葵 2.按右键创建豌豆射手', 26, (255, 0, 0))
    MainGame.window.blit(text1, (5, 5))

#2 渲染的文字和坐标位置
    MainGame.window.blit(self.draw_text('当前钱数$: {}'.format(MainGame.money), 26, (255, 0, 0)), (500, 40))
    MainGame.window.blit(self.draw_text(
            '当前关数{}，得分{},距离下关还差{}分'.format(MainGame.shaoguan, MainGame.score, MainGame.remnant_score), 26,
        (255, 0, 0)), (5, 40))
    self.load_help_text()


# 3 创建地图类
class Map():
    # 3 存储两张不同颜色的图片名称
    map_names_list = [IMAGE_PATH + '0.jpg', IMAGE_PATH + '0.jpg']

    # 3 初始化地图
    def __init__(self, x, y, img_index):
        self.image = pygame.image.load(Map.map_names_list[img_index])
        self.position = (x, y)
        # 是否能够种植
        self.can_grow = True

    # 3 加载地图
    def load_map(self):
        MainGame.window.blit(self.image, self.position)

    # 3 存储所有地图坐标点
    map_points_list = []
    # 3 存储所有的地图块
    map_list = []

    # 3 初始化坐标点
    def init_plant_points(self):
        for y in range(1, 7):
            points = []
            for x in range(10):
                point = (x, y)
                points.append(point)
            MainGame.map_points_list.append(points)
            print("MainGame.map_points_list", MainGame.map_points_list)

    # 3 初始化地图
    def init_map(self):
        for points in MainGame.map_points_list:
            temp_map_list = list()
            for point in points:
                # map = None
                if (point[0] + point[1]) % 2 == 0:
                    map = Map(point[0] * 80, point[1] * 80, 0)
                else:
                    map = Map(point[0] * 80, point[1] * 80, 1)
                # 将地图块加入到窗口中
                temp_map_list.append(map)
                print("temp_map_list", temp_map_list)
            MainGame.map_list.append(temp_map_list)
        print("MainGame.map_list", MainGame.map_list)

    # 3 将地图加载到窗口中
    def load_map(self):
        for temp_map_list in MainGame.map_list:
            for map in temp_map_list:
                map.load_map()

    # 3 初始化坐标和地图
        self.init_plant_points()
        self.init_map()

    # 3 需要反复加载地图
        self.load_map()


# 1启动主程序
if __name__ == '__main__':
    game = MainGame()
    game.start_game()

