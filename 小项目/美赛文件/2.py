import random

# 初始化种群参数
population_size = 100
female_ratio = 0.5  # 种群中雌性的比率
max_age = 10  # 七鳃鳗的最大年龄
birth_rate = 0.2  # 繁殖率
death_rate = 0.1  # 死亡率
time_steps = 50  # 模拟的时间步数

# 初始化种群
def initialize_population(size,female_ratio):
    population = []
    for _ in range(size):
        sex = 1 if random.random()<female_ratio else 0
        age = random.randint(1,max_age)
        population.append((sex,age))
        return population

# 进行一次模拟步骤
def simulation_step (population,birth_rate,death_rate,max_age):
    new_population = []
    births = 0
    deaths = 0

    # 模拟每个个体的存活和繁殖
    for individual in population:
        sex,age = individual

        # 检查个体是否死亡
        if random.randint()<death_rate or age>=max_age:
            deaths += 1
            continue


