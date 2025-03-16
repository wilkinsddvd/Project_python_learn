plays = ['charles','martina','michael','florence','eli']
print(plays[0:3])

print(plays[1:4])

print(plays[:4])

print(plays[2:])

print(plays[-3:])

plays = ['charles','martina','michael','florence','eli']

print("Here are the first three players on my team:")

for player in plays[:3]:
    print(player.title())

my_foods = ['pizza','falafel','carrot cake']
friend_foods = my_foods[:]

print("My favorite foods are:")
print(my_foods)

print("\nMy friend's favorite food are:")
print(friend_foods)

my_foods.append('cannoli')
friend_foods.append('ice cream')

print("\nMy favorite food are:")
print(my_foods)

print("\nMy friend's favorite food are:")
print(friend_foods)

friend_foods = my_foods #与前面friend_food = my_foods[:]对比,两者逻辑不一样

my_foods.append('cannoli')
friend_foods.append('ice cream')

print("\nMy favorite food are:")
print(my_foods)

print("\nMy friend's favorite food are:")
print(friend_foods)
