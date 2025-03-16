cars = ['audi','bmw','subaru','toyota']

for car in cars:
    if car == 'bmw':
        print(car.upper())
    else:
        print(car.title())

car = 'Audi'
print(car.lower() == 'audi')

requested_topping = 'mushrooms'

if requested_topping != 'anchovies':
    print("Hold the anchovies!")

answer = 17

if answer != 42:
    print("That is not the correct answer.Please try again!")

age_0 = 22
age_1 = 18
print(age_0 >= 21 and age_1 >=21)

requested_topping = ['mushroom','onions','pineapple']

print('mushroom' in requested_topping)

banned_users = ['andrew','carolina','david']
user = 'marie'

if user not in banned_users:
    print(f"{user.title()},you can post a response if you wish.")




