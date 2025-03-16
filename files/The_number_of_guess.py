import random
number = int(random.randint(1,102410))
guess = -1
print("----- Guess Number----")
while guess != number:
    guess = int(input("input number:"))

    if guess == number:
        print("Bingo")
    elif guess < number:
        print("Less...")
    elif guess > number:
        print("More...")