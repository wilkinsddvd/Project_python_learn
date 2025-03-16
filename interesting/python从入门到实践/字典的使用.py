user_0 = {
    'username' : 'abc',
    'first' : '1',
    'lasr' : '2',
}

for key,value in user_0.items():
    print(f"\nKey:{key}")
    print(f"\nValue:{value}")

favorite_language = {
    'jen' : 'python',
    'sarah' : 'c',
    'edward' : 'ruby',
    'phil' : 'python',
}

for name,value in favorite_language.items():
    print(f"\n{name.title()} 's favorite language is {value.title()}.")

favorite_language = {
    'jen' : 'python',
    'sarah' : 'c',
    'edward' : 'ruby',
    'phil' : 'python',
}

friend = ['phil','sarah']
for name in favorite_language.keys():
    print(f"Hi {name.title()}.")

    if name in friend:
        language = favorite_language[name].title()
        print(f"{name.title()},I see you love {language}.")

favorite_language = {
    'jen' : 'python',
    'sarah' : 'c',
    'edward' : 'ruby',
    'phil' : 'python',
}

for name in sorted(favorite_language.keys()):
    print(f"{name.title()} ,thank you for taking the poll.")

print("The following language s have been mentioned:")
for language in favorite_language.values():
    print(language.title())