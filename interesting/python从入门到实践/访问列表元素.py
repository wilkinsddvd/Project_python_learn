bicycle = ['trek','cannondale','redline','specialized']
print(bicycle[0])
print(bicycle[0].title())
print(bicycle[1])
print(bicycle[3])
print(bicycle[-1])

message = f"My first bicycle was a {bicycle[0].title()}"
print(message)

motorcycles = ['honda','yamaha','suzuki']
print(motorcycles)

motorcycles[0] = 'ducati'
print(motorcycles)

motorcycles.append('ducati')
print(motorcycles)

motorcycles = []

motorcycles.append('honda')
motorcycles.append('yamaha')
motorcycles.append('suzuki')

print(motorcycles)

del motorcycles[0]
print(motorcycles)
del motorcycles[1]
print(motorcycles)

motorcycles = ['honda','yamaha','suzuki']
print(motorcycles)

popped_motorcycle = motorcycles.pop()
print(motorcycles)
print(popped_motorcycle)

motorcycles = ['honda','yamaha','suzuki']

last_owned = motorcycles.pop()
print(f"The last motorcycle I owned was a {last_owned.title()}.")

first_owned = motorcycles.pop(0)
print(f"The first motorcycle I owned was a {first_owned.title()}.")

motorcycles = ['honda','yamaha','sukuzi','ducati']
print(motorcycles)

motorcycles.remove('ducati')
print(motorcycles)

motorcycles = ['honda','yamaha','sukuzi','ducati']
print(motorcycles)

too_expensive = 'ducati'
motorcycles.remove(too_expensive)

print(motorcycles)
print(f"A {too_expensive.title()} is too expensive for me.")

motorcycles = ['honda','yamaha','suzuki']
print(motorcycles[2])
#因为print(motorcycles[3])不存在，超出数组长度，所以，会报错，导致索引错误

motorcycles = ['honda','yamaha','suzuki']
print(motorcycles[-1])

#motorcycles=[]
#print(motorcycles[-1])
#当列表为空时，这时候输出列表-1会报错，因为列表中是空表，不存在任何元素，同样，也无法输出
