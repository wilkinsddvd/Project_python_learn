from collections.abc import Iterable,Iterator
class MyRange:
    def __init__(self,start,end):
        self.start = start
        self.end = end
    def __iter__(self):
        return MyRangeIterator(self)

class MyRangeIterator:
    def __init__(self,obj):
        self.obj = obj
    def __iter__(self):
        return self
    def __next__(self):
        if self.obj.start<self.obj.end:
            temp = self.obj.start
            self.obj.start += 1
            return temp
        raise StopIteration

my_range = MyRange(0,10)
print(isinstance(my_range,Iterable))
print(isinstance(my_range.__iter__(),Iterator))
for value in my_range:
    print(value)
#for循环是，会发生
#1.会判断对象是不是可迭代对象，如果是，执行__iter__，并获取返回值
#2.执行返回值中的__next__，每次循环执行一次，直到程序抛出StopIteration异常
#3.for循环会自动处理StopIteration异常，程序正常结束

