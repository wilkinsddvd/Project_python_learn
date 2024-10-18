def fibo(n):
    """递归函数实现斐波那契数列"""
    if n == 1 or n == 2:
        return 1
    else:
        return fibo(n-1) + fibo(n -2)


if __name__ == '__main__':
    n = int(input("请输入数列的项数："))
    res = fibo(n)
    print(res)