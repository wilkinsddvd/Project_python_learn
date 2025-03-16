import time

stamp = time.time()
print("timestamp:", stamp)

localtime = time.asctime(time.localtime(stamp))
print("time:", localtime)