import threading
import time

def test():
    for i in range(5):
        print(threading.current_thread())
        time.sleep(1)

thread = threading.Thread(target=test)
thread.start()

for i in range(5):
    print(threading.current_thread().name, i)
    time.sleep(1)