import os
import time
import multiprocessing

def trigno_open():
    os.system('python ./emg/emg_record.py')

def force_open():
    os.system('python ./force/force_record.py')

if __name__ == "__main__":
    start = time.time()
    print('开始时间：', time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(start)))

    p1 = multiprocessing.Process(target=trigno_open)
    p2 = multiprocessing.Process(target=force_open)

    # 启动子进程
    p1.start()
    time.sleep(3)
    p2.start()

    #等待程序执行完成
    p1.join()
    p2.join()
    end = time.time()
    print('结束时间：', time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(end)))