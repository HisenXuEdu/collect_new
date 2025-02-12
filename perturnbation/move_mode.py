import threading
from dobot_api import DobotApiDashboard, DobotApi, DobotApiMove, MyType, alarmAlarmJsonFile
from time import sleep
import time
import numpy as np
import re
import random
import pandas as pd
import datetime
import multiprocessing
import os



def ConnectRobot():
    try:
        ip = "192.168.5.1"
        dashboardPort = 29999
        movePort = 30003
        # print("正在建立连接...")
        dashboard = DobotApiDashboard(ip, dashboardPort)
        move = DobotApiMove(ip, movePort)
        # print(">.<move连接成功>!<")
        return dashboard, move
    except Exception as e:
        print(":(move连接失败:(")
        raise e
    
def pose_open():
    os.system('python ./cr5/pose_record.py')



if __name__ == '__main__':
    dashboard, move = ConnectRobot()
    dashboard.SetSafeSkin(0)
    dashboard.SetCollisionLevel(1)
    dashboard.SpeedFactor(10)
    dashboard.EnableRobot()
    dashboard.SpeedFactor(30)

    initial_angle = [90,0,90,0,-90,0]
    move.JointMovJ(90,0,90,0,-90,0)
    move.RelMovJUser(0,0,-100,0,0,0,0)

    sec = 30

    move.RelMovJUser(0,0,-10,0,0,0,0)
    sleep(3)

    p1 = multiprocessing.Process(target=pose_open)
    p1.start()
    time.sleep(3)

    move.RelMovJUser(30,0,0,0,0,0,0)
    sleep(0.5)
    move.RelMovJUser(-30,0,0,0,0,0,0)
    sleep(2)

    t_start = time.time()
    t_start_string = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(t_start))
    print("运动开始时间：", t_start_string)
    t_end = time.time() + sec

    dashboard.SpeedFactor(80)
    x_last = 0
    y_last = 0
    z_last = 0

    while time.time() < t_end:
        x = (random.random()-0.5)*30
        x_move = x - x_last
        y = (random.random()-0.5)*30
        y_move = y - y_last
        z = (random.random()-0.5)*30
        z_move = z - z_last
        move.RelMovJUser(x_move,y_move,z_move,0,0,0,0)
        sleep(0.5)
        # move.RelMovJUser(-x,0,0,0,0,0,0)
        # sleep(0.5)
        x_last = x
        y_last = y
        z_last = z
        


    t_end = time.time()
    t_end_string = time.strftime("%Y-%m-%d-%H:%M:%S", time.localtime(t_end))
    print("运动结束时间：", t_end_string)

    dashboard.close()
    move.close()

    
    p1.join()