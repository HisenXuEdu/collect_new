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
from force.force import Force
from emg.emg import Emg_S



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




class Per:
    def __init__(self):
        self.dashboard, self.move = self.__connect_robot()
        self.force=Force()
        self.channel = 4
        self.emg = Emg_S(channel=self.channel, host='127.0.0.1', samples_per_read=400)
        self.emg_list = []
        self.force_list = []
        self.pose_list = []

        self.force_thread = self.start_thread(target=self.force.get_force)
        self.record_thread = self.start_thread(target=self.record)
        self.record_emg_thread = self.start_thread(target=self.record_emg)
        self.time_thread = self.start_thread(target=self.time_generate)

        self.record_flag = False
        self.stop_flag = False

    def __connect_robot(self):
        try:
            ip = "192.168.5.1"
            dashboardPort = 29999
            movePort = 30003
            dashboard = DobotApiDashboard(ip, dashboardPort)
            move = DobotApiMove(ip, movePort)
            dashboard.EnableRobot()
            dashboard.ClearError()
            dashboard.SetSafeSkin(0)
            dashboard.SetCollisionLevel(0)
            dashboard.SpeedFactor(60)
            return dashboard, move
        except Exception as e:
            print(":(move连接失败:(")
            raise e

    def start_thread(self, target, args = ()):
        thread = threading.Thread(target=target, args=args)
        thread.daemon = True
        thread.start()
        return thread

    def record(self):
        # 每隔0.1s记录一次
        interval = 0.1
        force_l = []
        pose_l = []
        while True:
            start_time = time.time()  # 记录任务开始时间
            force_l.append(self.force.force)

            pose_data = self.dashboard.GetPose()
            pose_values = re.findall(r"[-+]?\d*\.\d+|\d+", pose_data)
            # 取pose_values除第一列以外的
            pose_values = pose_values[1:]
            pose_list = [float(value) for value in pose_values]
            pose_l.append(pose_list)
            print(pose_list)

            time.sleep(interval - ((time.time() - start_time) % interval))
            if self.record_flag:
                self.force_list.append(force_l)
                self.pose_list.append(pose_l)
                force_l = []
                pose_l = []
                # 等待下一次记录self.record_flag变成False
                while self.record_flag:
                    time.sleep(0.01)
            if self.stop_flag:
                break

    def record_emg(self):
        # 每隔0.1s记录一次
        interval = 0.1
        data_EMG = np.zeros((self.channel, 0))
        while True:
            start_time = time.time()
            emg = self.emg.get_single_raw()
            # 将emg沿着第一个轴拼接
            data_EMG = np.concatenate((data_EMG,emg),axis=1)
            if self.record_flag:
                self.emg_list.append(data_EMG)
                data_EMG = np.zeros((self.channel, 0))
                # 等待下一次记录self.record_flag变成False
                while self.record_flag:
                    time.sleep(0.01)
            if self.stop_flag:
                break
                

    def run(self):

        self.move.MovL(140, -480, 300, 179, 0, 179)
        self.move.Sync()
        t_start = time.time()
        t_start_string = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(t_start))
        print("运动开始时间：", t_start_string)
        t_end = time.time() + 40
        self.dashboard.SpeedFactor(80)
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
            self.move.RelMovJUser(x_move,y_move,z_move,0,0,0,0)
            sleep(0.5)
            x_last = x
            y_last = y
            z_last = z
    
    def time_generate(self):
        for i in range(6):
            self.record_flag = False
            time.sleep(5)
            self.record_flag = True
            print("record ", i)
            time.sleep(1)
            self.record_flag = False
        self.stop_flag = True
        self.stop()
    
    def stop(self):
        self.record_thread.join()
        self.record_emg_thread.join()
        # 将数据保存到文件
        self.save_data()
        print('数据保存成功')
    
    def save_data(self):
        time = datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
        # self.force_list, self.pose_list，self.emg_list合并成一个列表
        data = []
        data.append(self.force_list)
        data.append(self.pose_list)
        data.append(self.emg_list)
        pd.to_pickle(data, './data/3.8/data'+time+'.pkl')

if __name__ == '__main__':
    per = Per()
    per.run()
