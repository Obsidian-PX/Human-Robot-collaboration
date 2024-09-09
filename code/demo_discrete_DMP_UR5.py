# -*- coding: utf-8 -*-
#%% 导入数据包
import time
import numpy as np
import pandas as pd

import sys
sys.path.append('./UR5/VREP_RemoteAPIs')
import UR5.VREP_RemoteAPIs.sim as vrep_sim

sys.path.append('./UR5')
from UR5.UR5SimModel import UR5SimModel

sys.path.append('./DMP')
from DMP.dmp_discrete import dmp_discrete

import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter
from mpl_toolkits.mplot3d import Axes3D

#%% 程序
print ('Program started')

# ------------------------------- Connect to VREP (CoppeliaSim) ------------------------------- 
vrep_sim.simxFinish(-1) # 以防万一，关闭所有打开的连接
while True:
    client_ID = vrep_sim.simxStart('127.0.0.1', 19999, True, True, 5000, 5) # Connect to CoppeliaSim
    if client_ID > -1: # connected
        print('Connect to remote API server.')
        break
    else:
        print('Failed connecting to remote API server! Try it again ...')

# 暂停仿真
# res = vrep_sim.simxPauseSimulation(client_ID, vrep_sim.simx_opmode_blocking)

delta_t = 0.01 # simulation time step
# Set the simulation step size for VREP  设置VREP的仿真步长
vrep_sim.simxSetFloatingParameter(client_ID, vrep_sim.sim_floatparam_simulation_time_step, delta_t, vrep_sim.simx_opmode_oneshot)
# Open synchronous mode  打开同步模式
vrep_sim.simxSynchronous(client_ID, True)
# Start simulation  开始仿真
vrep_sim.simxStartSimulation(client_ID, vrep_sim.simx_opmode_oneshot)

# ------------------------------- Initialize simulation model  初始化模拟模型-------------------------------
UR5_sim_model = UR5SimModel()    #创建 UR5SimModel
UR5_sim_model.initializeSimModel(client_ID)    #调用 initializeSimModel 方法，可能是用来初始化模拟环境，client_ID 可能是客户端的标识符，用于连接到 V-REP 服务器。


return_code, initial_dummy_handle = vrep_sim.simxGetObjectHandle(client_ID, 'initial', vrep_sim.simx_opmode_blocking)
if (return_code == vrep_sim.simx_return_ok):
    print('get initial dummy handle ok.')

return_code, goal_dummy_handle = vrep_sim.simxGetObjectHandle(client_ID, 'goal', vrep_sim.simx_opmode_blocking)
if (return_code == vrep_sim.simx_return_ok):
    print('get goal dummy handle ok.')

return_code, UR5_target_dummy_handle = vrep_sim.simxGetObjectHandle(client_ID, 'UR5_target', vrep_sim.simx_opmode_blocking)
if (return_code == vrep_sim.simx_return_ok):
    print('get UR5 target dummy handle ok.')

return_code, via_dummy_handle = vrep_sim.simxGetObjectHandle(client_ID, 'via', vrep_sim.simx_opmode_blocking)
if (return_code == vrep_sim.simx_return_ok):
    print('get via dummy handle ok.')

time.sleep(0.1)

#%% DMP learning
# 从文件中获取演示轨迹
df = pd.read_csv('./demo_trajectory/demo_trajectory_for_discrete_dmp.csv', header=None)  #文件位置
reference_trajectory = np.array(df)
data_dim = reference_trajectory.shape[0]  #获取参考轨迹的维度（即行数）
data_len = reference_trajectory.shape[1]  #获取参考轨迹的长度（即列数）

dmp = dmp_discrete(n_dmps=data_dim, n_bfs=1000, dt=1.0/data_len)   #创建一个离散的 DMP 模型，指定 DMP 的数量为参考轨迹的维度，基函数的数量为 1000，时间步长为轨迹长度的倒数。
dmp.learning(reference_trajectory)

reproduced_trajectory, _, _ = dmp.reproduce()

import matplotlib
matplotlib.use('TKAgg')  # 使用无图形界面后端
import matplotlib.pyplot as plt
fig = plt.figure(dpi=300)
fig = plt.figure(dpi=300)  # 设置 dpi 参数为 300，可以根据需要调整这个值
ax = fig.add_subplot(111, projection='3d')  # 修改为 add_subplot 方法
plt.plot(reference_trajectory[0,:], reference_trajectory[1,:], reference_trajectory[2,:], 'g', label='reference')
plt.plot(reproduced_trajectory[:,0], reproduced_trajectory[:,1], reproduced_trajectory[:,2], 'r--', label='reproduce')
plt.legend()
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('z')

# fig = plt.figure(dpi=1200)  # 设置 dpi 参数为 300，可以根据需要调整这个值
plt.subplot(311)
plt.plot(reference_trajectory[0,:], 'g', label='reference')
plt.plot(reproduced_trajectory[:,0], 'r--', label='reproduce')
plt.legend()

plt.subplot(312)
plt.plot(reference_trajectory[1,:], 'g', label='reference')
plt.plot(reproduced_trajectory[:,1], 'r--', label='reproduce')
plt.legend()

plt.subplot(313)
plt.plot(reference_trajectory[2,:], 'g', label='reference')
plt.plot(reproduced_trajectory[:,2], 'r--', label='reproduce')
plt.legend()

plt.draw()

#%% 主回路
print("Main loop is begining ...")
max_loop = 4     #设置循环的最大次数为10。
#创建三个二维数组，用于记录每次循环中重现的轨迹的 x、y、z 分量。
reproduced_trajectory_record_x = np.zeros((data_len, max_loop))
reproduced_trajectory_record_y = np.zeros((data_len, max_loop))
reproduced_trajectory_record_z = np.zeros((data_len, max_loop))


for loop in range(max_loop):
    if loop == 0:
        # DMP reproduce the reference trajectory  DMP再现参考轨迹
        reproduced_trajectory, _, _ = dmp.reproduce()
        q = UR5_sim_model.getAllJointAngles()  # 导出关节角度
        print(q)

    if loop == 1:
        # randomly add offset to the trajectory initial and goal positions   将偏移量随机添加到轨迹的初始位置和目标位置
        initial_pos = [reference_trajectory[0, 0] + np.random.uniform(0, 0), reference_trajectory[1, 0] + np.random.uniform(0, 0), reference_trajectory[2, 0] + np.random.uniform(0.08, 0.08)]
        vrep_sim.simxSetObjectPosition(client_ID, initial_dummy_handle, -1, initial_pos, vrep_sim.simx_opmode_oneshot)

        goal_pos = [reference_trajectory[0, -1] + np.random.uniform(0, 0), reference_trajectory[1, -1] + np.random.uniform(0, 0), reference_trajectory[2, -1] + np.random.uniform(0, 0)]
        vrep_sim.simxSetObjectPosition(client_ID, goal_dummy_handle, -1, goal_pos, vrep_sim.simx_opmode_oneshot)

        # DMP reproduce with new goal positions   DMP用新的球门位置再现
        reproduced_trajectory, _, _ = dmp.reproduce(initial=initial_pos, goal=goal_pos)
    if loop == 2:
        # randomly add offset to the trajectory initial and goal positions  将偏移量随机添加到轨迹的初始位置和目标位置
        initial_pos = [reference_trajectory[0,0] + np.random.uniform(0,0), reference_trajectory[1,0]  + np.random.uniform(0, 0), reference_trajectory[2,0] + np.random.uniform(0, 0)]
        vrep_sim.simxSetObjectPosition(client_ID, initial_dummy_handle, -1, initial_pos, vrep_sim.simx_opmode_oneshot)

        goal_pos = [reference_trajectory[0,-1] + np.random.uniform(0,0), reference_trajectory[1,-1]  + np.random.uniform(0.08, 0.08), reference_trajectory[2,-1] + np.random.uniform(0, 0)]
        vrep_sim.simxSetObjectPosition(client_ID, goal_dummy_handle, -1, goal_pos, vrep_sim.simx_opmode_oneshot)

        # DMP reproduce with new initial and goal positions   DMP通过新的初始位置和目标位置进行复制
        reproduced_trajectory, _, _ = dmp.reproduce(initial=initial_pos, goal=goal_pos)
    if loop == 3:
        # randomly add offset to the trajectory initial and goal positions  将偏移量随机添加到轨迹的初始位置和目标位置
        initial_pos = [reference_trajectory[0,0] + np.random.uniform(0,0), reference_trajectory[1,0]  + np.random.uniform(0, 0), reference_trajectory[2,0] + np.random.uniform(0.08, 0.08)]
        vrep_sim.simxSetObjectPosition(client_ID, initial_dummy_handle, -1, initial_pos, vrep_sim.simx_opmode_oneshot)

        goal_pos = [reference_trajectory[0,-1] + np.random.uniform(0,0), reference_trajectory[1,-1]  + np.random.uniform(0.08, 0.08), reference_trajectory[2,-1] + np.random.uniform(0, 0)]
        vrep_sim.simxSetObjectPosition(client_ID, goal_dummy_handle, -1, goal_pos, vrep_sim.simx_opmode_oneshot)

        # DMP reproduce with new initial and goal positions   DMP通过新的初始位置和目标位置进行复制
        reproduced_trajectory, _, _ = dmp.reproduce(initial=initial_pos, goal=goal_pos)

    data_len = reproduced_trajectory.shape[0]
    reproduced_trajectory_record_x[:,loop] = reproduced_trajectory[:,0]
    reproduced_trajectory_record_y[:,loop] = reproduced_trajectory[:,1]
    reproduced_trajectory_record_z[:,loop] = reproduced_trajectory[:,2]

    # go to the goal position  去球门位置
    for i in range(data_len):
        UR5_target_pos = reproduced_trajectory[i,:]
        vrep_sim.simxSetObjectPosition(client_ID, via_dummy_handle, -1, UR5_target_pos, vrep_sim.simx_opmode_oneshot)
        q = UR5_sim_model.getAllJointAngles()  # 导出关节角度
        print(q)
        vrep_sim.simxSynchronousTrigger(client_ID)  # trigger one simulation step
        vrep_sim.simxGetPingTime(client_ID)  # Ensure that the last command sent out had time to arrive

    # go back to the initial position  回到初始位置
    for i in range(data_len-1, 0, -1):
        UR5_target_pos = reproduced_trajectory[i,:]
        vrep_sim.simxSetObjectPosition(client_ID, via_dummy_handle, -1, UR5_target_pos, vrep_sim.simx_opmode_oneshot)
        q = UR5_sim_model.getAllJointAngles()
        print(q)
        vrep_sim.simxSynchronousTrigger(client_ID)  # trigger one simulation step  触发一个模拟步骤
        vrep_sim.simxGetPingTime(client_ID)  # Ensure that the last command sent out had time to arrive

vrep_sim.simxStopSimulation(client_ID, vrep_sim.simx_opmode_blocking) # stop the simulation
vrep_sim.simxFinish(-1)  # Close the connection
print('Program terminated')

#%% Plot 在三维空间中绘制参考轨迹和各次循环中重现的轨迹。
fig = plt.figure(dpi=1200)  # 设置 dpi 参数为 300，可以根据需要调整这个值
ax = fig.add_subplot(111, projection='3d')  # 修改为 add_subplot 方法
plt.plot(reference_trajectory[0,:], reference_trajectory[1,:], reference_trajectory[2,:], 'g', label='teaching trajectory')
for i in range(max_loop):
    plt.plot(reproduced_trajectory_record_x[:,i], reproduced_trajectory_record_y[:,i], reproduced_trajectory_record_z[:,i], '--', label=f'learning trajectory-{i}')


# 绘制轨迹的起点
reference_start_point = reference_trajectory[:, 0]
ax.scatter(*reference_start_point, color='blue', label='start point-0,2')
# 绘制再现轨迹的起点
reproduced_start_point = [reproduced_trajectory_record_x[0, 1], reproduced_trajectory_record_y[0, 1], reproduced_trajectory_record_z[0, 1]]
ax.scatter(*reproduced_start_point, color='green', label='start point-1,3')
# 绘制轨迹的终点
reference_end_point = reference_trajectory[:, -1]
ax.scatter(*reference_end_point, color='yellow', label='end point-0,1')
# 绘制再现轨迹的终点
reproduced_end_point = [reproduced_trajectory_record_x[-1, -1], reproduced_trajectory_record_y[-1, -1],
                        reproduced_trajectory_record_z[-1, -1]]
ax.scatter(*reproduced_end_point, color='red', label='end point-2,3')




plt.legend(loc='upper left', bbox_to_anchor=(-0.4, 1.15), framealpha=0)  # 图例放在图的右上角，略微向右偏移)
ax.xaxis.set_major_formatter(FormatStrFormatter('%.2f'))
ax.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
ax.zaxis.set_major_formatter(FormatStrFormatter('%.2f'))
ax.set_xlabel('x (m)')
ax.set_ylabel('y (m)')
ax.set_zlabel('z (m)')
plt.show()

# import csv
#
# # 假设你想保存第一个循环后的轨迹，索引为0
# loop_index = 1
# x_coords = reproduced_trajectory_record_x[:, loop_index]
# y_coords = reproduced_trajectory_record_y[:, loop_index]
# z_coords = reproduced_trajectory_record_z[:, loop_index]
#
# # 指定CSV文件名
# csv_filename = 'trajectory_points4.csv'
#
# # 打开CSV文件并写入数据
# with open(csv_filename, mode='w', newline='') as file:
#     writer = csv.writer(file)
#     # 写入标题行
#     writer.writerow(['Point', 'X', 'Y', 'Z'])
#     # 写入坐标点
#     for i in range(len(x_coords)):
#         writer.writerow([i, x_coords[i], y_coords[i], z_coords[i]])
#
# print(f"Trajectory points have been saved to {csv_filename}")




