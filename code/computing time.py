import time
import numpy as np
import pandas as pd
from DMP.dmp_discrete import dmp_discrete

# 记录开始时间
start_time = time.time()

#%% DMP learning
# 从文件中获取演示轨迹
df = pd.read_csv('./demo_trajectory/demo_trajectory_for_discrete_dmp.csv', header=None)  #文件位置
reference_trajectory = np.array(df)
data_dim = reference_trajectory.shape[0]  #获取参考轨迹的维度（即行数）
data_len = reference_trajectory.shape[1]  #获取参考轨迹的长度（即列数）

dmp = dmp_discrete(n_dmps=data_dim, n_bfs=1000, dt=1.0/data_len)   #创建一个离散的 DMP 模型，指定 DMP 的数量为参考轨迹的维度，基函数的数量为 1000，时间步长为轨迹长度的倒数。
dmp.learning(reference_trajectory)

reproduced_trajectory, _, _ = dmp.reproduce(tau=1.0, initial=[-0.6, 0.2, 0.40], goal=[-0.27, -0.42, 0.20])

# 保存新的轨迹到 CSV 文件中
reproduced_trajectory_df = pd.DataFrame(reproduced_trajectory)
reproduced_trajectory_df.to_csv('./reproduced_trajectory.csv', header=False, index=False)

# 记录结束时间
end_time = time.time()

# 计算并打印运行时间
elapsed_time = end_time - start_time
print(f"程序运行时间: {elapsed_time:.4f} 秒")