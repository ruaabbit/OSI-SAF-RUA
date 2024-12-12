"""
Author: 爱吃菠萝 1690608011@qq.com
Date: 2024-09-24 21:40:18
LastEditors: 爱吃菠萝 1690608011@qq.com
LastEditTime: 2024-09-27 15:42:31
FilePath: /root/OSI-SAF/view.py
Description: 

Copyright (c) 2024 by 1690608011@qq.com, All Rights Reserved. 
"""

import xarray

dataset = xarray.open_dataset(
    "/root/autodl-tmp/OSI-SAF/1993/01/ice_conc_nh_ease2-250_cdr-v3p0_199301021200.nc"
)
print(dataset["ice_conc"])

# time = np.array(dataset["time"][:])

# sic_pred_SICFN = np.load("test_results/sic_pred_SICFN.npy")
# sic_pred_SICFN = sic_pred_SICFN[0, :, 0, :]

# times = np.load("test_results/times.npy")

# pred_length = times.shape[1] // 2
# pred_times = times[0, pred_length:]
# pred_start_time = pred_times[0]
# pred_end_time = pred_times[-1]

# indices = np.where(np.in1d(time, pred_times))[0]  # 找到pred_times在time中的位置索引

# # 将预测结果替换到指定索引位置的数据中
# dataset["imgs"][indices] = sic_pred_SICFN

# # 将修改后的数据写回到原始文件
# dataset.to_netcdf(f"/root/autodl-tmp/full_sic_{pred_start_time}-{pred_end_time}.nc")
