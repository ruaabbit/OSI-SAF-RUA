#!/bin/bash
###
 # @Author: 爱吃菠萝 zhangjia_liang@foxmail.com
 # @Date: 2023-10-20 23:16:39
 # @LastEditors: 爱吃菠萝 1690608011@qq.com
 # @LastEditTime: 2024-09-27 15:28:11
 # @FilePath: /root/OSI-SAF/data/gen_data_path.sh
 # @Description: 这是默认设置,请设置`customMade`, 打开koroFileHeader查看配置 进行设置: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AE
### 

DATA_DIR=/root/OSI-SAF-RUA/data
TEXTFILE=data_path.txt

# 清空或创建新的 data_path.txt 文件
> $TEXTFILE

for year in `ls $DATA_DIR`
do
    if [ -d "${DATA_DIR}/${year}" ]; then
        for month in `ls ${DATA_DIR}/${year}`
        do
            if [ -d "${DATA_DIR}/${year}/${month}" ]; then
                for datafile in `ls ${DATA_DIR}/${year}/${month}`
                do
                    echo ${DATA_DIR}/${year}/${month}/$datafile >> $TEXTFILE
                done
            fi
        done
    fi
done