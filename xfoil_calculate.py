import os
import sys
import subprocess
import multiprocessing
import glob
import numpy as np

XFOIL_PATH = r"D:\Program Files\XFOIL6.99\xfoil.exe"
DATA_DIR = r"D:\Project\IP\DCGAN_result\generated_airfoil_generator_best_20000\airfoil_coordinates\\Analysis"
RESULT_FILE = "simulation_result.csv"

REYNOLDS = 1000000
MACH = 0.01

# --- 修改 1: 定义攻角序列的起始、结束和步长 ---
ALPHA_START = -5
ALPHA_END = 12
ALPHA_STEP = 1

N_PROCESSES = 5

def preprocess_airfoil(file_path):
    file_name = os.path.basename(file_path) #提取文件名，不带路径
    airfoil_name = os.path.splitext(file_name)[0]
    
    with open(file_path, 'r') as f: #读取原始坐标文件
        lines = f.readlines()
        
    pts = []
    for line in lines:
        try:
            parts = line.strip().split()
            if len(parts) >= 2:
                pts.append([float(parts[0]), float(parts[1])])
        except ValueError:
            pass
            
    if len(pts) < 10:
        return
        
    pts = np.array(pts)
    le_idx = np.argmin(pts[:, 0]) #找到最前端的点，即前缘点索引
    
    branch1 = pts[:le_idx+1] #将翼型坐标分成两部分
    branch2 = pts[le_idx:]
    
    def clean_branch(arr): #按照x坐标排序并去除重复点
        arr = arr[np.argsort(arr[:, 0])]
        _, idx = np.unique(arr[:, 0], return_index=True)
        return arr[idx]
        
    b1 = clean_branch(branch1)
    b2 = clean_branch(branch2)
    
    if len(b1) < 2 or len(b2) < 2:
        return
        
    min_x = max(b1[0, 0], b2[0, 0])
    max_x = min(b1[-1, 0], b2[-1, 0])
    mid_x = 0.5 * (min_x + max_x)
    
    y1_mid = np.interp(mid_x, b1[:, 0], b1[:, 1])
    y2_mid = np.interp(mid_x, b2[:, 0], b2[:, 1])
    
    if y1_mid > y2_mid: #根据y值判断上下表面
        upper, lower = b1, b2
    else:
        upper, lower = b2, b1
        
    N = 80 
    beta = np.linspace(0, np.pi, N)
    dist = 0.5 * (1.0 - np.cos(beta))  #使用余弦分布重新参数化，使点在前缘和尾缘附近更密集
    
    xp_upper, yp_upper = upper[:, 0], upper[:, 1]
    x_new_upper = xp_upper[0] + (xp_upper[-1] - xp_upper[0]) * dist
    y_upper = np.interp(x_new_upper, xp_upper, yp_upper)
    
    xp_lower, yp_lower = lower[:, 0], lower[:, 1]
    x_new_lower = xp_lower[0] + (xp_lower[-1] - xp_lower[0]) * dist
    y_lower = np.interp(x_new_lower, xp_lower, yp_lower)
    
    with open(file_path, 'w') as f: #按照xfoil要求的格式重写文件
        f.write(f"{airfoil_name}\n") #第一行是翼型名称
        for i in range(len(x_new_upper)-1, -1, -1): #先写上表面坐标，从后缘到前缘
            f.write(f"{x_new_upper[i]:.6f} {y_upper[i]:.6f}\n")
        for i in range(1, len(x_new_lower)): #再写下表面坐标，从前缘到后缘
            f.write(f"{x_new_lower[i]:.6f} {y_lower[i]:.6f}\n")

def run_single_airfoil(file_name): #单个翼型文件调用xfoil计算的函数定义
    worker_id = multiprocessing.current_process().name
    temp_output = f"temp_polar_{worker_id}_{os.getpid()}.txt" 
    
    if os.path.exists(temp_output):
        os.remove(temp_output)
        
    # --- 修改 2: 更新 XFOIL 命令序列使用 ASEQ ---
    cmds = [
        f"LOAD {file_name}",     # 加载翼型文件
        "NORM",                  # 归一化坐标（弦长为1）
        "PANE",                  # 生成面网格
        "OPER",                  # 进入操作模式
        f"VISC {REYNOLDS}",      # 设置粘性计算和雷诺数
        f"M {MACH}",             # 设置马赫数
        "ITER 300",              # 设置最大迭代次数
        "PACC",                  # 开始数据积累
        f"{temp_output}",        # 输出文件
        "",                      # 空行（不输入描述）
        f"ASEQ {ALPHA_START} {ALPHA_END} {ALPHA_STEP}", # 攻角序列计算
        "QUIT"                   # 退出XFOIL
    ]
    
    cmd_str = "\n".join(cmds)
    
    try:
        process = subprocess.Popen( #运行xfoil进程
            [XFOIL_PATH],
            stdin=subprocess.PIPE,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            text=True,
            cwd=DATA_DIR 
        )
        process.communicate(input=cmd_str, timeout=30) #稍微增加超时时间，因为序列计算耗时更长
    except subprocess.TimeoutExpired:
        process.kill()
        
    # --- 修改 3: 提取所有收敛的攻角数据 ---
    airfoil_data = [] # 用于存储当前翼型的所有收敛结果
    
    if os.path.exists(temp_output):
        with open(temp_output, 'r') as f:
            lines = f.readlines()
            for line in lines:
                parts = line.split()
                # 确保行内有足够的数据且不是表头分隔符 (如 '----')
                if len(parts) > 5 and not "---" in line:
                    try:
                        val_alpha = float(parts[0])
                        cl = float(parts[1])
                        cd = float(parts[2])
                        airfoil_data.append((val_alpha, cl, cd))
                    except ValueError:
                        continue
        os.remove(temp_output) #清理临时文件
        
    return (file_name, airfoil_data)

def main(): #定义主函数
    if not os.path.exists(DATA_DIR):
        print(f"Directory not found: {DATA_DIR}")
        sys.exit(1)
        
    os.chdir(DATA_DIR)
    
    dat_files = glob.glob("*.dat") #获取所有.dat文件列表
    if not dat_files:
        print("No .dat files found.")
        sys.exit(1)
        
    for f in dat_files:
        preprocess_airfoil(f) #预处理所有翼型文件
        
    with multiprocessing.Pool(processes=N_PROCESSES) as pool: #并行计算
        results = pool.map(run_single_airfoil, dat_files)
        
    # --- 修改 4: 保存数据时支持多个攻角 ---
    with open(RESULT_FILE, 'w') as f:
        f.write("Filename,Alpha,Cl,Cd\n") # 添加 Alpha 列
        for name, data in results:
            if not data: # 如果该翼型没有任何收敛结果
                f.write(f"{name},NaN,NaN,NaN\n")
            else:
                for val_alpha, cl, cd in data:
                    f.write(f"{name},{val_alpha},{cl},{cd}\n")

if __name__ == "__main__":
    multiprocessing.freeze_support()
    main()