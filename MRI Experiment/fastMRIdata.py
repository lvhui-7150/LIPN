import os
import h5py

# ========================================================
# 1. 核心修改：请把这里改成你本地电脑上 singlecoil_val 文件夹的【绝对路径】
# 注意：前面加一个 r 可以防止 Windows 路径中的反斜杠 \ 发生转义错误
# ========================================================
data_dir = r"C:\Users\Administrator\Desktop\论文\变分不等式第二篇\数值实验\knee_singlecoil_val\knee_singlecoil_val\singlecoil_val"
# 如果你下载到了其他盘（比如 D 盘），请写成类似于：r"D:\datasets\knee_singlecoil_val\singlecoil_val"

# 检查路径是否存在
if not os.path.exists(data_dir):
    print(f"❌ 错误：路径不存在！请检查路径是否抄错了：\n{data_dir}")
else:
    # 2. 自动获取该目录下所有真实存在的 .h5 文件
    all_h5_files = [f for f in os.listdir(data_dir) if f.endswith('.h5')]

    if len(all_h5_files) == 0:
        print(f"❌ 错误：在文件夹中没有找到任何 .h5 文件，请确认数据是否解压到了该目录下。")
    else:
        print(f"✅ 成功找到 {len(all_h5_files)} 个数据文件。")

        # 3. 自动取第一个真实存在的文件，再也不会因为名字敲错而报错了
        first_real_file = all_h5_files[0]
        file_path = os.path.join(data_dir, first_real_file)

        print(f" 正在尝试打开文件: {file_path}\n" + "-" * 40)

        # 4. 打开并读取文件
        with h5py.File(file_path, 'r') as f:
            print("=== 文件内部的键 (Keys) ===")
            print(list(f.keys()))

            if 'kspace' in f:
                print("\nkspace 数据维度:", f['kspace'].shape)

            for img_key in ['reconstruction_esc', 'reconstruction_rss']:
                if img_key in f:
                    print(f"\n地面真值图像 ({img_key}) 维度:", f[img_key].shape)