import os
import cv2
import numpy as np
import random
import shutil
from collections import defaultdict
from tqdm import tqdm

# ================= 导师的硬核配置区 =================
# 原始数据源
DIR_2500_IMG = r"C:\Users\Administrator\Desktop\Dataset\data2500\images"
DIR_2500_MSK = r"C:\Users\Administrator\Desktop\Dataset\data2500\masks"
DIR_1242_IMG = r"C:\Users\Administrator\Desktop\Dataset1\data1242_1080\images"
DIR_1242_MSK = r"C:\Users\Administrator\Desktop\Dataset1\data1242_1080\masks" # 5分类PNG路径

# 终极挂载目录
OUTPUT_ROOT = r"C:\Users\Administrator\Desktop\PIDNet_Power_Dataset"

# 离线滑窗参数
CROP_SIZE = 512
OVERLAP_RATIO = 0.10
STRIDE = int(512 * (1 - 0.10))
EMPTY_KEEP_PROB = 0.01

random.seed(2026) 
# ===================================================

def analyze_mask(mask_path):
    """动态类别嗅探器：生成多类别组合指纹"""
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    if mask is None: return None
    vals = np.unique(mask)
    targets = sorted([int(v) for v in vals if v not in [0, 255]])
    
    if len(targets) == 0:
        return "empty"
    else:
        return "_".join(map(str, targets))

def create_dirs(root):
    if os.path.exists(root):
        shutil.rmtree(root)
    for split in ['train', 'val', 'test']:
        os.makedirs(os.path.join(root, 'images', split))
        os.makedirs(os.path.join(root, 'masks', split))
        os.makedirs(os.path.join(root, 'mother_images', split))
        os.makedirs(os.path.join(root, 'mother_masks', split))
    os.makedirs(os.path.join(root, 'list'))

def process_and_crop(img_path, mask_path, split, is_4k, scene_id):
    """阶段3核心：母图备份与滑窗切分"""
    out_img_dir = os.path.join(OUTPUT_ROOT, 'images', split)
    out_msk_dir = os.path.join(OUTPUT_ROOT, 'masks', split)
    
    # 1. 备份母图
    ext = os.path.splitext(img_path)[1]
    shutil.copy2(img_path, os.path.join(OUTPUT_ROOT, 'mother_images', split, f"{scene_id}{ext}"))
    shutil.copy2(mask_path, os.path.join(OUTPUT_ROOT, 'mother_masks', split, f"{scene_id}.png"))
    
    generated_lines = []
    
    # 2. 如果是 512 图，直接拷贝并返回
    if not is_4k:
        new_name = f"{scene_id}_base.png"
        shutil.copy2(img_path, os.path.join(out_img_dir, new_name))
        shutil.copy2(mask_path, os.path.join(out_msk_dir, new_name))
        return [f"images/{split}/{new_name} masks/{split}/{new_name}"]
    
    # 3. 如果是 4K 图，执行滑窗
    img = cv2.imread(img_path)
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    h, w = img.shape[:2]
    generated_positions = set()

    for y in range(0, h - 512 + 1, STRIDE):
         for x in range(0, w - 512 + 1, STRIDE):
            y_s = y if y + CROP_SIZE <= h else h - CROP_SIZE
            x_s = x if x + CROP_SIZE <= w else w - CROP_SIZE

            if (y_s, x_s) in generated_positions:
                continue
            generated_positions.add((y_s, x_s))

            msk_crop = mask[y_s:y_s+CROP_SIZE, x_s:x_s+CROP_SIZE]
            vals = np.unique(msk_crop)
            targets = [v for v in vals if v not in [0, 255]]

            # 纯背景空图过滤逻辑
            if len(targets) == 0 and random.random() > EMPTY_KEEP_PROB:
                continue

            img_crop = img[y_s:y_s+CROP_SIZE, x_s:x_s+CROP_SIZE]
            crop_name = f"{scene_id}_y{y_s}_x{x_s}.png"
            cv2.imwrite(os.path.join(out_img_dir, crop_name), img_crop)
            cv2.imwrite(os.path.join(out_msk_dir, crop_name), msk_crop)
            generated_lines.append(f"images/{split}/{crop_name} masks/{split}/{crop_name}")

    return generated_lines


if __name__ == "__main__":
    print("=== 阶段 1：组合筛选与双数据源动态建档 ===")
    datasets = {
        "DS_2500": {"imgs": sorted(os.listdir(DIR_2500_IMG)), "img_dir": DIR_2500_IMG, "msk_dir": DIR_2500_MSK, "is_4k": False},
        "DS_1242": {"imgs": sorted(os.listdir(DIR_1242_IMG)), "img_dir": DIR_1242_IMG, "msk_dir": DIR_1242_MSK, "is_4k": True}
    }

    # 数据结构: categorized_data[数据集来源][组合指纹] = [母图列表]
    categorized_data = {
        "DS_2500": defaultdict(list),
        "DS_1242": defaultdict(list)
    }
    global_scene_idx = 1

    for ds_name, ds_info in datasets.items():
        print(f"正在扫描 {ds_name}...")
        for img_name in tqdm(ds_info["imgs"], desc="扫描进度"):
            if not img_name.lower().endswith(('.jpg', '.png', '.jpeg')): continue
            img_path = os.path.join(ds_info["img_dir"], img_name)
            mask_name = os.path.splitext(img_name)[0] + ".png"
            mask_path = os.path.join(ds_info["msk_dir"], mask_name)
            
            if not os.path.exists(mask_path): continue
            
            cat = analyze_mask(mask_path)
            if cat:
                scene_id = f"scene_{global_scene_idx:04d}"
                categorized_data[ds_name][cat].append((img_path, mask_path, ds_info["is_4k"], scene_id))
                global_scene_idx += 1

    print("\n=== 阶段 2：基于数据源与组合基因的母图 8:1:1 独立分配 ===")
    create_dirs(OUTPUT_ROOT)
    
    # 存储分好类的母图列表: mother_splits[split_name] = [母图信息列表]
    mother_splits = {"train": [], "val": [], "test": []}

    for ds_name, ds_cats in categorized_data.items():
        print(f"\n--- {ds_name} 内部切分 ---")
        for category, items in ds_cats.items():
            total = len(items)
            print(f"指纹 [{category}]: {total} 张母图", end=" -> ")
            if total == 0: continue
            
            random.shuffle(items)
            # 小样本保护机制
            if total < 5:
                n_train, n_val = total, 0
                print("触发保护(全入Train)")
            else:
                n_train = int(total * 0.8)
                n_val = int(total * 0.1)
                print("正常 8:1:1 划分")
                
            mother_splits["train"].extend(items[:n_train])
            mother_splits["val"].extend(items[n_train:n_train+n_val])
            mother_splits["test"].extend(items[n_train+n_val:])

    print("\n=== 阶段 3：执行滑窗切分与物理落盘 (需等待几分钟) ===")
    split_records = {"train": [], "val": [], "test": []}
    
    for split_name in ["train", "val", "test"]:
        print(f"正在处理 {split_name} 集合...")
        for img_p, msk_p, is_4k, scene_id in tqdm(mother_splits[split_name], desc=f"切片 {split_name}"):
            lines = process_and_crop(img_p, msk_p, split_name, is_4k, scene_id)
            split_records[split_name].extend(lines)

    print("\n=== 阶段 4：写入 PIDNet 标准索引文件 ===")
    for split in ["train", "val", "test"]:
        with open(os.path.join(OUTPUT_ROOT, 'list', f'{split}.txt'), 'w') as f:
            for line in split_records[split]:
                f.write(line + "\n")

    print(f"\n🎉 完美多维解耦闭环！")
    print(f"👉 Train 切片: {len(split_records['train'])} 张")
    print(f"👉 Val   切片: {len(split_records['val'])} 张")
    print(f"👉 Test  切片: {len(split_records['test'])} 张")