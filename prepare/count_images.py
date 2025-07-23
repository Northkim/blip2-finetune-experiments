import os

data_root = '/mnt/workspace/data/hmi_dataset'

total_count = 0
class_counts = {}

for class_name in os.listdir(data_root):
    class_dir = os.path.join(data_root, class_name)
    if os.path.isdir(class_dir):
        num_files = len([f for f in os.listdir(class_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))])
        class_counts[class_name] = num_files
        total_count += num_files

print(" 每个类的图片数量：")
for cls, count in class_counts.items():
    print(f"  {cls}: {count} 张")

print(f"\n 总图片数量: {total_count} 张")
