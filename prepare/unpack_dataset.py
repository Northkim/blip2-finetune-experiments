from modelscope.msdatasets import MsDataset
import os

output_dir = '/mnt/workspace/data/hmi_dataset'
os.makedirs(output_dir, exist_ok=True)

print("Loading dataset...")
ds = MsDataset.load('Northkim/archive_processed_data_by_class', split='train')

print("开始保存图片文件...")
for i, item in enumerate(ds):
    image = item['jpg']
    key = item['__key__']  # 例如 './eat_drink_image_17908'

    # 解析 label 和 image_id
    key = key.replace('./', '')  # 'eat_drink_image_17908'
    parts = key.split('_image_')
    if len(parts) == 2:
        label, img_id = parts
    else:
        label = 'unknown'
        img_id = key

    # 按照 label 分类保存
    label_dir = os.path.join(output_dir, label)
    os.makedirs(label_dir, exist_ok=True)

    save_path = os.path.join(label_dir, f"{img_id}.jpg")
    image.save(save_path)

    if (i+1) % 1000 == 0:
        print(f"已保存 {i+1} 张图片...")

print(f" All done! 按 label 分类保存完毕！")

