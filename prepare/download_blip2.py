# download_blip2.py
from modelscope import snapshot_download

local_dir = snapshot_download(
    'goldsj/blip2-opt-2.7b',          # ModelScope 仓库名
    cache_dir='/mnt/workspace/models' # 你想存放的位置
)
print("模型已下载到：", local_dir)
