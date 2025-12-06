import warnings
from ultralytics import YOLO
import os  # 导入os库用于处理文件名

# 忽略不必要的警告
warnings.filterwarnings('ignore')

if __name__ == '__main__':

    model_configs = [
        'DART-YOLO.yaml',

    ]

    # 2. 使用 for 循环遍历列表，依次训练每个模型
    for config_file in model_configs:
        print(f"\n{'=' * 20}")
        print(f"开始训练模型: {config_file}")
        print(f"{'=' * 20}\n")

        # 每次循环都初始化一个新的模型
        model = YOLO(config_file)

        # 从配置文件名生成一个独特的训练名称，例如 'yolo11_WTConv.yaml' -> 'yolo11_WTConv_train'
        # os.path.splitext(config_file)[0] 会获取文件名（不带.yaml后缀）
        training_name = f"{os.path.splitext(config_file)[0]}_train"

        # 调用训练方法
        model.train(
            data=r"data.yaml",
            cache=False,
            imgsz=640,
            epochs=200,
            single_cls=False,
            batch=16,
            close_mosaic=0,
            workers=0,
            device='0',
            optimizer='SGD',
            amp=True,
            project='runs/VIS',
            name=training_name,  # 关键：为每次训练设置一个唯一的名称
            # resume='path/to/last.pt', # 如果需要续训，请在这里为每个模型单独处理
        )