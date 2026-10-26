1![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/06abbdd6a3584837a23c08ee5e4ed7bd.png)
1
![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/3c4c1c0910494dd2b62e968c8cdc927f.png)
1
无人机光伏板红外图像热斑缺陷数据集，类别为两种 0为热斑；1为热点；2为遮挡。训练+验证+测试共2w多张图片，标签已打好，可直接用于训练。

![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/1b865fb9e0f44bf6aa9a4184790524ff.png)
以下是 **光伏板红外图像热斑缺陷数据集** 的详细信息汇总表：

| 项目 | 内容描述 |
|------|--------|
| **数据集名称** | 光伏板红外热斑缺陷检测数据集（自建） |
| **图像类型** | 红外热成像图（灰度或伪彩色） |
| **总样本数量** | 超过 20,000 张（训练 + 验证 + 测试） |
| **标注状态** | ✅ 所有图像均已标注，可直接用于训练 |
| **任务类型** | 语义分割 或 目标检测（根据标签格式）（注：若为 YOLO 格式则为目标检测；若为 mask 则为分割） |
| **类别定义** | - `0`: **热斑（Hot Spot）** —— 局部异常发热，存在潜在故障- `1`: **热点（Hot Region）** —— 温度较高但未达故障程度的区域- `2`: **遮挡（Shading/Occlusion）** —— 由树叶、鸟粪、灰尘等引起的遮挡区域 |
| **类别数量** | 3 类 |
| **标签格式** | - 若用于 **目标检测**：YOLO 格式（`.txt` 文件，每行：`class_id x_center y_center width height`）- 若用于 **语义分割**：PNG 格式掩码（像素值 0/1/2 对应三类）（具体以实际提供为准） |
| **数据划分** | - **训练集（train）**：约 14,000 张（70%）- **验证集（val）**：约 4,000 张（20%）- **测试集（test）**：约 2,000+ 张（10%） |
| **图像分辨率** | 常见尺寸如 640×512、1024×768 等（依采集设备而定） |
| **配套文件** | - `data.yaml`（含路径与类别名）- `labels/` 与 `images/` 目录结构清晰- 可选：预划分的 `train.txt`, `val.txt`, `test.txt` |
| **适用模型** | YOLOv5/v8（检测）、U-Net、DeepLab、SegFormer（分割）等 |
| **应用场景** | 光伏电站智能巡检、无人机红外自动诊断、缺陷预警系统 |
| **优势特点** | - 规模大（>2w）- 类别覆盖典型缺陷类型- 标签质量高，可开箱训练- 包含遮挡干扰项，提升模型鲁棒性 |

> 💡 **使用提示**：  
> - 若用于 **YOLO 训练**，确保 `data.yaml` 中 `nc: 3` 且 `names: ['hotspot', 'hotregion', 'occlusion']`；  
> - 若用于 **分割**，需确认标签 PNG 的像素值严格为 0/1/2（无其他值）。
![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/eb2f6feb7359437ebd399cab514c6dd2.jpeg)
 **光伏板红外热斑缺陷数据集**（3 类：0=热斑，1=热点，2=遮挡；20,000+ 张已标注图像）的 **完整、可直接运行的训练代码**。  
假设你的数据集采用 **YOLO 目标检测格式**（即每张图对应一个 `.txt` 标签文件，含归一化边界框和类别 ID），并已按标准目录结构组织。

---

## ✅ 一、数据集目录结构（前提）

请确保你的数据集解压后结构如下：

```
pv_thermal_defect/
├── images/
│   ├── train/      # ~14,000 张 .jpg / .png
│   ├── val/        # ~4,000 张
│   └── test/       # ~2,000 张
├── labels/
│   ├── train/      # 对应标签 .txt
│   ├── val/
│   └── test/
└── pv_thermal.yaml
```

### 📄 `pv_thermal.yaml` 文件内容（必须正确）：
```yaml
# pv_thermal.yaml
train: ./images/train
val: ./images/val
test: ./images/test  # 可选，部分框架忽略

nc: 3
names: ['hotspot', 'hotregion', 'occlusion']
```

> 🔍 类别说明：
> - `hotspot`（0）：严重热斑，需预警  
> - `hotregion`（1）：高温区域，非故障但需关注  
> - `occlusion`（2）：遮挡物（如鸟粪、树叶），可能诱发热斑  

---

## 🧪 二、环境准备（Python + YOLOv8）

```bash
# 安装 PyTorch（以 CUDA 11.8 为例）
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# 安装 Ultralytics YOLO（支持 YOLOv8）
pip install ultralytics
```

---

## ▶️ 三、详细训练代码（Python 脚本版）

将以下代码保存为 `train_pv_thermal.py`，放在 `pv_thermal_defect/` 目录下：

```python
# train_pv_thermal.py
from ultralytics import YOLO
import os

# 可选：设置工作目录（若脚本不在数据集根目录）
# os.chdir('/path/to/pv_thermal_defect')

def main():
    # 1. 加载预训练模型（YOLOv8n 轻量高效，适合红外小目标）
    model = YOLO('yolov8n.pt')  # 可替换为 yolov8s.pt 提升精度

    # 2. 开始训练
    results = model.train(
        data='pv_thermal.yaml',          # 数据配置文件
        epochs=100,                      # 训练轮数
        imgsz=640,                       # 输入尺寸（红外图通常无需过大）
        batch=16,                        # 批大小（根据 GPU 显存调整，见下文）
        name='pv_thermal_yolov8n',       # 实验名称
        device=0,                        # GPU ID（0,1,... 或 'cpu'）
        workers=4,                       # 数据加载线程数
        cache=False,                     # 不缓存图像到内存（节省 RAM）
        patience=25,                     # 早停：mAP 25 轮无提升则停止
        save=True,                       # 保存 best/last 模型
        plots=True,                      # 自动生成 PR 曲线、混淆矩阵等
        project='runs/detect',           # 结果保存目录
        exist_ok=False                   # 若实验名重复是否覆盖
    )

    print(f"✅ 训练完成！结果保存在: {results.save_dir}")
    
    # 3. （可选）在测试集上评估
    metrics = model.val(data='pv_thermal.yaml', split='test')
    print(f"Test mAP@0.5: {metrics.box.map50:.4f}")

if __name__ == '__main__':
    main()
```

---

## ⚙️ 四、关键参数调优建议（防 OOM & 提升性能）

| 参数 | 推荐值 | 说明 |
|------|--------|------|
| `batch` | **8~32** | - 12GB GPU（如 3060）：`batch=16`- 8GB GPU（如 2070）：`batch=8`- 若 OOM，先设 `batch=4` |
| `imgsz` | **640** | 红外热斑通常为局部亮斑，640 足够；不建议 >1024 |
| `model` | `yolov8n` 或 `yolov8s` | 小目标多时，`yolov8s` 更佳；部署受限则用 `yolov8n` |
| `augment` | 默认开启 | YOLO 自动启用 HSV、翻转、mosaic 增强，适合红外图 |

> 💡 **若遇 OOM（显存不足）**，请参考：
> ```python
> batch=8,
> imgsz=512,
> cache=False,
> workers=2
> ```

---

## ▶️ 五、命令行训练方式（替代 Python 脚本）

如果你偏好命令行，可直接运行：

```bash
cd /path/to/pv_thermal_defect

yolo detect train \
    data=pv_thermal.yaml \
    model=yolov8n.pt \
    epochs=100 \
    imgsz=640 \
    batch=16 \
    name=pv_thermal_yolov8n \
    device=0 \
    plots=True
```

---

## 📊 六、训练结果输出位置

训练完成后，所有结果保存在：
```
runs/detect/pv_thermal_yolov8n/
├── weights/best.pt        # 最佳模型（按 val mAP 选择）
├── weights/last.pt        # 最终模型
├── results.csv            # 指标日志（loss, mAP 等）
├── results.png            # 训练曲线
├── pr_curve.png           # 各类 Precision-Recall 曲线
├── confusion_matrix.png
└── val_batch0_pred.jpg    # 验证集预测示例
```

---

## 🔍 七、推理与部署示例

```python
from ultralytics import YOLO

# 加载最佳模型
model = YOLO('runs/detect/pv_thermal_yolov8n/weights/best.pt')

# 单图推理
results = model.predict(
    source='images/test/0001.jpg',
    conf=0.3,          # 置信度阈值
    save=True,         # 保存带框图像
    show=False
)

# 批量推理整个测试集
model.predict(source='images/test/', save=True, project='output_test')
```

---

## 📌 八、注意事项

1. **标签格式验证**：确保 `.txt` 文件中类别 ID 仅为 `0`, `1`, `2`，且坐标归一化；
2. **图像格式**：支持 `.jpg`, `.png`，建议统一为 `.jpg` 节省空间；
3. **测试集评估**：YOLOv8 的 `val` 默认用 `val` 集，若要测 `test` 集需显式指定 `split='test'`；
4. **类别不平衡**：若“遮挡”样本极少，可在 `train.py` 中启用 `class_weights`（YOLOv8 暂未开放，可考虑过采样）。

---


