# YOLO 训练数据自动预标注工具

本工具用于给未标注图片批量生成 YOLO detection 格式的预标注 `.txt` 文件，方便后续人工检查、修正和训练 YOLO 模型。它不是最终真值标注工具，自动标注结果必须人工检查。

## 类别定义

类别顺序固定如下，不要和项目里旧的两类或旧顺序 `data.yaml` 混用：

```text
0 traffic_cone
1 yellow_car
2 exit
```

## 安装依赖

```bash
pip install -r requirements_autolabel.txt
```

默认开放词汇检测模型为 HuggingFace `IDEA-Research/grounding-dino-tiny`。首次运行需要联网下载模型权重；如果环境不能联网，请提前手动下载模型到本地缓存，或使用 `--model_id` 指向本地模型目录。

## 自动标注

```bash
python tools/auto_label_yolo.py \
  --image_dir dataset_raw/images \
  --label_dir dataset_raw/labels \
  --vis_dir dataset_raw/vis \
  --box_threshold 0.35 \
  --text_threshold 0.25 \
  --save_empty
```

如果漏标较多，可以降低阈值：

```bash
python tools/auto_label_yolo.py \
  --image_dir dataset_raw/images \
  --label_dir dataset_raw/labels \
  --vis_dir dataset_raw/vis_low_threshold \
  --box_threshold 0.25 \
  --text_threshold 0.20 \
  --overwrite
```

如果误标较多，可以提高阈值：

```bash
python tools/auto_label_yolo.py \
  --image_dir dataset_raw/images \
  --label_dir dataset_raw/labels \
  --vis_dir dataset_raw/vis_high_threshold \
  --box_threshold 0.45 \
  --text_threshold 0.30 \
  --overwrite
```

已有 YOLO 权重也可以作为可选预标注来源：

```bash
python tools/auto_label_yolo.py \
  --image_dir data/train \
  --label_dir data/autolabels \
  --vis_dir data/autolabels_vis \
  --use_yolo_model models/yolo/phantom_cone_exit_best.pt
```

注意：已有 YOLO 模型的类别名必须能映射到 `traffic_cone`、`yellow_car`、`exit`。如果旧模型类别顺序不同，请重点检查输出标签。

## 可视化检查

```bash
python tools/visualize_labels.py \
  --image_dir dataset_raw/images \
  --label_dir dataset_raw/labels \
  --vis_dir dataset_raw/vis_check
```

也可以抽样检查：

```bash
python tools/visualize_labels.py \
  --image_dir dataset_raw/images \
  --label_dir dataset_raw/labels \
  --vis_dir dataset_raw/vis_check \
  --shuffle \
  --max_images 100
```

建议先抽查 50-100 张图片，再决定是否调整阈值或提示词。

## 划分数据集

```bash
python tools/split_dataset.py \
  --image_dir dataset_raw/images \
  --label_dir dataset_raw/labels \
  --output_dir dataset \
  --val_ratio 0.2 \
  --seed 42
```

输出结构：

```text
dataset/
  images/
    train/
    val/
  labels/
    train/
    val/
  data.yaml
```

生成的 `data.yaml` 类别固定为：

```yaml
path: ./dataset
train: images/train
val: images/val

names:
  0: traffic_cone
  1: yellow_car
  2: exit
```

## YOLO 训练示例

```bash
yolo detect train data=dataset/data.yaml model=yolo11n.pt epochs=100 imgsz=640 batch=16
```

## 常见问题

**漏标多怎么办？**  
降低 `--box_threshold` 和 `--text_threshold`，例如 `0.25/0.20`，并重新可视化检查。交通锥通常是小目标，要重点检查远处、边缘、遮挡的锥桶。

**误标多怎么办？**  
提高阈值，例如 `0.45/0.30`。对 `yellow_car` 要特别注意，不要把普通黄色物体、黄色墙面、黄色标识误标成车。

**没有 GPU 怎么办？**  
使用 `--device cpu` 可以运行，但速度会慢很多。建议先用 `--max_images 50` 小批量测试阈值，再跑完整数据。

**出口类别边界如何统一？**  
项目组必须提前统一规则。若出口是标志牌，就只框出口标志牌；若出口是门洞、通道或闸门，就框整个可通行区域。规则不一致会导致 YOLO 难以收敛。

**为什么自动标注后仍需要人工检查？**  
开放词汇模型输出的是预标注，不是最终真值。它可能漏掉小交通锥、把黄色物体误认为车辆，或对出口边界理解不一致。训练前必须人工抽查和修正，尤其要保证类别顺序和边界定义一致。
