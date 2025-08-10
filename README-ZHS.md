## ComfyUI-S4Motion 节点说明

本仓库提供一组用于在 ComfyUI 中为前景图层制作动画的节点，包括旋转、位置、路径位置、缩放、透明度、形变、蒙版等 effectors，并由核心节点统一合成导出 APNG/WEBP 动画与帧序列。

### 安装与依赖

- 运行环境：Python 3.10+，ComfyUI
- 依赖（核心/可选）：
  - 必需：`numpy`、`Pillow`
  - 建议：`opencv-python`、`scikit-image`、`scipy`
  - 可选：`bezier`（更平滑的缓动曲线）

安装示例：

```bash
pip install numpy Pillow opencv-python scikit-image scipy bezier
```

首次运行核心节点会在日志中打印依赖检查结果，缺失项会提示安装命令。

### 节点类别

所有节点均位于 ComfyUI 类别：`💀S4Motion`

---

## 核心节点

### Motion Config (`py/motionConfig.py`)

- **功能**：统一时间轴、采样率与导出设置；接收各类 effector 的逐帧数据；对前景图层进行形变、蒙版、旋转、缩放、位置与透明度合成；输出动画文件与帧序列。
- **输入（必填）**：
  - `layer_image: IMAGE` 前景图（可为单帧或序列）
  - `background_image: IMAGE` 背景图（可为单帧或序列）
  - `time: FLOAT` 动画时长（秒）
  - `fps: INT` 帧率（1–60）
  - `loop: BOOLEAN` 是否往复循环
  - `format: webp | apng` 导出格式
  - `rotation: FLOAT` 基础旋转（度）
  - `position_x, position_y: FLOAT` 基础位置（像素）
  - `scale: FLOAT` 基础缩放
  - `opacity: FLOAT` 基础透明度（0–1）
- **输入（可选）**：
  - `rotation_effector: S4_ROTATION_EFFECTOR`
  - `position_effector: S4_POSITION_EFFECTOR`（支持参数/路径两种模式）
  - `scale_effector: S4_SCALE_EFFECTOR`
  - `opacity_effector: S4_OPACITY_EFFECTOR`
  - `distortion_effector: S4_DISTORTION_EFFECTOR`
  - `mask_effector: S4_MASK_EFFECTOR`
- **输出**：
  - `STRING` 导出文件路径（保存至仓库 `output/` 目录）
  - `IMAGE` 帧序列（torch tensor，帧维度在 batch 维，形如 `(F, H, W, C)`，范围 [0,1]）
- **合成规则要点**：
  - 形变（distortion）在旋转/缩放之前应用；蒙版在合成前应用到前景图层的 alpha。
  - 位置对齐：路径模式使用中心对齐（`center_align=True`），参数模式使用左上角对齐。
  - 位置、旋转、缩放、透明度均为“基础值 ×/＋ effector 值”的结果；长度不足会自动补齐到总帧数。
  - 坐标允许为负，图层可出界；旋转与缩放使用高质量重采样（小图小角度旋转带超采样优化）。
  - 保存：WEBP 使用无损；`loop=True` 时循环值为 0（无限循环），否则为 1。

---

## Effector 节点

### Motion Rotation (`py/motionRotation.py`)

- **功能**：输出逐帧旋转角度（度）。
- **主要输入**：`default_rotation`、`target_rotation`、`duration`、`delay`、`motion_curve`、`inverse`、`loop`
- **输出**：`S4_ROTATION_EFFECTOR`（可调用：`effector(total_frames, total_time) -> List[float]`）
- **说明**：支持 `linear / ease_in / ease_out / ease_in_out`，安装 `bezier` 时使用贝塞尔获得更平滑曲线。

### Motion Position (`py/motionPosition.py`)

- **功能**：输出逐帧位置（x, y）。
- **主要输入**：`default_x`、`default_y`、`target_x`、`target_y`、`duration`、`delay`、`motion_curve`、`inverse`、`loop`
- **输出**：`S4_POSITION_EFFECTOR`（`is_path_mode=False`，`effector(total_frames, total_time) -> List[(x,y)]`）

### Motion Position on Path (`py/motionPositionOnPath.py`)

- **功能**：从路径图像中提取路径骨架，按时序沿路径输出逐帧位置。
- **主要输入**：`path_image`、`duration`、`delay`、`motion_curve`、`inverse`、`loop`
- **输出**：`S4_POSITION_EFFECTOR`（`is_path_mode=True`）
- **路径提取流程（要点）**：
  - 将输入图像转为灰度并归一化；使用阈值 0.5 提取“较暗”路径区域。
  - 去噪与填洞后进行骨架化；选取最大连通域为主路径；按 8 邻域有序遍历形成连续路径。
  - 可对路径进行样条平滑与等距重采样至运动帧数；再按缓动曲线重映射时序。

### Motion Scale (`py/motionScale.py`)

- **功能**：输出逐帧缩放倍率。
- **主要输入**：`default_scale`、`target_scale`、`duration`、`delay`、`motion_curve`、`inverse`、`loop`
- **输出**：`S4_SCALE_EFFECTOR`
- **说明**：缩放下限为 0.01，避免无效或负值缩放。

### Motion Opacity (`py/motionOpacity.py`)

- **功能**：输出逐帧透明度（0–1）。
- **主要输入**：`default_opacity`、`target_opacity`、`duration`、`delay`、`motion_curve`、`inverse`、`loop`
- **输出**：`S4_OPACITY_EFFECTOR`

### Motion Distortion (`py/motionDistortion.py`)

- **功能**：输出逐帧形变规格，供核心节点进行图像形变。
- **主要输入**：
  - 时间轴：`mode`（`wave | swirl | radial`）、`duration`、`delay`、`motion_curve`、`inverse`、`loop`
  - Wave：`wave_amplitude_*`、`wave_wavelength_*`、`wave_phase_*`、`wave_direction(x|y|xy)`
  - Swirl：`swirl_strength_*`、`swirl_radius_*`、`swirl_center_x/y_*`（中心采用 0–1 相对坐标）
  - Radial：`radial_k_*`、`radial_center_x/y_*`（桶形/枕形）
- **输出**：`S4_DISTORTION_EFFECTOR`（每帧生成 `{mode, params}`）
- **核心节点中的执行**：
  - `wave`/`radial` 通过 OpenCV `remap`；`swirl` 通过 scikit-image `swirl`；均在旋转/缩放前处理。

### Motion Mask (`py/motionMask.py`)

- **功能**：提供基础蒙版图与逐帧蒙版变换规格，用于裁剪/淡入淡出等效果。
- **主要输入**：`mask(MASK)`、`duration`、`delay`、`motion_curve`、`inverse`、`loop`、以及 `x/y/rotation/scale/opacity` 的默认与目标值
- **输出**：`S4_MASK_EFFECTOR`（携带 `mask_image` 与逐帧 `{x,y,rotation,scale,opacity}`）
- **核心节点中的执行**：
  - 将基础蒙版转为 L 通道，按帧进行旋转/缩放/平移与不透明度处理。
  - 把变换后的蒙版贴到与图层同尺寸的画布上，再与图层 alpha 相乘，实现裁剪/遮罩。

---

## 基本用法（ComfyUI 工作流）

1. 放置 `Motion Config` 节点，设置 `time / fps / loop / format` 与基础的 `rotation/position/scale/opacity`。
2.（可选）按需添加各类 effector：`Motion Rotation / Position / Position on Path / Scale / Opacity / Distortion / Mask`，并连接到 `Motion Config` 对应输入端口。
3. 连接 `layer_image` 与 `background_image`（均支持单帧或图像序列）。
4. 运行后将：
   - 在仓库 `output/` 目录生成动画文件（WEBP/APNG）；
   - 输出一份帧序列张量，可继续送入其他 ComfyUI 节点。

### 提示与注意

- 输入支持 `PIL / numpy / torch`，会自动转换；维度过深会自动去批次维度。
- effector 返回长度不足时会自动补齐到总帧数；延迟 `delay` 会用首帧值填充。
- 路径模式为中心对齐；参数模式为左上角对齐。
- 未安装 `bezier` 时将回退到数学缓动函数。

---

## 目录结构

```
py/
  motionConfig.py              # 核心合成/导出
  motionRotation.py            # 旋转 effector
  motionPosition.py            # 位置 effector（参数）
  motionPositionOnPath.py      # 位置 effector（路径）
  motionScale.py               # 缩放 effector
  motionOpacity.py             # 透明度 effector
  motionDistortion.py          # 形变 effector（wave/swirl/radial）
  motionMask.py                # 蒙版 effector
```

如需补充示例工作流或演示图，请告知具体需求与素材偏好。


