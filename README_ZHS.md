# ComfyUI-S4Motion

**版本：1.4.0**

ComfyUI 的综合动态动画工具包，提供 12 个专业级动态控制节点，用于创建具有生产品质和可靠性的动态动画。

## 🚀 功能特色

### 核心动态控制
- **💀Motion Config** - 动态动画的中央配置中枢
- **💀Motion Position** - 具有平滑曲线的精确位置动画
- **💀Motion Position On Path** - 复杂轨迹的路径动态控制
- **💀Motion Rotation** - 具有可自定义轴心点的平滑旋转动画
- **💀Motion Scale** - 具有比例控制的动态缩放效果
- **💀Motion Opacity** - 透明度通道动画淡入淡出效果

### 高级效果
- **💀Motion Distortion** - 专业变形效果（波浪、漩涡、径向）
- **💀Motion Shake** - 逼真的震动和振动效果
- **💀Motion Mask** - 选择性效果的动画遮罩控制

### 视频处理
- **💀Video Crop** - 支持动画的精确视频裁剪
- **💀Video Frames** - 高级帧提取和处理
- **💀Video Combine** - 在时间上连接两个视频或图像序列

## 📦 安装

### 方法一：ComfyUI Manager（推荐）
1. 打开 ComfyUI Manager
2. 搜索 "S4Motion" 
3. 点击安装
4. 重启 ComfyUI

### 方法二：手动安装
1. 导航至您的 ComfyUI custom_nodes 目录：
   ```
   cd ComfyUI/custom_nodes/
   ```
2. 复制此存储库：
   ```
   git clone https://github.com/S4MUEL-404/ComfyUI-S4Motion.git
   ```
3. 安装依赖项：
   ```
   pip install -r ComfyUI-S4Motion/requirements.txt
   ```
4. 重启 ComfyUI

## 🔧 依赖项

插件具有智能依赖管理功能，具有优雅的降级机制：

### 核心依赖项（必需）
- **PyTorch** - 核心张量运算和 ComfyUI 兼容性
- **Pillow** - 专业图像处理和动画帧
- **NumPy** - 动态曲线的高效数值计算

### 可选依赖项（增强功能）
- **OpenCV** - 视频处理和高级变形效果
- **Scikit-image** - 高品质图像变换
- **SciPy** - 高级动态曲线的科学计算
- **Bezier** - 专业动态曲线计算

所有依赖项会在启动时自动验证，具有生产品质的日志记录和降级机制。

## 📖 使用方法

1. **查找节点**：所有 S4Motion 节点在 ComfyUI 节点浏览器中都以 💀 为前缀
2. **分类**：在「💀S4Motion」分类下查找
3. **生产就绪**：所有节点都包含全面的错误处理和日志记录
4. **示例**：查看 `examples/` 文件夹中的工作流程文档

### 快速入门示例
1. 添加 Motion Config 节点来建立动画时间轴
2. 连接动态效果器节点（Position、Rotation、Scale 等）
3. 配置动态参数和曲线
4. 连接到您的图像/视频处理工作流程
5. 执行动画

## 🎯 主要功能

- ✅ **生产品质** - 企业级错误处理和验证
- ✅ **智能依赖** - 具有降级功能的自动依赖管理
- ✅ **动态曲线** - 支持贝塞尔曲线的专业缓动
- ✅ **时间轴控制** - 具有延迟、持续时间和循环选项的精确定时
- ✅ **高级效果** - 波浪、漩涡和径向变形功能
- ✅ **视频支持** - 全面的视频处理和帧控制
- ✅ **路径动画** - 支持精密动态的复杂轨迹

## 📁 项目结构

```
ComfyUI-S4Motion/
├── py/                    # 核心节点实现
│   ├── motionConfig.py    # 中央动态配置
│   ├── motionPosition.py  # 位置动画控制
│   ├── motionRotation.py  # 旋转效果
│   ├── motionScale.py     # 缩放动画
│   ├── motionOpacity.py   # 透明度/alpha 控制
│   ├── motionPositionOnPath.py # 基于路径的动态
│   ├── motionDistortion.py # 变形效果
│   ├── motionShake.py     # 震动/振动效果
│   ├── motionMask.py      # 动画遮罩
│   ├── videoCrop.py       # 视频裁剪
│   ├── videoFrames.py     # 帧处理
│   └── videoCombine.py    # 视频组合
├── examples/              # 使用示例和工作流程
├── summary_md/            # 开发摘要和笔记
├── __init__.py           # 插件初始化
├── dependency_manager.py  # 依赖管理
└── requirements.txt      # Python 依赖项
```

## 🎨 动态曲线

S4Motion 支持专业缓动函数：
- **Linear** - 恒定速度动态
- **Ease In** - 渐进加速
- **Ease Out** - 渐进减速  
- **Ease In Out** - 平滑加速和减速

支持可选的贝塞尔曲线，提供超平滑的专业动画。

## 🔄 动画控制

每个动态节点都支持：
- **Duration** - 动画长度（秒）
- **Delay** - 序列动画的开始延迟
- **Inverse** - 反向动态以实现回到原点效果
- **Loop** - 无限动画的连续重复

## 🤝 贡献

欢迎贡献！请随时提交拉取请求或报告问题。

## 📜 许可证

此项目为开源项目。请尊重许可条款。

---

**作者：** S4MUEL  
**网站：** [s4muel.com](https://s4muel.com)  
**版本：** 1.4.0
