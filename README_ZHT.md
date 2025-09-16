# ComfyUI-S4Motion

**版本：1.2.0**

ComfyUI 的綜合動態動畫工具包，提供 11 個專業級動態控制節點，用於創建具有生產品質和可靠性的動態動畫。

## 🚀 功能特色

### 核心動態控制
- **💀Motion Config** - 動態動畫的中央配置中樞
- **💀Motion Position** - 具有平滑曲線的精確位置動畫
- **💀Motion Position On Path** - 複雜軌跡的路徑動態控制
- **💀Motion Rotation** - 具有可自定義軸心點的平滑旋轉動畫
- **💀Motion Scale** - 具有比例控制的動態縮放效果
- **💀Motion Opacity** - 透明度通道動畫淡入淡出效果

### 進階效果
- **💀Motion Distortion** - 專業變形效果（波浪、漩渦、徑向）
- **💀Motion Shake** - 逼真的震動和振動效果
- **💀Motion Mask** - 選擇性效果的動畫遮罩控制

### 視頻處理
- **💀Video Crop** - 支援動畫的精確視頻裁切
- **💀Video Frames** - 進階幀提取和處理

## 📦 安裝

### 方法一：ComfyUI Manager（建議）
1. 開啟 ComfyUI Manager
2. 搜尋 "S4Motion" 
3. 點擊安裝
4. 重啟 ComfyUI

### 方法二：手動安裝
1. 導航至您的 ComfyUI custom_nodes 目錄：
   ```
   cd ComfyUI/custom_nodes/
   ```
2. 複製此存儲庫：
   ```
   git clone https://github.com/S4MUEL/ComfyUI-S4Motion.git
   ```
3. 安裝依賴項：
   ```
   pip install -r ComfyUI-S4Motion/requirements.txt
   ```
4. 重啟 ComfyUI

## 🔧 依賴項

插件具有智慧依賴管理功能，具有優雅的降級機制：

### 核心依賴項（必需）
- **PyTorch** - 核心張量運算和 ComfyUI 相容性
- **Pillow** - 專業圖像處理和動畫幀
- **NumPy** - 動態曲線的高效數值計算

### 可選依賴項（增強功能）
- **OpenCV** - 視頻處理和進階變形效果
- **Scikit-image** - 高品質圖像變換
- **SciPy** - 進階動態曲線的科學計算
- **Bezier** - 專業動態曲線計算

所有依賴項會在啟動時自動驗證，具有生產品質的日誌記錄和降級機制。

## 📖 使用方法

1. **尋找節點**：所有 S4Motion 節點在 ComfyUI 節點瀏覽器中都以 💀 為前綴
2. **分類**：在「💀S4Motion」分類下查找
3. **生產就緒**：所有節點都包含全面的錯誤處理和日誌記錄
4. **範例**：查看 `examples/` 資料夾中的工作流程文件

### 快速入門範例
1. 添加 Motion Config 節點來建立動畫時間軸
2. 連接動態效果器節點（Position、Rotation、Scale 等）
3. 配置動態參數和曲線
4. 連接到您的圖像/視頻處理工作流程
5. 執行動畫

## 🎯 主要功能

- ✅ **生產品質** - 企業級錯誤處理和驗證
- ✅ **智慧依賴** - 具有降級功能的自動依賴管理
- ✅ **動態曲線** - 支援貝塞爾曲線的專業緩動
- ✅ **時間軸控制** - 具有延遲、持續時間和迴圈選項的精確定時
- ✅ **進階效果** - 波浪、漩渦和徑向變形功能
- ✅ **視頻支援** - 全面的視頻處理和幀控制
- ✅ **路徑動畫** - 支援精密動態的複雜軌跡

## 📁 專案結構

```
ComfyUI-S4Motion/
├── py/                    # 核心節點實現
│   ├── motionConfig.py    # 中央動態配置
│   ├── motionPosition.py  # 位置動畫控制
│   ├── motionRotation.py  # 旋轉效果
│   ├── motionScale.py     # 縮放動畫
│   ├── motionOpacity.py   # 透明度/alpha 控制
│   ├── motionPositionOnPath.py # 基於路徑的動態
│   ├── motionDistortion.py # 變形效果
│   ├── motionShake.py     # 震動/振動效果
│   ├── motionMask.py      # 動畫遮罩
│   ├── videoCrop.py       # 視頻裁切
│   └── videoFrames.py     # 幀處理
├── examples/              # 使用範例和工作流程
├── summary_md/            # 開發摘要和筆記
├── __init__.py           # 插件初始化
├── dependency_manager.py  # 依賴管理
└── requirements.txt      # Python 依賴項
```

## 🎨 動態曲線

S4Motion 支援專業緩動函數：
- **Linear** - 恆定速度動態
- **Ease In** - 漸進加速
- **Ease Out** - 漸進減速  
- **Ease In Out** - 平滑加速和減速

支援可選的貝塞爾曲線，提供超平滑的專業動畫。

## 🔄 動畫控制

每個動態節點都支援：
- **Duration** - 動畫長度（秒）
- **Delay** - 序列動畫的開始延遲
- **Inverse** - 反向動態以實現回到原點效果
- **Loop** - 無限動畫的連續重複

## 🤝 貢獻

歡迎貢獻！請隨時提交拉取請求或報告問題。

## 📜 許可證

此專案為開源專案。請尊重許可條款。

---

**作者：** S4MUEL  
**網站：** [s4muel.com](https://s4muel.com)  
**版本：** 1.2.0