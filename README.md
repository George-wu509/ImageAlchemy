# --- 檔案名稱: README.md ---
<div align="center">
  <h1>ImageAlchemy</h1>
  <p>
    一個基於 Stable Diffusion、ControlNet 及其他頂尖 AI 模型的進階 Python 圖像處理、增強與生成函式庫。
  </p>
  <p>
    <a href="https://github.com/[your-username]/ImageAlchemy/blob/main/LICENSE"><img alt="License" src="https://img.shields.io/github/license/[your-username]/ImageAlchemy?style=flat-square"></a>
    <a href="https://github.com/[your-username]/ImageAlchemy/stargazers"><img alt="Stars" src="https://img.shields.io/github/stars/[your-username]/ImageAlchemy?style=social"></a>
    <a href="https://github.com/[your-username]/ImageAlchemy/actions/workflows/python-package.yml"><img alt="CI" src="https://img.shields.io/github/actions/workflow/status/[your-username]/ImageAlchemy/python-package.yml?branch=main&style=flat-square"></a>
  </p>
</div>

---

**ImageAlchemy** 為複雜的圖像操作任務提供了一個簡單、高階的 API，這些任務通常需要深厚的生成式 AI 專業知識。它將最先進的模型抽象化為直觀的單行指令，讓開發者與研究人員能輕易地進行進階的圖像編輯。

## ✨ 功能亮點

ImageAlchemy 將複雜的生成式流程封裝成強大且易於使用的函式。

-   **圖像增強與修復:**
    -   `Denoise`: 去除照片中的噪點。
    -   `Sharpen` / `Deblur`: 校正失焦或模糊的圖像。
    -   `Super-Resolution`: 使用生成式放大技術增加圖像尺寸與細節。
    -   `Dehaze`: 去除風景照中的霧或霾。
    -   `Colorize`: 為黑白圖像添加自然的色彩。
    -   `Correct Light` / `Fix White Balance`: 透過提示詞調整光照與白平衡。
    -   `Apply HDR`: 生成高動態範圍（HDR）效果。
-   **物件與場景操作 (由 SAM 驅動):**
    -   `Inpaint`: 重建圖像中缺失或損壞的部分。
    -   `Remove Object`: 使用邊界框無縫地移除場景中的物件。
    -   `Add Object`: 根據文字提示在指定區域添加新物件。
    -   `Reposition Object`: 將物件從一個位置移動到另一個位置。
-   **生成式魔法:**
    -   `Generate Background`: 將圖像的背景替換為生成的場景。
    -   `Generative Zoom`: 透過迭代式外繪（outpainting）創造「無限變焦」效果。
    -   `Style Transfer`: 使用基於提示詞的引導來改變圖像的藝術風格。

## 🚀 快速開始

首先，設定專案並安裝其依賴項。

### 1. 安裝

**使用 Conda (推薦，便於 CUDA 管理):**
```bash
# 複製儲存庫
git clone [https://github.com/](https://github.com/)[your-username]/ImageAlchemy.git
cd ImageAlchemy

# 從提供的檔案建立 Conda 環境
conda env create -f environment.yml
conda activate image_alchemy

# 以可編輯模式安裝函式庫
pip install -e .
```

**僅使用 `pip`:**
```bash
git clone [https://github.com/](https://github.com/)[your-username]/ImageAlchemy.git
cd ImageAlchemy
pip install .
```

### 2. 使用範例

以下範例展示了如何輕鬆地放大圖像，然後移除一個不需要的物件。

```python
from PIL import Image
from image_alchemy import ImageAlchemy
from image_alchemy.utils.visualization import compare_images

# 初始化 alchemy 引擎 (首次執行時會下載模型)
# 使用 'cuda' 代表 GPU, 'cpu' 代表 CPU。
engine = ImageAlchemy(device='cuda')

# 載入您的圖片
input_image = Image.open("path/to/your/image.jpg")

# 1. 使用超解析度增強圖片
print("正在執行超解析度...")
sr_image = engine.enhancement.super_resolution(
    image=input_image,
    scale=2,
    prompt="a high-resolution, ultra-detailed photograph"
)

# 2. 從放大後的圖片中移除一個物件（使用邊界框）
# 函式庫內部會使用 SAM 來建立精確的遮罩。
print("正在移除物件...")
object_bounding_box = [250, 300, 450, 500] # [x1, y1, x2, y2]

final_image = engine.manipulation.remove_object(
    image=sr_image,
    mask=object_bounding_box,
    prompt="a beautiful landscape photograph, professional, 8k"
)

# 視覺化比較變更
compare_images(input_image, final_image, before_text="原始圖片", after_text="最終結果")

# 儲存結果
final_image.save("final_output.png")
print("最終圖片已儲存至 final_output.png")
```
請參閱 `examples/` 目錄以獲得更詳細的腳本。

## 🛠️ 專案架構

本函式庫遵循現代 Python 套件標準，包括：
- **`src` 佈局**: 為了清晰地分離原始碼與專案檔案，避免匯入衝突。
- **延遲載入 (Lazy Loading)**: AI 模型僅在首次呼叫其特定功能時才被載入到記憶體中，以優化資源使用。
- **模組化設計**: 功能被邏輯地劃分為 `enhancement`（增強）、`manipulation`（操作）和 `generative`（生成）模組。

## 🤝 貢獻

歡迎各種貢獻！請 fork 此儲存庫，進行您的修改，然後提交一個 pull request。對於重大變更，請先開一個 issue 來討論您想要改變的內容。

## 📜 授權

本專案採用 MIT 授權 - 詳情請參閱 [LICENSE](LICENSE) 檔案。