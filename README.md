<div align="center">
  
# 🧥 Next-Gen AI Fashion Advisor
A high-accuracy, full-stack recommendation engine and retail analytics platform bridging the gap between computer vision and the second-hand fashion marketplace.
  
![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white)
![FastAPI](https://img.shields.io/badge/FastAPI-005571?style=for-the-badge&logo=fastapi)
![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)
![Transformers](https://img.shields.io/badge/HuggingFace-FFD21E?style=for-the-badge&logo=huggingface&logoColor=black)

</div>

<br>

Welcome to the **Second-Hand Fashion Advisor AI**. This application utilizes deep learning to identify apparel categories from raw images and augments the identification with intelligent retail logic. Whether you're trying to price a thrifted jacket or translate your physical measurements to luxury brand sizing, this tool has you covered.

## ✨ Core Features

* 🧠 **High-Fidelity Image Classification**: Fine-tuned on over 44,000 fashion items across 142 distinct categories. Currently operating at **99.30%+ test accuracy**.
* 💸 **Fair Resale Estimator**: Uses a robust logic engine to calculate secondary market valuations based on real-world condition depreciation matrices across fast fashion (Zara, H&M) and luxury (Gucci) domains.
* 📏 **Cross-Brand Size Translation**: Takes user biometric measurements (Chest, Waist, Hip) and translates them natively into perfect fit recommendations across leading retail brands.
* ⚡ **Ultra-Fast Visual Recommendations**: Leveraging FAISS indexing to instantly query the multi-GB image space for matching semantic fashion styles.
* 🎨 **Premium UI**: Operates via a beautifully animated, ultra-modern Glassmorphism frontend with built-in shimmering skeleton data loaders.

---

## 🚀 Quick Start Local Deployment

Everything is natively wired through FastAPI serving static frontend assets. 

### 1. Prerequisites
Ensure you have Python 3.9+ installed and clone the repository.
```bash
git clone https://github.com/nitishhansa1/Second-Hand-Fashion-Advisor-Ai-Model-.git
cd Second-Hand-Fashion-Advisor-Ai-Model-
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```
*(Optionally requires `faiss-cpu` / `faiss-gpu`, `torch`, `torchvision`, `transformers` for training pipelines)*

### 3. Start the API Server
Simply boot the web interface and the inference engine using Uvicorn.
```bash
uvicorn main:app --host 0.0.0.0 --port 8000
```

### 4. Interact
Open your browser and navigate to:
**👉 `http://localhost:8000`**

---

## 🛠 Project Architecture

| File/Directory | Purpose |
|------------------|---------|
| `main.py` | FastAPI application, endpoints `/predict`, `/estimate-resale`, `/translate-size`. |
| `frontend/` | Next-gen Glassmorphism frontend files (`index.html`, `script.js`, `styles.css`). |
| `model.pth` | Core PyTorch trained weights file for inference. |
| `classes.pkl` & `image_paths.pkl` | Dataset structural mapping allowing O(1) inference matching. |
| `evaluate.py` | Standalone script for evaluating model accuracy over the 44,400+ archive dataset. |
| `FashionAdvisor.ipynb` | The original raw Jupyter runtime environments for data processing / DINOv2 integration. |

## 📊 Dataset Requirements
Because of GitHub's strict file limitations, if you clone this repo locally, ensure your high-res fashion images sit inside an `archive/categorized_images/` directory so that `image_paths.pkl` correctly hooks into your local filesystem. The core architecture uses the huge open-source Myntra/Kaggle datasets to generate the prediction mapping logic.

---

<div align="center">
  <i>Built with ❤️ for deep learning and sustainable second-hand fashion.</i>
</div>
