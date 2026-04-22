# Scene Classification — CNN vs Transfer Learning

**Live Demo:** [Try it on Hugging Face Spaces](https://huggingface.co/spaces/kishore-9/scene-classification-cnn)

Comparing custom-built CNNs against pretrained transfer learning models on an outdoor scene classification task.

**Dataset:** Intel Image Classification (Kaggle `puneet6060/intel-image-classification`)  
6 classes — buildings, forest, glacier, mountain, sea, street  
~14,000 training images / ~3,000 test images at 150×150px

---

## Phase 1 — Custom CNNs

Built three CNN architectures from scratch to understand how design choices affect accuracy and parameter efficiency.

| Model | Architecture | Test Accuracy | Params |
|-------|-------------|:---:|---:|
| CNN v1 | 3 conv blocks + Flatten + FC head | 86.77% | 10.7M |
| **CNN v2** | **BatchNorm + GlobalAveragePooling** | **88.97%** | **504K** |
| CNN v3 | Block-wise Dropout + large FC | 85.43% | 43.1M |

CNN v2 was the clear winner — best accuracy and only 504K parameters (95% fewer than v3). BatchNorm stabilized training and GAP replaced the heavy FC layers entirely. v3 showed that more parameters and heavier regularization don't automatically mean better results on a ~14K image dataset.

---

## Phase 2 — Transfer Learning

Fine-tuned three ImageNet-pretrained models using a two-stage strategy: freeze the base and train the head first, then unfreeze the top layers for fine-tuning at a much lower learning rate.

| Model | Test Accuracy | Params | Size |
|-------|:---:|---:|---:|
| ResNet50 | 93.87% | 24.1M | 92.0 MB |
| MobileNetV2 | 91.07% | 2.6M | 9.9 MB |
| EfficientNetB0 | 90.43% | 4.4M | 16.7 MB |

All three transfer models outperformed the best custom CNN (88.97%). MobileNetV2 hit 91.07% with only 2.6M parameters — best efficiency of the group. ResNet50 achieved the highest accuracy at 93.87% but at 24M parameters and 92 MB.

---

## Deployment

`app.py` is a Gradio web app that runs scene classification with EfficientNetB0.

**Run locally:**
```bash
pip install -r requirements.txt
python app.py
```
> The model file `final_model_efficientnetb0.h5` needs to be in the same folder as `app.py`.  
> Download it from the project's Google Drive folder.

---

## Notebooks

| Notebook | Description |
|----------|-------------|
| `phase1_cnn.ipynb` | Custom CNN — 3 variants, training, evaluation, comparison |
| `phase2_transfer_learning.ipynb` | Transfer learning — ResNet50, MobileNetV2, EfficientNetB0 |

Both notebooks run on **Google Colab (T4 GPU)**. Models are saved to Google Drive after training — re-running any cell automatically loads from Drive instead of retraining.

---

## Results Summary

Transfer learning beats custom CNNs by a meaningful margin (~5 percentage points) with no extra training data. The pretrained ImageNet weights already encode low-level visual features — edges, textures, shapes — that transfer well to scene classification.

Key finding from Phase 1: architecture matters more than parameter count. CNN v2 with 504K parameters outperformed CNN v3 with 43M parameters.
