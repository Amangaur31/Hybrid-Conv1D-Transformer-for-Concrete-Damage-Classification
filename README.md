# AEFormer: A Hybrid Conv1D-Transformer Model for Concrete Damage Classification

## Description
This project introduces **AEFormer**, a lightweight, hybrid deep learning model designed for the accurate classification of concrete damage using **acoustic emission (AE) signals**.  

It integrates:
- **Conv1D layers** for local feature extraction  
- **Transformer encoders** to capture global patterns and long-range dependencies  

The model is specifically tailored for deployment on **low-power microcontrollers and edge devices**, making it a practical solution for **real-time Structural Health Monitoring (SHM)** applications.  

With fewer than **28,000 trainable parameters**, AEFormer achieves high computational efficiency while maintaining exceptional classification accuracy.

---

## Features
- **Hybrid Architecture** → Combines Conv1D for local feature extraction with a Transformer encoder for global signal patterns  
- **Lightweight Design** → <28K trainable parameters, ideal for embedded systems  
- **High Accuracy** → Test accuracy of **99.82%** on a benchmark dataset  
- **Real-time SHM** → Compact and efficient, enabling real-time deployment on edge devices  

---

## Performance

AEFormer was evaluated on a benchmark AE dataset containing **15,000 signals** across three damage classes: *tensile, shear, and mixed*.  
The model demonstrated superior performance compared to other lightweight models.

| Model         | Parameters | Test Acc (%) | Val Acc (%) | Val Loss |
|---------------|------------|--------------|-------------|----------|
| CNN [13]      | 20,243     | 98.67        | 98.44       | 0.0449   |
| Tiny ANN [14] | 4,019      | 98.40        | 98.71       | 0.0702   |
| **AEFormer**  | 27,843     | **99.82**    | **99.87**   | **0.0049** |

For class-wise metrics, **AEFormer consistently achieved F1 scores above 0.998**, demonstrating reliability and balanced performance across all damage types.

---

## Installation

Clone the repository and install dependencies:

```bash
git clone https://github.com/your-username/AEFormer-Concrete-Damage-Classification.git
cd AEFormer-Concrete-Damage-Classification
pip install -r requirements.txt
```
```bash
main/model.py
```
This will load the data, build the model, train it for 100 epochs with early stopping, and print the final test accuracy and loss.

🗄️ Dataset
The model uses a benchmark acoustic emission (AE) dataset collected from reinforced concrete blocks. The dataset consists of 15,000 signals, each downsampled to 1,000 points and labeled into one of three classes: tensile, shear, or mixed tensile-shear. The data is partitioned into 70% for training, 15% for validation, and 15% for testing.
