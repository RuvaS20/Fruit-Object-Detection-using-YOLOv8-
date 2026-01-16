# Fruit Object Detection using YOLOv8n

**CS 463 Final Project - Vision Systems in the Wild**  
**Author:** Ruvarashe Sadya

Multi-class fruit detection system using transfer learning with YOLOv8n. Detects and localizes 6 fruit types: apple, banana, grape, orange, pineapple, and watermelon.

---

## Project Overview

- **Model:** YOLOv8n
- **Dataset:** 8,479 images from Kaggle (84-10-6 train-val-test split)
- **Classes:** 6 fruit types
- **Training Strategy:** Progressive fine-tuning (frozen backbone then partial unfreeze then full fine-tune)

---

## Running the Code

### **Option 1: Google Colab**

1. **Upload to Colab:**
   - Upload `Ruva_Final_Project_CV` to Google Colab

2. **Set up Kaggle API:**
   - Download your `kaggle.json` from Kaggle
   - When prompted by `setup()`, upload the file

3. **Mount Google Drive:**
   - Code will prompt you to authorize Google Drive access
   - Required for saving model checkpoints

4. **Run:**
   - Execute all cells sequentially
   - GPU runtime recommended (e.g T4 GPU)

---

### **Option 2: Local Execution**

#### **Prerequisites:**
```bash
pip install torch torchvision ultralytics kaggle pycocotools pillow matplotlib numpy pyyaml
```

#### **Required Code Modifications:**

**1. Remove Google Colab dependencies:**
```python
# Examples:
from google.colab import drive
from google.colab import files
from IPython.display import Image as IPImage, display
drive.mount('/content/drive')
```

**2. Update dataset path (Line 82):**
```python
# CHANGE FROM:
dataset_root = '/content/fruit-detection'

# TO:
dataset_root = './fruit-detection'  # your preferred local path
```

**3. Modify setup() function :**
```python
# REPLACE setup() with:
def setup():
    """Set up for local execution"""
    import subprocess
    
    # Install packages
    subprocess.run(['pip', 'install', '-q', 'kaggle', 'ultralytics'])
    
    # Ensure kaggle.json is in ~/.kaggle/
    # Manual step: Place your kaggle.json in ~/.kaggle/
    
    # Download dataset
    print("\nDownloading the fruit detection dataset")
    subprocess.run(['kaggle', 'datasets', 'download', '-d', 'lakshaytyagi01/fruit-detection'])
    subprocess.run(['unzip', '-q', 'fruit-detection.zip', '-d', './fruit-detection'])
    
    print("\nDataset downloaded to ./fruit-detection")
```

**4. Update Google Drive paths:**
```python
# CHANGE FROM:
drive_root = Path("/content/drive/MyDrive/fruit_detection_models")

# TO:
your local directory

```
---

## Key Functions

- `setup()` - Download dataset and install dependencies
- `show_sample_images()` - Visualize dataset samples
- `object_class_distribution()` - Analyze class imbalance
- `object_size_distribution()` - Analyze bounding box sizes
- `add_coco_backgrounds()` - Add 800 background images
- `train_model()` - Train baseeline YOLOv8 with specified config
- `plot_all_metrics()` - Display confusion matrix, PR curves, F1 curves
- `analyze_missed_detection_sizes()` - Analyze failure modes

---

## Results Summary (More in the report)

| Model | mAP@0.5 | Precision | Recall |
|-------|---------|-----------|--------|
| Baseline | 51.07% | 0.607 | 0.394 |
| Phase 1 (Frozen) | 48.26% | 0.547 | 0.405 |
| Phase 2 (Partial) | 51.39% | 0.571 | 0.419 |
| Phase 3 (Full) | 51.19% | 0.577 | 0.410 |

**Key Findings:**
- Object size > class imbalance for performance
- 39.3% of objects are tiny (<1% of image area)
- Orange detection worst despite being 43.5% of dataset
- Background confusion: 44-65% miss rate across classes

---

## Citation

**Dataset:**  
L. Tyagi, "Fruit Detection Dataset," Kaggle, 2023. DOI: 10.34740/kaggle/dsv/4922010

---

## License

This project is for educational purposes (CS 463 coursework).

---

## Acknowledgments

- YOLOv8 by Ultralytics
- Dataset by Lakshay Tyagi (Kaggle)
- COCO Dataset for background images
