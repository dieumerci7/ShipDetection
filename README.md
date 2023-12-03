# Ship Detection

This project aims to build a semantic segmentation model for ship detection using TensorFlow and the UNet architecture (specifically, MobileNetV2). The [dataset](https://www.kaggle.com/c/airbus-ship-detection/data?select=train_ship_segmentations_v2.csv) used for training and testing is from the Kaggle Airbus Ship Detection Challenge.

## Repository Structure

- [eda.ipynb](eda.ipynb): Jupyter notebook with exploratory data analysis (EDA). Assumes the presence of the `train_v2` directory from the dataset in the project under the name `training_images`.

- [train.py](train.py): Python script for model training. Execute as follows:
  ```bash
  python train.py --n_epochs [n_epochs]
  ```
  
- [inference.py](inference.py): Python script for model inference. Execute as follows:
```bash
python inference.py --input_path [input_path] --output_name [output_name]
```
The result is saved under `segmented_images/[output_name]` (so far containing an example result).

[requirements.txt](requirements.txt): File containing the required Python modules. Install dependencies using:
```bash
pip install -r requirements.txt
```
# Model Details

The chosen model architecture is **MobileNetV2** with a UNet-like structure for semantic segmentation (written with TensorFlow). The best-trained model achieved a **Dice score of 99.64** on the test set after 5 epochs. It is saved under `models/final_model.h5`.

# Dataset

The Kaggle Airbus Ship Detection Challenge [dataset](https://www.kaggle.com/c/airbus-ship-detection/data?select=train_ship_segmentations_v2.csv) is utilized. Relevant images are extracted to prevent an imbalance of images without ships in the training set.

# Additional Information

- The project uses **TensorFlow** for deep learning tasks.
- The UNet architecture, specifically **MobileNetV2**, is employed for semantic segmentation.
- `requirements.txt` contains all necessary Python modules.

# Contributors

Bohdan-Yarema Dekhtiar
