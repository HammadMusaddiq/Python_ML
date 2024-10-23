# 3D Object Detection Using SVD and LiDAR-Camera Sensor Fusion

## Overview
This repository implements a 3D object detection model using LiDAR and camera sensor fusion, with and without Singular Value Decomposition (SVD). The project evaluates the performance of these models on the KITTI dataset.

The code explores the effectiveness of SVD as a dimensionality reduction technique to improve both detection accuracy and computational efficiency in autonomous systems like self-driving cars or robotics.

## Dataset
The KITTI dataset is used for training and evaluation. It includes synchronized 3D point cloud data from LiDAR and 2D camera images, along with annotated object labels.

## Models
Two models are implemented:
1. **Baseline Model:** Sensor fusion model without using any linear algebra techniques.
2. **SVD Model:** Enhanced sensor fusion model incorporating Singular Value Decomposition (SVD) for dimensionality reduction on LiDAR data.

## Requirements
- Python 3.x
- PyTorch
- OpenCV
- NumPy
- TorchVision

You can install the required dependencies by running:

```bash
pip install -r requirements.txt
```

## Directory Structure
```
.
├── kitti_dataset.py          # Dataset handler for loading and preprocessing KITTI data
├── sf_model.py               # Baseline sensor fusion model
├── svd_sf_model.py           # Sensor fusion model with SVD
├── main.py                   # Main script for training, testing, and evaluation
├── README.md                 # Project documentation
└── kitti_data                # KITTI dataset (not included, download separately)
```

## Usage
1. **Prepare the Dataset**: Download the KITTI dataset from the official site and place it in the `kitti_data` directory. You should have both the LiDAR point clouds and camera images.

2. **Training the Model**:
   To train the baseline model without SVD:

   ```bash
   python main.py --model baseline
   ```

   To train the model with SVD:

   ```bash
   python main.py --model svd
   ```

3. **Evaluating the Model**:
   The models will automatically evaluate on the test set after training. The results, including precision, recall, and mean Average Precision (mAP), will be logged.

## Results
- Detailed results will be saved in the `results/` directory after each training session.
- The evaluation metrics can be visualized using provided scripts.

## Contributing
Contributions are welcome! Please open an issue or submit a pull request for any improvements or fixes.

## License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments
- [KITTI Dataset](http://www.cvlibs.net/datasets/kitti/)
- [PyTorch](https://pytorch.org/)
- [OpenCV](https://opencv.org/)
