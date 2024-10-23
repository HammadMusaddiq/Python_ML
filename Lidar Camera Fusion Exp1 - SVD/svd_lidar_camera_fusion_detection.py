import os
import cv2
import torch
from torch.utils.data import DataLoader
import torch.nn.functional as F
import torchvision.transforms as transforms
import numpy as np
import argparse

from kitti_dataset import KittiDataset
from sf_model import SensorFusionModel
from svd_sf_model import SensorFusionModelwithSVD

def load_kitti_annotations(annotation_path):
    annotations = []
    with open(annotation_path, 'r') as file:
        for line in file:
            parts = line.strip().split()
            annotations.append({
                'type': parts[0],
                'bbox': [float(p) for p in parts[4:8]],
                'location': [float(p) for p in parts[11:14]],
                'rotation_y': float(parts[14])
            })
    return annotations

def load_kitti_data(base_path, indices):
    lidar_data = []
    camera_images = []
    annotations = []

    for idx in indices:
        lidar_file = os.path.join(base_path, 'data_object_velodyne/training/velodyne', f'{idx:06d}.bin')
        image_file = os.path.join(base_path, 'data_object_image_2/training/image_2', f'{idx:06d}.png')
        annotation_file = os.path.join(base_path, 'data_object_label_2/training/label_2', f'{idx:06d}.txt')
        
        lidar = np.fromfile(lidar_file, dtype=np.float32).reshape(-1, 4)
        image = cv2.imread(image_file)
        annot = load_kitti_annotations(annotation_file)
        
        lidar_data.append(lidar)
        camera_images.append(image)
        annotations.append(annot)
        
    return lidar_data, camera_images, annotations

# Separate the dataset into Car, Truck, and None subsets
def separate_classes(dataset):
    car_indices = []
    truck_indices = []
    none_indices = []
    
    for idx in range(len(dataset)):
        _, _, labels = dataset[idx]
        val = labels[1]
        # print(val)
        if (val == 2).any().item():
            truck_indices.append(idx)
        elif (val == 1).any().item():
            car_indices.append(idx)
        else:
            none_indices.append(idx)
    
    return car_indices, truck_indices, none_indices

# Split the data into train and test sets
def split_data(car_indices, truck_indices, none_indices, train_ratio=0.8):
    np.random.shuffle(car_indices)
    np.random.shuffle(truck_indices)
    np.random.shuffle(none_indices)
    
    train_size_car = int(train_ratio * len(car_indices))
    train_size_truck = int(train_ratio * len(truck_indices))
    train_size_none = int(train_ratio * len(none_indices))
    
    train_indices = (
        car_indices[:train_size_car] + 
        truck_indices[:train_size_truck] + 
        none_indices[:train_size_none]
    )
    test_indices = (
        car_indices[train_size_car:] + 
        truck_indices[train_size_truck:] + 
        none_indices[train_size_none:]
    )
    
    np.random.shuffle(train_indices)
    np.random.shuffle(test_indices)
    
    return train_indices, test_indices

# Create train and test subsets
def create_subsets(dataset, train_indices, test_indices):
    train_dataset = torch.utils.data.Subset(dataset, train_indices)
    test_dataset = torch.utils.data.Subset(dataset, test_indices)
    return train_dataset, test_dataset

def loss_fn(predictions, bbox_targets, class_targets):
    # Extract predicted bounding boxes and class scores
    pred_bboxes = predictions[:, :, :4]  # Shape: [batch_size, num_objects, 4]
    pred_classes = predictions[:, :, 4:]  # Shape: [batch_size, num_objects, num_classes]
    
    # Flatten the targets and predictions
    pred_bboxes = pred_bboxes.contiguous().view(-1, 4)  # Flatten to [batch_size * num_objects, 4]
    pred_classes = pred_classes.contiguous().view(-1, 3)  # Flatten to [batch_size * num_objects, num_classes]
    bbox_targets = bbox_targets.contiguous().view(-1, 4)  # Flatten to [batch_size * num_objects, 4]
    class_targets = class_targets.contiguous().view(-1)  # Flatten to [batch_size * num_objects]
    
    # Bounding box regression loss
    bbox_loss = F.mse_loss(pred_bboxes, bbox_targets)
    
    # Class prediction loss
    class_loss = F.cross_entropy(pred_classes, class_targets)
    
    return bbox_loss + class_loss

def calculate_iou(pred_boxes, gt_boxes):
    x1 = torch.max(pred_boxes[:, 0], gt_boxes[:, 0])
    y1 = torch.max(pred_boxes[:, 1], gt_boxes[:, 1])
    x2 = torch.min(pred_boxes[:, 2], gt_boxes[:, 2])
    y2 = torch.min(pred_boxes[:, 3], gt_boxes[:, 3])
    
    intersection = torch.clamp(x2 - x1, min=0) * torch.clamp(y2 - y1, min=0)
    
    pred_area = (pred_boxes[:, 2] - pred_boxes[:, 0]) * (pred_boxes[:, 3] - pred_boxes[:, 1])
    gt_area = (gt_boxes[:, 2] - gt_boxes[:, 0]) * (gt_boxes[:, 3] - gt_boxes[:, 1])
    union = pred_area + gt_area - intersection
    
    return intersection / union

def calculate_precision_recall(pred_boxes, gt_boxes, pred_scores, iou_threshold):
    ious = calculate_iou(pred_boxes, gt_boxes)
    sorted_indices = torch.argsort(pred_scores, descending=True)
    pred_boxes = pred_boxes[sorted_indices]
    ious = ious[sorted_indices]
    
    tp = torch.zeros(pred_boxes.size(0))
    fp = torch.zeros(pred_boxes.size(0))
    
    detected = []
    for i in range(pred_boxes.size(0)):
        if ious[i] >= iou_threshold:
            if gt_boxes[i] not in detected:
                tp[i] = 1
                detected.append(gt_boxes[i])
            else:
                fp[i] = 1
        else:
            fp[i] = 1
    
    tp_cumsum = torch.cumsum(tp, dim=0)
    fp_cumsum = torch.cumsum(fp, dim=0)
    
    precision = tp_cumsum / (tp_cumsum + fp_cumsum)
    recall = tp_cumsum / len(gt_boxes)
    
    return precision, recall

def calculate_ap(precision, recall):
    precision = torch.cat([torch.tensor([0.0]), precision, torch.tensor([0.0])])
    recall = torch.cat([torch.tensor([0.0]), recall, torch.tensor([1.0])])
    
    for i in range(precision.size(0) - 1, 0, -1):
        precision[i - 1] = torch.max(precision[i - 1], precision[i])
    
    indices = torch.where(recall[1:] != recall[:-1])[0]
    ap = torch.sum((recall[indices + 1] - recall[indices]) * precision[indices + 1])
    return ap

def calculate_map(pred_boxes, gt_boxes, pred_scores, iou_thresholds=[0.25, 0.5]):
    aps = []
    for iou_threshold in iou_thresholds:
        precision, recall = calculate_precision_recall(pred_boxes, gt_boxes, pred_scores, iou_threshold)
        ap = calculate_ap(precision, recall)
        aps.append(ap)
    return torch.mean(torch.tensor(aps))

def evaluate_model(model, dataloader):
    model.eval()
    all_pred_boxes = []
    all_gt_boxes = []
    all_pred_scores = []
    total_loss = 0.0
    total_correct = 0
    total_samples = 0
    
    with torch.no_grad():
        for lidar, image, (bbox_targets, class_targets) in dataloader:
            lidar = lidar.view(lidar.size(0), -1, 4)
            image = image.view(image.size(0), 3, 224, 224)
            outputs = model(lidar, image)
            
            pred_bboxes = outputs[:, :, :4].contiguous().view(-1, 4)
            pred_scores = outputs[:, :, 4:].contiguous().view(-1, 3)
            pred_scores, pred_classes = torch.max(pred_scores, dim=1)
            
            gt_bboxes = bbox_targets.view(-1, 4)
            all_pred_boxes.append(pred_bboxes)
            all_gt_boxes.append(gt_bboxes)
            all_pred_scores.append(pred_scores)
            
            # Calculate loss
            loss = loss_fn(outputs, bbox_targets, class_targets)
            total_loss += loss.item()
            
            # Calculate accuracy
            total_correct += (pred_classes == class_targets.view(-1)).sum().item()
            total_samples += pred_classes.size(0)
    
    all_pred_boxes = torch.cat(all_pred_boxes)
    all_gt_boxes = torch.cat(all_gt_boxes)
    all_pred_scores = torch.cat(all_pred_scores)
    
    iou_thresholds = [0.25, 0.5]
    mAP = calculate_map(all_pred_boxes, all_gt_boxes, all_pred_scores, iou_thresholds)
    accuracy = total_correct / total_samples
    average_loss = total_loss / len(dataloader)
    
    print(f'mAP @ IoU thresholds {iou_thresholds}: {mAP.item():.4f}')
    print(f'Accuracy: {accuracy:.4f}')
    print(f'Loss: {average_loss:.4f}')
    
    return mAP, accuracy, average_loss

def train_model(model, train_dataloader, val_dataloader, optimizer, num_epochs=3):
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        total_correct = 0
        total_samples = 0
        for lidar, image, (bbox_targets, class_targets) in train_dataloader:
            optimizer.zero_grad()
            lidar = lidar.view(lidar.size(0), -1, 4)
            image = image.view(image.size(0), 3, 224, 224)
            outputs = model(lidar, image)
            loss = loss_fn(outputs, bbox_targets, class_targets)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            
            pred_classes = torch.argmax(outputs[:, :, 4:], dim=2)
            total_correct += (pred_classes == class_targets).sum().item()
            total_samples += pred_classes.size(0) * pred_classes.size(1)
        
        epoch_loss = running_loss / len(train_dataloader)
        epoch_accuracy = total_correct / total_samples
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {epoch_loss:.4f}, Accuracy: {epoch_accuracy:.4f}')
        
        # Evaluate the model on the validation set
        val_map, val_accuracy, val_loss = evaluate_model(model, val_dataloader)
        print(f'Validation mAP @ IoU thresholds [0.25, 0.5]: {val_map.item():.4f}')
        print(f'Validation Accuracy: {val_accuracy:.4f}')
        print(f'Validation Loss: {val_loss:.4f}')

def main(model_type):

    # Load data for example indices
    indices = range(50)
    base_path = 'kitti_data'
    lidar_data, camera_images, annotations = load_kitti_data(base_path, indices)

    # Define transforms
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize((224, 224)),
    ])

    # Create dataset and dataloader
    dataset = KittiDataset(lidar_data, camera_images, annotations, transform=transform)

    car_indices, truck_indices, none_indices = separate_classes(dataset)
    train_indices, test_indices = split_data(car_indices, truck_indices, none_indices, train_ratio=0.8)
    # print(car_indices, truck_indices)

    train_dataset, test_dataset = create_subsets(dataset, train_indices, test_indices)

    # Print sizes   
    print(f"Total dataset size: {len(dataset)}")
    print(f"Train dataset size: {len(train_dataset)}")
    print(f"Test dataset size: {len(test_dataset)}")
    
    # Initialize the dataloaders
    train_dataloader = DataLoader(train_dataset, batch_size=4, shuffle=True)
    test_dataloader = DataLoader(test_dataset, batch_size=4, shuffle=True)

    # Initialize the model based on the input argument
    if model_type == 'baseline':
        model = SensorFusionModel()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        print("Training Lidar-Camera Model")
        train_model(model, train_dataloader, test_dataloader, optimizer, num_epochs=10)

    elif model_type == 'svd':
        model_with_svd = SensorFusionModelwithSVD()
        svd_optimizer = torch.optim.Adam(model_with_svd.parameters(), lr=0.001)
        print("Training Lidar-Camera Model with SVD")
        train_model(model_with_svd, train_dataloader, test_dataloader, svd_optimizer, num_epochs=10)

    else:
        print("Invalid model type specified. Choose either 'baseline' or 'svd'.")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train 3D Object Detection Models')
    parser.add_argument('--model', type=str, required=True, choices=['baseline', 'svd'],
                        help='Specify the model type to train (baseline or svd)')
    args = parser.parse_args()
    
    main(args.model)