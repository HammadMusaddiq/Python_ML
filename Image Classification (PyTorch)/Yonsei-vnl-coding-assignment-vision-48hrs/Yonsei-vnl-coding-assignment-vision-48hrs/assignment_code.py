import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
from PIL import Image
import pandas as pd
import numpy as np

# Define your custom dataset class
class CustomCIFAR100Dataset:
    def __init__(self, csv_file, transform=None):
        self.data = pd.read_csv(csv_file, header=None)
        print("Actual Data: ", self.data.shape)
        self.data.dropna(subset=1, inplace=True)
        print("Data after Data imputaiton", self.data.shape)
        self.transform = transform

         # Create a mapping between class labels and numerical indices
        self.label_map = {}  # Initialize an empty mapping
        self.label_counter = 0  # Counter for assigning numerical labels

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_name = self.data.iloc[idx, 0]  # Assuming the image path is in the first column
        image = Image.open("dataset/" + img_name)
        label_str = self.data.iloc[idx, 1]  # Assuming the label is in the second column

         # Convert the string label to a numerical index
        if label_str not in self.label_map:
            self.label_map[label_str] = self.label_counter
            self.label_counter += 1

        label = self.label_map[label_str]

        if self.transform:
            image = self.transform(image)

        return image, label


    def classes(self):
        # Create a list of class names based on the label_map
        class_names = [''] * len(self.label_map)
        for label_str, label_index in self.label_map.items():
            class_names[label_index] = label_str
        return class_names
    

# Define data augmentation and transformations
transform = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.Resize((32, 32)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  # Normalize to the range [-1, 1]
])

# Create an instance of your custom dataset
dataset = CustomCIFAR100Dataset('dataset\data\cifar100_nl.csv', transform=transform)

# # Create a random sampler if you want to shuffle the data
# # If you don't want to shuffle, you can remove this part
# shuffle = True
# sampler = torch.utils.data.RandomSampler(dataset) if shuffle else None

# Define batch size
batch_size = 64

# Create your own batch loading logic (without DataLoader)
batched_data = []

image,label = dataset[0]
print(image.shape)
print(label)


print(dataset.classes)
print(len(dataset.classes))

# if sampler is not None:
#     indices = list(sampler)
# else:
#     indices = list(range(len(dataset)))

# for i in range(0, len(indices), batch_size):
#     batch_indices = indices[i:i + batch_size]
#     batch = [dataset[idx] for idx in batch_indices]
#     batch_images, batch_labels = zip(*batch)
#     batch_images = torch.stack(batch_images)
#     # Convert batch_labels to a list to keep labels as strings
#     batch_labels = list(batch_labels)
#     batched_data.append((batch_images, batch_labels))

# Shuffle the dataset indices
indices = np.arange(len(dataset))
np.random.shuffle(indices)

# Create batches manually
batches = [indices[i:i+batch_size] for i in range(0, len(dataset), batch_size)]

# Create a model (e.g., a simple CNN)
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, 3)
        self.fc1 = nn.Linear(64 * 10 * 10, 100)

    def forward(self, x):
        x = self.conv1(x)
        x = x.view(-1, 64 * 10 * 10)
        x = self.fc1(x)
        return x

model = Net()

# Define loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training loop
num_epochs = 10  # Adjust as needed
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0

    for batch_indices in batches:
        batch_images = []
        batch_labels = []

        for idx in batch_indices:
            image, label_str = dataset[idx]
            batch_images.append(image)
            batch_labels.append(label_str)

        batch_images = torch.stack(batch_images).to(device)
        # labels_tensor = torch.tensor(batch_labels).to(device)
        labels_list = batch_labels

        optimizer.zero_grad()

        outputs = model(batch_images)
        loss = criterion(outputs, labels_list)
    
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    # Print training loss for this epoch
    print(f"Epoch [{epoch + 1}/{num_epochs}] Loss: {running_loss / len(batches)}")

# Save the trained model
torch.save(model.state_dict(), 'custom_cifar100_model.pth')

print("Model Saved")