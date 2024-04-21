import cv2
import matplotlib.pyplot as plt
import numpy as np

# Exercise 1
img = "image_task1.jpeg"

original_image = cv2.imread(img)
# cv2.imshow('Original Image', original_image)
# cv2.waitKey(0)
cv2.imwrite("task1_original_image.jpeg", original_image)
# Gray Image
gray_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2GRAY)


negative_image = 255 - original_image
#cv2.imshow('Negative Image', negative_image)
#cv2.waitKey(0)
cv2.imwrite("task1_negative_image.jpeg", negative_image)


# Brightness Enhanced (50)
# Convert the image to the HSV color space
hsv_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2HSV)
# Adjust the brightness channel (value channel)
hsv_image[:, :, 2] += 50
# Clip the pixel values to ensure they remain in the valid range [0, 255]
hsv_image[:, :, 2] = np.clip(hsv_image[:, :, 2], 0, 255)
# Convert the image back to the BGR color space
enhanced_image = cv2.cvtColor(hsv_image, cv2.COLOR_HSV2BGR)
# Display the result
# cv2.imshow('Brightened Image (50)', enhanced_image)
# cv2.waitKey(0)
cv2.imwrite("task1_brightened_image(50).jpeg", enhanced_image)


# Brightness Enhanced (-50)
# Convert the image to the HSV color space
hsv_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2HSV)
# Adjust the brightness channel (value channel)
hsv_image[:, :, 2] -= 50
# Clip the pixel values to ensure they remain in the valid range [0, 255]
hsv_image[:, :, 2] = np.clip(hsv_image[:, :, 2], 0, 255)
# Convert the image back to the BGR color space
enhanced_image = cv2.cvtColor(hsv_image, cv2.COLOR_HSV2BGR)
# Display the result
# cv2.imshow('Brightened Image (-50)', enhanced_image)
cv2.waitKey(0)
cv2.imwrite("task1_brightened_image(-50).jpeg", enhanced_image)


# Contrast Enhanced (50)
# Apply contrast adjustment
contrast_enhanced_image = cv2.add(gray_image, 50)
# Display the result
# cv2.imshow('Contrast Enhanced Image (+50)', contrast_enhanced_image)
# cv2.waitKey(0)
cv2.imwrite("task1_contrast_enhanced_image(+50).jpeg", contrast_enhanced_image)


# Contrast Enhanced (-50)
# Apply contrast adjustment
contrast_enhanced_image = cv2.subtract(gray_image, 50)
# Display the result
#cv2.imshow('Contrast Enhanced Image (-50)', contrast_enhanced_image)
#cv2.waitKey(0)
cv2.imwrite("task1_contrast_enhanced_image(-50).jpeg", contrast_enhanced_image)


# Exercise 2
img2 = "image_task2.jpeg"

original_image2 = cv2.imread(img2)
# cv2.imshow('Original Image 2', original_image2)
# cv2.waitKey(0)
cv2.imwrite("task2_original_image.jpeg", original_image2)
# Gray Image
gray_image2 = cv2.cvtColor(original_image2, cv2.COLOR_BGR2GRAY)


# Enhance the contrast of the input image using Gamma correction
gamma = 2.5
gamma_corrected_image = np.clip((original_image2 / 255.0) ** (1 / gamma) * 255.0, 0, 255).astype(np.uint8)
#cv2.imshow('Contrast Enhanced Gamma Correction', gamma_corrected_image)
#cv2.waitKey(0)
cv2.imwrite("task2_contrast_enhanced_gamma_correction.jpeg", gamma_corrected_image)


# Enhance the contrast of the input image using histogram equalization technique
equalized_image = cv2.equalizeHist(gray_image2)
#cv2.imshow('Contrast Enhanced Histogram Equalization ', equalized_image)
#cv2.waitKey(0)
cv2.imwrite("task2_contrast_enhanced_histogram_equalization.jpeg", equalized_image)


# Histogram of grayscale intensities for the enhanced image
plt.hist(equalized_image.flatten(), bins=256, range=[0,256], color='b')
plt.xlabel('Pixel Intensity', color='black')
plt.ylabel('Frequency')
plt.title('Histogram of Grayscale Intensities')
plt.grid(True)
plt.savefig("task2_histogram_grayscale_intensities.jpeg")
# plt.show()


# Exercise 3
img3 = "image_task3.jpeg"

original_image3 = cv2.imread(img3)
# cv2.imshow('Original Image 3', original_image3)
# cv2.waitKey(0)
cv2.imwrite("task3_original_image.jpeg", original_image3)


# RGB to Gray
gray_image3 = cv2.cvtColor(original_image3, cv2.COLOR_BGR2GRAY)
# cv2.imshow('Gray Image', gray_image)
# cv2.waitKey(0)
cv2.imwrite("task3_gray_image.jpeg", gray_image3)


# Detect connected components
image3 = original_image3.copy()
# Threshold the grayscale image to create a binary image
_, thresh = cv2.threshold(gray_image3, 127, 255, cv2.THRESH_BINARY)
# Find contours in the binary image
contours1, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
# Draw each component with a different color
for i, contour in enumerate(contours1):
    # Create a mask for the current component
    colored_mask = np.zeros_like(gray_image3)
    # Draw the contours with the corresponding coloR
    cv2.drawContours(colored_mask, [contour], -1, 255, thickness=cv2.FILLED)
    image3[colored_mask == 0] = (0, 255, 0)
    image3[colored_mask == 255] = (255, 255, 255)
# Apply Gaussian blur
blurred_image = cv2.GaussianBlur(gray_image3, (5, 5), 0)
# Adaptive thresholding
binary_image = cv2.adaptiveThreshold(blurred_image, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 11, 4)
# Find contours in the binary image
contours, _ = cv2.findContours(binary_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
# Create a color mask for visualization
colors = [(0, 0, 255), (255, 0, 0)]
# Draw each component on the image with its original color
for i, contour in enumerate(contours[:1] + contours[2:]):
    # Get the bounding rectangle of the contour
    x, y, w, h = cv2.boundingRect(contour)
    image3[y:y+h, x:x+w] = colors[i]
# Display the result
# cv2.imshow('Result', original_image3)
# cv2.waitKey(0)
cv2.imwrite("task3_connected_components.jpeg", image3)


# A bar plot to compare the sizes (number of pixels)
component_sizes = [cv2.contourArea(contour) for contour in contours]
# Plot the sizes of connected components
plt.figure(figsize=(10, 6))
plt.bar(range(len(component_sizes)), component_sizes, color='blue')
plt.xlabel('Component Index')
plt.ylabel('Size (Number of Pixels)')
plt.title('Sizes of Connected Components')
plt.xticks(range(len(component_sizes)), range(1, len(component_sizes) + 1))
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.savefig("task3_bar_plot_pixels.jpeg")
# plt.show()


# Compute and print the percentage size of each connected component
total_pixels = original_image3.shape[0] * original_image3.shape[1]
for i, contour in enumerate(contours):
    # Compute the area of the current component
    component_area = cv2.contourArea(contour)
    # Compute the percentage size of the current component relative to the total area of the image
    percentage_size = (component_area / total_pixels) * 100
    print(f"Component {i + 1}: {percentage_size:.2f}%")


cv2.destroyAllWindows()


