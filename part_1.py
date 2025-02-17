import cv2
import matplotlib.pyplot as plt


# Part-a: Detect all coins in the image
# -> Use edge detection, to detect all coins in the image.
# -> Visualize the detected coins by outlining them in the image.

# Loading the image
image_path = "./images/coins.jpg"
image = cv2.imread(image_path)
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Applying Gaussian Blur to smooth the image
blurred = cv2.GaussianBlur(gray, (15, 15), 0)

# Using adaptive thresholding to separate coins from the background
thresh = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 3)

# Finding contours
contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# Filtering out small contours based on area
min_area = 5000
filtered_contours = [cnt for cnt in contours if cv2.contourArea(cnt) > min_area]

output = image.copy()
cv2.drawContours(output, filtered_contours, -1, (0, 255, 0), 3)

# Converting to RGB for displaying in Matplotlib
output_rgb = cv2.cvtColor(output, cv2.COLOR_BGR2RGB)

# Display the result
plt.figure(figsize=(8, 6))
plt.imshow(output_rgb)
plt.axis("off")
plt.show()



# Part-b: Segmentation of Each Coin
# -> Apply region-based segmentation techniques to isolate individual coins from the image.
# -> Provide segmented outputs for each detected coin.


# Loading image
image_path = "./images/coins.jpg"
image = cv2.imread(image_path)
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Applying Gaussian Blur to reduce noise
blurred = cv2.GaussianBlur(gray, (15, 15), 0)

# Thresholding to create a binary mask
_, thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

# Finding contours
contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# Filtering contours to remove small unwanted detections
min_area = 5000
filtered_contours = [cnt for cnt in contours if cv2.contourArea(cnt) > min_area]

# Sorting contours by area (largest first) to prioritize outer edges of coins
filtered_contours = sorted(filtered_contours, key=cv2.contourArea, reverse=True)

# Removing nested contours (inner details like ₹10's zero)
final_contours = []
for i, cnt in enumerate(filtered_contours):
    keep = True
    x, y, w, h = cv2.boundingRect(cnt)
    
    for j, other_cnt in enumerate(filtered_contours):
        if i != j:
            x2, y2, w2, h2 = cv2.boundingRect(other_cnt)
            # If the current contour is completely inside another, discarding it
            if x > x2 and y > y2 and x + w < x2 + w2 and y + h < y2 + h2:
                keep = False
                break

    if keep:
        final_contours.append(cnt)

# Extracting segmented coins
segmented_coins = []
for cnt in final_contours:
    x, y, w, h = cv2.boundingRect(cnt)
    coin_segment = image[y:y+h, x:x+w]
    segmented_coins.append(coin_segment)

# Displaying segmented coins
fig, axes = plt.subplots(1, len(segmented_coins), figsize=(15, 5))
for ax, coin in zip(axes, segmented_coins):
    ax.imshow(cv2.cvtColor(coin, cv2.COLOR_BGR2RGB))
    ax.axis("off")
plt.show()


# Part-c: Count the Total Number of Coins
# -> Write a function to count the total number of coins detected in the image.
# -> Display the final count as an output.


def count_coins(image_path):

    image = cv2.imread(image_path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Applying Gaussian Blur to reduce noise
    blurred = cv2.GaussianBlur(gray, (15, 15), 0)

    # Thresholding to create a binary mask
    _, thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # Finding contours
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Filtering contours to remove small unwanted detections
    min_area = 5000  # Adjust based on coin size
    filtered_contours = [cnt for cnt in contours if cv2.contourArea(cnt) > min_area]

    # Removing nested contours (fixes ₹10 coin inner ring issue)
    final_contours = []
    for i, cnt in enumerate(filtered_contours):
        keep = True
        x, y, w, h = cv2.boundingRect(cnt)
        
        for j, other_cnt in enumerate(filtered_contours):
            if i != j:
                x2, y2, w2, h2 = cv2.boundingRect(other_cnt)
                # If the contour is completely inside another, discarding it
                if x > x2 and y > y2 and x + w < x2 + w2 and y + h < y2 + h2:
                    keep = False
                    break

        if keep:
            final_contours.append(cnt)

    # Counting the number of coins
    total_coins = len(final_contours)

    # Drawing contours and displaying count on the image
    result_image = image.copy()
    cv2.drawContours(result_image, final_contours, -1, (0, 255, 0), 3)

    # Adjusting font
    font_scale = 5  
    font_thickness = 6  

    cv2.putText(result_image, f"Total Coins: {total_coins}", (50, 200), 
                cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 0, 255), font_thickness, cv2.LINE_AA)

    # Showing result
    cv2.imshow("Detected Coins", result_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    return total_coins

image_path = "./images/coins.jpg"   
total = count_coins(image_path)
print(f"Total number of coins detected: {total}")
