import cv2
from cvzone.HandTrackingModule import HandDetector
import numpy as np
import os
import traceback

# Constants
DATA_DIR = "D://test_data_2.0"  
IMAGE_SIZE = 400
WHITE_COLOR = (255, 255, 255)
GREEN_COLOR = (0, 255, 0)
RED_COLOR = (0, 0, 255)

# Initialize hand detector
hd = HandDetector(maxHands=1)
hd2 = HandDetector(maxHands=1)

# Function to create a white image
def create_white_image():
    return np.ones((IMAGE_SIZE, IMAGE_SIZE), np.uint8) * 255

# Function to draw hand skeleton
def draw_hand_skeleton(image, hand):
    pts = hand['lmList']
    offset = (IMAGE_SIZE // 2) - 15
    for t in range(0, 4, 1):
        cv2.line(image, (pts[t][0] + offset, pts[t][1] + offset), (pts[t + 1][0] + offset, pts[t + 1][1] + offset), GREEN_COLOR, 3)
    for t in range(5, 8, 1):
        cv2.line(image, (pts[t][0] + offset, pts[t][1] + offset), (pts[t + 1][0] + offset, pts[t + 1][1] + offset), GREEN_COLOR, 3)
    for t in range(9, 12, 1):
        cv2.line(image, (pts[t][0] + offset, pts[t][1] + offset), (pts[t + 1][0] + offset, pts[t + 1][1] + offset), GREEN_COLOR, 3)
    for t in range(13, 16, 1):
        cv2.line(image, (pts[t][0] + offset, pts[t][1] + offset), (pts[t + 1][0] + offset, pts[t + 1][1] + offset), GREEN_COLOR, 3)
    for t in range(17, 20, 1):
        cv2.line(image, (pts[t][0] + offset, pts[t][1] + offset), (pts[t + 1][0] + offset, pts[t + 1][1] + offset), GREEN_COLOR, 3)
    cv2.line(image, (pts[5][0] + offset, pts[5][1] + offset), (pts[9][0] + offset, pts[9][1] + offset), GREEN_COLOR, 3)
    cv2.line(image, (pts[9][0] + offset, pts[9][1] + offset), (pts[13][0] + offset, pts[13][1] + offset), GREEN_COLOR, 3)
    cv2.line(image, (pts[13][0] + offset, pts[13][1] + offset), (pts[17][0] + offset, pts[17][1] + offset), GREEN_COLOR, 3)
    cv2.line(image, (pts[0][0] + offset, pts[0][1] + offset), (pts[5][0] + offset, pts[5][1] + offset), GREEN_COLOR, 3)
    cv2.line(image, (pts[0][0] + offset, pts[0][1] + offset), (pts[17][0] + offset, pts[17][1] + offset), GREEN_COLOR, 3)
    for i in range(21):
        cv2.circle(image, (pts[i][0] + offset, pts[i][1] + offset), 2, RED_COLOR, 1)

# Function to process hand image
def process_hand_image(frame, hands, offset):
    if hands:
        hand = hands[0]
        x, y, w, h = hand['bbox']
        image = frame[y - offset:y + h + offset, x - offset:x + w + offset]

        # Process image for different types
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (1, 1), 2)
        gray2 = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        blur2 = cv2.GaussianBlur(gray2, (5, 5), 2)
        th3 = cv2.adaptiveThreshold(blur2, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2)
        _, test_image = cv2.threshold(th3, 27, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

        # Create image for display
        img_final = create_white_image()
        h = test_image.shape[0]
        w = test_image.shape[1]
        img_final[((IMAGE_SIZE - h) // 2):((IMAGE_SIZE - h) // 2) + h, ((IMAGE_SIZE - w) // 2):((IMAGE_SIZE - w) // 2) + w] = test_image

        img_final1 = create_white_image()
        h = blur.shape[0]
        w = blur.shape[1]
        img_final1[((IMAGE_SIZE - h) // 2):((IMAGE_SIZE - h) // 2) + h, ((IMAGE_SIZE - w) // 2):((IMAGE_SIZE - w) // 2) + w] = blur

        return image, img_final, img_final1
    return None, None, None

# Function to save images
def save_images(img_final, img_final1, p_dir, c_dir, count):
    cv2.imwrite(os.path.join(DATA_DIR, "Binary_imgs", p_dir, c_dir + str(count) + ".jpg"), img_final)
    cv2.imwrite(os.path.join(DATA_DIR, "Gray_imgs", p_dir, c_dir + str(count) + ".jpg"), img_final1)

# Main loop
if __name__ == "__main__":
    offset = 30
    step = 1
    flag = False
    suv = 0
    p_dir = "A"
    c_dir = "a"

    capture = cv2.VideoCapture(0)
    while True:
        try:
            _, frame = capture.read()
            frame = cv2.flip(frame, 1)

            # Detect hands in the main frame
            hands = hd.findHands(frame, draw=False, flipType=True)

            # Process hand image for binary and gray images
            image, img_final, img_final1 = process_hand_image(frame, hands, offset)

            # Draw hand skeleton on white image
            white_image = create_white_image()
            if hands:
                hand = hands[0]
                x, y, w, h = hand['bbox']
                cv2.rectangle(white_image, (x - offset, y - offset), (x + w, y + h), (3, 255, 25), 3)
                image1 = frame[y - offset:y + h + offset, x - offset:x + w + offset]
                handz = hd2.findHands(image1, draw=False, flipType=True)
                if handz:
                    draw_hand_skeleton(white_image, handz[0])
                    cv2.imshow("skeleton", white_image)

            # Display images
            cv2.imshow("frame", frame)
            cv2.imshow("binary", img_final)

            # Key press handling
            interrupt = cv2.waitKey(1)
            if interrupt & 0xFF == 27:
                break
            if interrupt & 0xFF == ord('n'):
                p_dir = chr(ord(p_dir) + 1)
                c_dir = chr(ord(c_dir) + 1)
                if ord(p_dir) == ord('Z') + 1:
                    p_dir = "A"
                    c_dir = "a"
                flag = False
                count = len(os.listdir(os.path.join(DATA_DIR, "Gray_imgs", p_dir)))
            if interrupt & 0xFF == ord('a'):
                flag = not flag
                if flag:
                    suv = 0
            if flag:
                if suv == 50:
                    flag = False
                if step % 2 == 0:
                    save_images(img_final, img_final1, p_dir, c_dir, count)
                    count += 1
                    suv += 1
                step += 1

        except Exception:
            print("==", traceback.format_exc())

    capture.release()
    cv2.destroyAllWindows()







# Resize image to 224x224
#img_final = cv2.resize(img_final, (224, 224))

# Create a white background image of size 400x400
#img_finalf = np.ones((400, 400, 3), np.uint8) * 255
#print("img final shape= ", img_final.shape)

# Iterate over each pixel and set it to either white or black
#for i in range(400):
    #for j in range(400):
        #if img_final[i][j] == 255:
            #img_finalf[i][j] = [255, 255, 255]
        #else:
            #img_finalf[i][j] = [0, 0, 0]

#print("img final f shape= ", img_finalf.shape)

# Apply median blur to the test image
#image = cv2.medianBlur(test_image, 5)

# Define kernels for dilation and erosion
#kernel = np.ones((3, 3), np.uint8)
#kernel1 = np.ones((1, 1), np.uint8)

# Perform dilation followed by erosion
#dilate = cv2.dilate(image, kernel, iterations=1)
#dilate = cv2.erode(dilate, kernel1, iterations=1)

# Display the images
#cv2.imshow("gray", gray)
#cv2.imshow("blur", blur)
#cv2.imshow("adapt threshold", th3)
#cv2.imshow("roi", test_image)

# White increase

#if flag:
    #if step % 2 == 0:
        #cv2.imwrite("D:\\sign_data\\B\\b" + str(count) + ".jpg", img_final)
        #print(count)
        #count += 1
    #step += 1
