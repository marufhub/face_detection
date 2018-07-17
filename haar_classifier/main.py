import cv2
import matplotlib.pyplot as plt

#load cascade classifier training file for haarcascade
haar_face_cascade = cv2.CascadeClassifier('../lib/haarcascades/haarcascade_frontalface_alt.xml')


def convertToRGB(img):
    """convert the test image to gray image as opencv face detector expects gray images"""
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

def detect_faces(f_cascade, img, scale_factor=1.1):
    """Face detect and make a rectangle over face and retur the image.
    """
    #make a copy of original images
    img_copy = img.copy()

    #convert the test image to gray image as opencv face detector expects gray images
    img_gray = cv2.cvtColor(img_copy, cv2.COLOR_BGR2GRAY)

    #let's detect multiscale (some images may be closer to camera than others) images
    faces = f_cascade.detectMultiScale(img_gray, scaleFactor=scale_factor, minNeighbors=5)

    #go over list of faces and draw them as rectangles on original colored img
    for (x, y, w, h) in faces:
        cv2.rectangle(img_copy, (x, y), (x+w, y+h), (0, 255, 0), 2)

    return img_copy

#load image
img = cv2.imread('../images/group_image.jpg')

#call our function to detect faces
faces_detected_img = detect_faces(haar_face_cascade, img)

#convert image to RGB and show image
plt.imshow(convertToRGB(faces_detected_img))
plt.show()
plt.close()
