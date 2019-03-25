import cv2

# include protocols for image comparison
face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

# create initial image in grayscale
img = cv2.imread("photo.jpg")
grey_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# set image parameters
faces = face_cascade.detectMultiScale(grey_img,
scaleFactor = 1.05,
minNeighbors = 5)

# create bounding box around face
for x, y, w, h in faces:
    img = cv2.rectangle(img, (x, y), (x+w, y+h), (0,255,0), 3)

# update image capture
cv2.imshow("grey", img)
cv2.waitKey(0)
cv2.destroyAllWindows()
