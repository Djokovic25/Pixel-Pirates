
import cv2
import numpy as np

def estimate_size(img, scale_factor=1.0):
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    _, thresh = cv2.threshold(gray, 0, 255,
                              cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    contours, _ = cv2.findContours(thresh,
                                   cv2.RETR_EXTERNAL,
                                   cv2.CHAIN_APPROX_SIMPLE)

    if len(contours) == 0:
        return 0, None

    contour = max(contours, key=cv2.contourArea)

    rect = cv2.minAreaRect(contour)
    box = cv2.boxPoints(rect).astype(int)

    max_dist = 0
    for i in range(len(box)):
        for j in range(i+1, len(box)):
            dist = np.linalg.norm(box[i] - box[j])
            max_dist = max(max_dist, dist)

    size_um = max_dist * scale_factor
    return size_um, contour
import cv2
import numpy as np

# 🔥 Find largest contour (particle)
def get_main_contour(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5,5), 0)

    _, thresh = cv2.threshold(blur, 0, 255,
                              cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    contours, _ = cv2.findContours(
        thresh,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    if len(contours) == 0:
        return None

    return max(contours, key=cv2.contourArea)


# 🔥 Feret diameter (max distance between contour points)
def feret_diameter(contour):
    if contour is None:
        return 0

    pts = contour.reshape(-1, 2)

    max_dist = 0
    for i in range(len(pts)):
        for j in range(i+1, len(pts)):
            dist = np.linalg.norm(pts[i] - pts[j])
            if dist > max_dist:
                max_dist = dist

    return max_dist


# 🔥 FINAL size estimation
def estimate_size(img, scale_factor=1.0):
    contour = get_main_contour(img)

    if contour is None:
        return 0, None

    feret_px = feret_diameter(contour)

    size_um = feret_px * scale_factor

    return size_um, contour