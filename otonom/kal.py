import matplotlib.pylab as plt
import cv2
import numpy as np

def region_of_interest(img, vertices):
    mask = np.zeros_like(img)
    match_mask_color = 255
    cv2.fillPoly(mask, vertices, match_mask_color)
    masked_image = cv2.bitwise_and(img, mask)
    return masked_image

def draw_the_lines(img, lines):
    img = np.copy(img)
    blank_image = np.zeros((img.shape[0], img.shape[1], 3), dtype=np.uint8)

    for line in lines:
        for x1, y1, x2, y2 in line:
            # Çizginin eğimini hesapla
            slope = (y2 - y1) / (x2 - x1 + 1e-6)  # Bölme hatasını önlemek için küçük bir değer ekledik
            angle = np.degrees(np.arctan(slope))

            # Yatay çizgileri (kaldırım çizgilerini) ayırt et
            if -10 <= angle <= 10:
                color = (0, 0, 255)  # kırmızı (Kaldırım çizgileri)
            else:
                color = (0, 255, 0)  # yeşil (Diğer şeritler)

            cv2.line(blank_image, (x1, y1), (x2, y2), color, thickness=10)

    img = cv2.addWeighted(img, 0.8, blank_image, 1, 0.0)
    return img

def process(image):
    height = image.shape[0]
    width = image.shape[1]
    region_of_interest_vertices = [
        (0, height),
        (width / 2, height / 2),
        (width, height)
    ]

    gray_image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    canny_image = cv2.Canny(gray_image, 100, 120)
    cropped_image = region_of_interest(canny_image, np.array([region_of_interest_vertices], np.int32))

    lines = cv2.HoughLinesP(cropped_image,
                            rho=2,
                            theta=np.pi / 180,
                            threshold=50,
                            lines=np.array([]),
                            minLineLength=40,
                            maxLineGap=100)

    if lines is not None:
        image_with_lines = draw_the_lines(image, lines)
        return image_with_lines

    return image

cap = cv2.VideoCapture('1.mp4')

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame = process(frame)
    cv2.imshow('frame', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
