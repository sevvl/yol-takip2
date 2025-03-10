import matplotlib.pylab as plt
import cv2
import numpy as np

# İlgi bölgesi maskesi
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
            # Eğimi hesapla
            if x2 - x1 != 0:  # Bölme hatasını önlemek için
                slope = (y2 - y1) / (x2 - x1)
                if abs(slope) < 0.5:  # Yatay çizgileri filtrele (eğim ~0 olanlar)
                    continue
            # Çizgiyi çiz
            cv2.line(blank_image, (x1, y1), (x2, y2), (0, 255, 0), thickness=10)
    img = cv2.addWeighted(img, 0.8, blank_image, 1, 0.0)
    return img

# Görüntüyü okuma
image = cv2.imread("res1.jpeg")
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
height = image.shape[0]
width = image.shape[1]

# Beyaz renk eşikleme
hsv_image = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
white_mask = cv2.inRange(hsv_image, (0, 0, 160), (180, 50, 255))  # Beyaz renk aralığını genişlettik
white_image = cv2.bitwise_and(image, image, mask=white_mask)

# Beyaz maske kontrolü
plt.imshow(white_image)
plt.title("Beyaz Maskelenmiş Görüntü")
plt.show()

# Gri tonlamaya çevirme
gray_image = cv2.cvtColor(white_image, cv2.COLOR_RGB2GRAY)

# Kenar tespiti (Eşik değerleri kontrol edildi)
canny_image = cv2.Canny(gray_image, 50, 150)

# Kenar tespiti görselleştirme
plt.imshow(canny_image, cmap='gray')
plt.title("Kenar Tespiti (Canny)")
plt.show()

# İlgi bölgesi maskesi uygulama
region_of_interest_vertices = [
    (0, height),
    (width * 0.5, height * 0.6),
    (width, height)
]
cropped_image = region_of_interest(canny_image, np.array([region_of_interest_vertices], np.int32))

# İlgi bölgesi görselleştirme
plt.imshow(cropped_image, cmap='gray')
plt.title("İlgi Bölgesi")
plt.show()

# Hough dönüşümü ile çizgileri tespit etme
lines = cv2.HoughLinesP(
    cropped_image,  
    rho=2,
    theta=np.pi / 180,
    threshold=50,  # Daha az katı bir eşik
    minLineLength=20,  # Daha kısa çizgiler için
    maxLineGap=30      # Çizgiler arası boşluk toleransı artırıldı
)

# Çizgileri görüntüye ekleme
if lines is not None:
    image_with_lines = draw_the_lines(image, lines)
else:
    print("Hiçbir çizgi tespit edilemedi.")
    image_with_lines = image

# Sonucu görselleştirme
plt.imshow(image_with_lines)
plt.title("Sonuç")
plt.show()