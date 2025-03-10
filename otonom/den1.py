import matplotlib.pylab as plt
import cv2
import numpy as np

def region_of_interest(img, vertices):    ## ilgili resim ve 3 köşe noktası verilmişken maskelenmiş resmi geri döndüren fonksiyon
    mask = np.zeros_like(img)    ##her bir kareyi maskeliyoruz
    #channel_count = img.shape[2]
    match_mask_color = 255
    cv2.fillPoly(mask, vertices, match_mask_color)  ##Verilen köşe noktalarıyla (bu örnekte bir üçgen) maskenin içine beyaz bir alan
    masked_image = cv2.bitwise_and(img, mask)   ## Maskeyi, verilen resimle bitwise AND işlemi uygulayarak yalnızca maskelenmiş bölgeleri tutar, geri kalan alanları siyah yapar.
    return masked_image

def drow_the_lines(img, lines):     ## şerit çizgilerinin üzerinde renkli çizgi çizdiren fonksiyon
    img = np.copy(img)
    blank_image = np.zeros((img.shape[0], img.shape[1], 3), dtype=np.uint8)   ## orijinal resim ile aynı büyuklükte boş bir resim oluşturuyoruz.

    for line in lines:      ## bu for döngüsü ile daha önce oluşturduğumuz boş resimin içine tespit ettiğimiz tüm şerit çizgilerini ekliyoruz.
        for x1, y1, x2, y2 in line:
            cv2.line(blank_image, (x1,y1), (x2,y2), (0, 255, 0), thickness=10)    ## (0,255,0) yeşil renk, istenirse değiştirilebilir.

    img = cv2.addWeighted(img, 0.8, blank_image, 1, 0.0)    ## orijinal resim ile şerit çizgilerini içeren blank image üst üste ekleniyor. ağırlıklı (0.8) birleştirme uygulanıyor.
    return img


#image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
def process(image):                               #### şerit algılamayı içeren tüm işlemleri process isimli bu fonksiyonda topluyoruz.
    print(image.shape)
    height = image.shape[0]
    width = image.shape[1]
    region_of_interest_vertices = [      ## Üçgen şeklinde sınırlı alanımızın üç noktadaki köşelerini tanımlıyoruz.
        (0, height),
        (width/2, height/2),
        (width, height)
    ]
    gray_image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)   ## resmi gri ölçeğe dönüştürme
    canny_image = cv2.Canny(gray_image, 100, 120)          ## Canny fonksiyonu ile iki eşik değeri (100,120) arasında kenar çizgileri algılatıyoruz (edge detection)
    cropped_image = region_of_interest(canny_image,        ##yalnızca bell üçgen alanını tutar gerisini maskeler
                    np.array([region_of_interest_vertices], np.int32),)
    lines = cv2.HoughLinesP(cropped_image,        ## HoughLinesP transform ile probabilistik (olasılıksal) dönüştürme yapılıyor.
                            rho=2,                ## burada HoughLinesP transform ile resimdeki çizgiler (şerit çizgileri) tanımlanmış oluyor.
                            theta=np.pi/180,
                            threshold=50,
                            lines=np.array([]),
                            minLineLength=40,
                            maxLineGap=100)
    image_with_lines = drow_the_lines(image, lines)
    return image_with_lines

cap = cv2.VideoCapture('2.mp4')     #### video dosyasından okutuyoruz.

while cap.isOpened():                    #### video capture açık olduğu sürece bu while döngüsü devam edecek. Q tuşuna basılınca döngü bitecek.
    ret, frame = cap.read()
    frame = process(frame)
    cv2.imshow('frame', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
