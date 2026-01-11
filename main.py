import urllib.request
from urllib.error import HTTPError, URLError
from io import BytesIO

import numpy as np
from PIL import Image, ImageOps, ImageEnhance
import matplotlib.pyplot as plt



# STALY ZDALNY OBRAZ (NIE LOSOWY)
IMAGE_URL = "https://raw.githubusercontent.com/opencv/opencv/master/samples/data/lena.jpg"

# LICZBA PRZEDZIALOW HISTOGRAMU
BINS = 256


# WCZYTANIE OBRAZU Z INTERNETU
def load_remote_image(url):
    request = urllib.request.Request(
        url,
        headers={"User-Agent": "Mozilla/5.0"}
    )
    with urllib.request.urlopen(request) as response:
        data = response.read()

    image = Image.open(BytesIO(data))
    image = image.convert("RGB")
    return image


# LICZENIE HISTOGRAMOW
def compute_histograms(image):
    array = np.array(image)

    # KANAŁY RGB
    r = array[:, :, 0].ravel()
    g = array[:, :, 1].ravel()
    b = array[:, :, 2].ravel()

    hr, _ = np.histogram(r, bins=BINS, range=(0, 255))
    hg, _ = np.histogram(g, bins=BINS, range=(0, 255))
    hb, _ = np.histogram(b, bins=BINS, range=(0, 255))

    # CALY OBRAZ JAKO SKALA SZAROSCI
    gray = image.convert("L")
    y = np.array(gray).ravel()
    hy, _ = np.histogram(y, bins=BINS, range=(0, 255))

    return hy, hr, hg, hb, y


# WYSWIETLANIE HISTOGRAMOW
def show_histograms(hy, hr, hg, hb):
    x = np.arange(BINS)

    plt.figure()
    plt.plot(x, hy)
    plt.title("HISTOGRAM CALY OBRAZ")
    plt.xlabel("JASNOSC")
    plt.ylabel("LICZBA PIKSELI")
    plt.show()

    plt.figure()
    plt.plot(x, hr, label="R")
    plt.plot(x, hg, label="G")
    plt.plot(x, hb, label="B")
    plt.title("HISTOGRAMY KANALOW RGB")
    plt.xlabel("JASNOSC")
    plt.ylabel("LICZBA PIKSELI")
    plt.legend()
    plt.show()


# ANALIZA JAKOSCI OBRAZU
def analyze_quality(hy, y_values):
    total = hy.sum()

    # SPRAWDZENIE CIENI I SWIATLA
    shadow_clip = hy[0:3].sum() / total
    highlight_clip = hy[253:256].sum() / total

    mean = np.mean(y_values)
    std = np.std(y_values)

    score = 100
    problems = []

    if shadow_clip > 0.01:
        problems.append("ZA DUZO CIEMNYCH PIXELI")
        score -= 20

    if highlight_clip > 0.01:
        problems.append("ZA DUZO JASNYCH PIXELI")
        score -= 20

    if mean < 90:
        problems.append("OBRAZ NIEDOSWIETLONY")
        score -= 15

    if mean > 165:
        problems.append("OBRAZ PRZESWIETLONY")
        score -= 15

    if std < 35:
        problems.append("NISKI KONTRAST")
        score -= 10

    if score >= 80:
        verdict = "DOBRA JAKOSC"
    elif score >= 60:
        verdict = "SREDNIA JAKOSC"
    else:
        verdict = "SLABA JAKOSC"

    return score, verdict, problems


# POPRAWA JAKOSCI OBRAZU (BONUS)
def improve_image(image, score):
    improved = ImageOps.autocontrast(image, cutoff=1)

    if score < 80:
        improved = ImageEnhance.Contrast(improved).enhance(1.1)
        improved = ImageEnhance.Brightness(improved).enhance(1.1)

    return improved


# PROGRAM GLOWNY
def main():
    try:
        image = load_remote_image(IMAGE_URL)
    except (HTTPError, URLError) as error:
        print("BLAD WCZYTYWANIA OBRAZU:", error)
        return

    plt.imshow(image)
    plt.title("OBRAZ ORYGINALNY")
    plt.axis("off")
    plt.show()

    hy, hr, hg, hb, y = compute_histograms(image)
    show_histograms(hy, hr, hg, hb)

    score, verdict, problems = analyze_quality(hy, y)

    print("OCENA JAKOSCI:", verdict)
    print("WYNIK:", score, "/ 100")

    if problems:
        print("WYKRYTE PROBLEMY:")
        for p in problems:
            print("-", p)

    if score < 80:
        improved = improve_image(image, score)

        plt.imshow(improved)
        plt.title("OBRAZ PO POPRAWIE")
        plt.axis("off")
        plt.show()


if __name__ == "__main__":
    main()

