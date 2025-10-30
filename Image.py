import numpy as np
# from matplotlib.image import imread, imsave
from matplotlib.pyplot import imshow
import cv2
import matplotlib.pyplot as plt
from enum import Enum


class ColorModel(Enum):
    rgb = 0
    hsv = 1
    hsi = 2
    hsl = 3
    gray = 4  # obraz 2d


class BaseImage:
    data: np.ndarray  # tensor przechowujacy piksele obrazu
    color_model: ColorModel  # atrybut przechowujacy biezacy model barw obrazu

    def __init__(self, path: str = None) -> None:
        """
        inicjalizator wczytujacy obraz do atrybutu data na podstawie sciezki
        """
        if path is None:
            self.data = np.array([])
            self.color_model = None
            return
        self.data = cv2.imread(path)
        self.data = cv2.cvtColor(self.data, cv2.COLOR_BGR2RGB)
        self.data = cv2.convertScaleAbs(self.data)
        # self.data = imread(path)
        # if not (self.data > 1).any():
        #     self.data = (self.data * 255).astype(np.uint8)
        #     self.data = self.data[:, :, :3]
        dimensions = np.ndim(self.data)
        if dimensions == 3:
            self.color_model = ColorModel.rgb
        elif dimensions == 2:
            self.color_model = ColorModel.gray
        else:
            self.color_model = None

    def __str__(self) -> str:
        return f"Obiekt BaseImage\nModel: {self.color_model}\nWymiary: {self.data.shape}\nDane ({self.data.dtype}):\n{self.data}"

    def save_img(self, path: str) -> None:
        """
        metoda zapisujaca obraz znajdujacy sie w atrybucie data do pliku
        """
        if self.color_model is not ColorModel.rgb:
            raise ValueError("Można zapisać tylko obraz w formacie RGB.")
        cv2.imwrite(path, self.data)
        # imsave(path, self.data)

    def show_img(self) -> None:
        """
        metoda wyswietlajaca obraz znajdujacy sie w atrybucie data
        from matplotlib.colors import hsv_to_rgb
        """
        if self.color_model is ColorModel.gray:
            imshow(self.data, cmap='gray')
        else:
            imshow(self.data)

    def get_layer(self, layer_id: int) -> 'BaseImage':
        """
        metoda zwracajaca warstwe o wskazanym indeksie
        """
        if self.color_model is ColorModel.gray:
            raise ValueError("Nie można rozbić dwuwymiarowego obrazu na warstwy.")
        if layer_id not in [0, 1, 2]:
            raise ValueError("Nieprawidłowy indeks. Możliwe opcje to 0, 1 lub 2.")
        layers = np.squeeze(np.dsplit(self.data, self.data.shape[-1]))
        layer = BaseImage()
        layer.data = layers[layer_id]
        layer.color_model = ColorModel.gray
        return layer

    def to_hsv(self) -> 'BaseImage':
        """
        metoda dokonujaca konwersji obrazu w atrybucie data do modelu hsv
        metoda zwraca nowy obiekt klasy image zawierajacy obraz w docelowym modelu barw
        """
        if self.color_model is not ColorModel.rgb:
            raise ValueError("Podany obraz nie jest zapisany w modelu RGB.")

        def rgb_hsv(R, G, B):
            Rp, Gp, Bp = R / 255, G / 255, B / 255
            V = max(Rp, Gp, Bp)
            diff = V - min(Rp, Gp, Bp)
            if V > 0:
                S = diff / V
            else:
                S = 0
            licznik = 1 / 2 * (R-G + R-B)
            mianownik = np.sqrt((R-G)**2 + (R-B)*(G-B))
            H = np.degrees(np.arccos(licznik / mianownik))
            if B > G:
                H = 360 - H
            return np.array([int(round(H)), float(round(S, 4)), float(round(V, 4))])

        hsv_image = BaseImage()
        operate = np.array(self.data.astype(float).reshape(-1, 3))
        for i, pixel in enumerate(operate):
            r, g, b = pixel
            operate[i] = rgb_hsv(r, g, b)
        hsv_image.data = operate.reshape(self.data.shape)
        hsv_image.color_model = ColorModel.hsv
        return hsv_image

    def to_hsi(self) -> 'BaseImage':
        """
        metoda dokonujaca konwersji obrazu w atrybucie data do modelu hsi
        metoda zwraca nowy obiekt klasy image zawierajacy obraz w docelowym modelu barw
        """
        if self.color_model is not ColorModel.rgb:
            raise ValueError("Podany obraz nie jest zapisany w modelu RGB.")

        def rgb_hsi(R, G, B):
            Rp = R / 255
            Gp = G / 255
            Bp = B / 255
            I = (Rp + Gp + Bp) / 3
            mianownik1 = R + G + B
            if mianownik1 == 0:
                mianownik1 = 0.0001
            S = 1 - 3 / mianownik1 * min(R, G, B)
            licznik = 1 / 2 * (R-G + R-B)
            mianownik2 = np.sqrt((R-G)**2 + (R-B)*(G-B))
            H = np.degrees(np.arccos(licznik / mianownik2))
            if B > G:
                H = 360 - H
            return np.array([round(H, 4), round(S, 4), round(I, 4)]).astype(float)

        hsi_image = BaseImage()
        operate = np.array(self.data.astype(float).reshape(-1, 3))
        for i, pixel in enumerate(operate):
            r, g, b = pixel
            operate[i] = rgb_hsi(r, g, b)
        hsi_image.data = operate.reshape(self.data.shape)
        hsi_image.color_model = ColorModel.hsi
        return hsi_image

    def to_hsl(self) -> 'BaseImage':
        """
        metoda dokonujaca konwersji obrazu w atrybucie data do modelu hsl
        metoda zwraca nowy obiekt klasy image zawierajacy obraz w docelowym modelu barw
        """
        if self.color_model is not ColorModel.rgb:
            raise ValueError("Podany obraz nie jest zapisany w modelu RGB.")
        
        def rgb_hsl(R, G, B):
            M = max(R, G, B)
            m = min(R, G, B)
            d = (M - m) / 255
            L = (0.5 * (M + m)) / 255
            if L > 0:
                S = d / (1 - abs(2 * L - 1))
            else:
                S = 0
            licznik = 1 / 2 * (R-G + R-B)
            mianownik2 = np.sqrt((R-G)**2 + (R-B)*(G-B))
            H = np.degrees(np.arccos(licznik / mianownik2))
            if B > G:
                H = 360 - H
            return np.array([int(round(H)), float(round(S, 4)), float(round(L, 4))])
        
        hsl_image = BaseImage()
        operate = np.array(self.data.astype(float).reshape(-1, 3))
        for i, pixel in enumerate(operate):
            r, g, b = pixel
            operate[i] = rgb_hsl(r, g, b)
        hsl_image.data = operate.reshape(self.data.shape)
        hsl_image.color_model = ColorModel.hsl
        return hsl_image

    def to_rgb(self) -> 'BaseImage':
        """
        metoda dokonujaca konwersji obrazu w atrybucie data do modelu rgb
        metoda zwraca nowy obiekt klasy image zawierajacy obraz w docelowym modelu barw
        """
        if self.color_model is ColorModel.rgb:
            raise ValueError("Podany obraz jest już zapisany w modelu RGB.")
        
        def hsv_rgb(H: int, S: float, V: float):
            M = 255 * V
            m = M * (1-S)
            z = (M - m) * (1 - abs(((H / 60) % 2) - 1))
            R, G, B = 0, 0, 0
            if 0 <= H < 60:
                R = M
                G = z + m
                B = m
            elif 60 <= H < 120:
                R = z + m
                G = M
                B = m
            elif 120 <= H < 180:
                R = m
                G = M
                B = z + m
            elif 180 <= H < 240:
                R = m
                G = M
                B = z + m
            elif 240 <= H < 300:
                R = z + m
                G = m
                B = M
            elif 300 <= H <= 360:
                R = M
                G = m
                B = z + m
            return np.array([round(R), round(G), round(B)]).astype(np.uint8)

        def hsi_rgb(H: float, S: float, I: float):
            R, G, B = 0, 0, 0
            def cos_degree(x):
                return np.cos(np.radians(x))
            if 0 <= H < 120:
                R = I * (1 + S*cos_degree(H) / cos_degree(60 - H))
                B = I * (1 - S)
                G = 3 * I - (R + B)
            elif 120 <= H < 240:
                H = H - 120
                R = I * (1 - S)
                G = I * (1 + S*cos_degree(H) / cos_degree(60 - H))
                B = 3 * I - (R + G)
            elif 240 <= H <= 360:
                H = H - 240
                G = I * (1 - S)
                B = I * (1 + S*cos_degree(H) / cos_degree(60 - H))
                R = 3 * I - (G + B)
            return np.array([round(R*255), round(G*255), round(B*255)]).astype(np.uint8)
            
        def hsl_rgb(H: float, S: float, L: float):
            R, G, B = 0, 0, 0
            d = S * (1 - abs(2*L - 1))
            m = 255 * (L - 0.5 * d)
            x = d * (1 - abs(((H / 60) % 2) - 1))
            if 0 <= H < 60:
                R = 255 * d + m
                G = 255 * x + m
                B = m
            if 60 <= H < 120:
                R = 255 * x + m
                G = 255 * d + m
                B = m
            if 120 <= H < 180:
                R = m
                G = 255 * d + m
                B = 255 * x + m
            if 180 <= H < 240:
                R = m
                G = 255 * x + m
                B = 255 * d + m
            if 240 <= H < 300:
                R = 255 * x + m
                G = m
                B = 255 * d + m
            if 300 <= H <= 360:
                R = 255 * d + m
                G = m
                B = 255 * x + m
            return np.array([round(R), round(G), round(B)]).astype(np.uint8)

        rgb_image = BaseImage()
        operate = np.array(self.data.astype(float).reshape(-1, 3))
        for i, pixel in enumerate(operate):
            h, s, v = pixel
            if self.color_model is ColorModel.hsv:
                operate[i] = hsv_rgb(h, s, v)
            elif self.color_model is ColorModel.hsi:
                operate[i] = hsi_rgb(h, s, v)
            elif self.color_model is ColorModel.hsl:
                operate[i] = hsl_rgb(h, s, v)
        rgb_image.data = operate.reshape(self.data.shape)
        rgb_image.color_model = ColorModel.rgb
        rgb_image.data = rgb_image.data.astype(np.uint8)
        return rgb_image

    def baseimage_to_image(self) -> 'Image':
        result = Image()
        result.data = self.data
        result.color_model = self.color_model
        return result


class GrayScaleTransform(BaseImage):

    def __init__(self, path: str) -> None:
        super().__init__(path)

    def to_gray(self) -> BaseImage:
        """
        metoda zwracajaca obraz w skali szarosci jako obiekt klasy BaseImage
        """
        if self.color_model is not ColorModel.rgb:
            raise ValueError("Obraz musi być w modelu RGB.")
            
        gray_image = BaseImage()
        # gray_image.data = np.mean(self.data.reshape(-1, 3), axis=1).astype(np.uint8).reshape(self.data.shape[0], self.data.shape[1])
        # gray_image.data = np.average(self.data.reshape(-1, 3), axis=1, weights=[0.299, 0.587, 0.114]).astype(np.uint8).reshape(self.data.shape[0], self.data.shape[1])
        gray_image.data = cv2.cvtColor(self.data, cv2.COLOR_RGB2GRAY)
        gray_image.color_model = ColorModel.gray
        return gray_image


    def to_sepia(self, alpha_beta: tuple = (None, None), w: int = None) -> BaseImage:
        """
        metoda zwracajaca obraz w sepii jako obiekt klasy BaseImage
        sepia tworzona metoda 1 w przypadku przekazania argumentu alpha_beta
        lub metoda 2 w przypadku przekazania argumentu w
        """
        if alpha_beta == (None, None) and w is None:
            raise ValueError("Należy podać jeden argument (alpha, beta) lub w.")
            
        sepia_image: BaseImage = self.to_gray()
        sepia_image.data = sepia_image.data.astype(np.float64)
        if alpha_beta != (None, None):
            alpha, beta = alpha_beta
            if alpha < 1 or beta > 1 or alpha+beta != 2:
                raise ValueError("Alfa musi byc wieksza, a beta mniejsza niz 1 oraz ich suma musi wynosić 2.")

            L_zero = sepia_image.data * alpha
            L_one = sepia_image.data
            L_two = sepia_image.data * beta

        if w is not None:
            if not 20 <= w <= 40:
                raise ValueError("Wartość w musi się znaleźć w przedziale [20, 40].")

            L_zero = sepia_image.data + 2 * w
            L_one = sepia_image.data + w
            L_two = sepia_image.data

        for array in L_zero, L_one, L_two:
            array[(array > 255)] = 255
        sepia_image.data = np.dstack((L_zero, L_one, L_two)).astype(np.uint8)
        sepia_image.color_model = ColorModel.rgb
        return sepia_image


class Histogram:
    """
    klasa reprezentujaca histogram danego obrazu
    """
    values: np.ndarray  # atrybut przechowujacy wartosci histogramu danego obrazu
    cumulated_values: np.ndarray # atrybut przechowujacy wartosci skumulowane histogramu

    def __init__(self, values: np.ndarray) -> None:
        if np.ndim(values) == 2:
            self.values = values.flatten()
        elif np.ndim(values) == 3:
            self.values = np.squeeze(np.dsplit(values, values.shape[-1]))
        else:
            raise ValueError("Tablica wejsciowa musi byc 2-wymiarowa (odcienie szarości) lub 3-wymiarowa (RGB).")
        self.cumulated_values = None

    def plot(self, cumulated: bool = False, layered: bool = False) -> None:
        """
        metoda wyswietlajaca histogram na podstawie atrybutu values
        """
        if np.ndim(self.values) == 1:
            plt.hist(self.values, bins=256, range=(0, 256), histtype='bar', color='gray', cumulative=cumulated)
            plt.ylabel("Liczba")
            plt.xlabel("Nasycenie")
            plt.ylim(bottom=-100)
            plt.title("Odcienie szarosci")
            
        elif np.ndim(self.values) == 3:
            color = ['red', 'green', 'blue']
            f, ax_arr = plt.subplots(1, 3, figsize=(14, 4))
            for i in range(3):
                layer = self.values[i].flatten()
                ax_arr[i].hist(layer, bins=256, range=(0, 256), histtype='bar', color=color[i], cumulative=cumulated)
                ax_arr[i].set_title(color[i].capitalize())
                ax_arr[i].set_ylim(bottom=-100)
                # ax_arr[i].set_xlim(right=np.max(layer))
            ax_arr[0].set_xlabel("Nasycenie")
            ax_arr[0].set_ylabel("Liczba")
        plt.suptitle("Histogramy - liczba pikseli warstwy o konkretnym nasyceniu")
        plt.show()

    def to_cumulated(self):
        """
        metoda zwracajaca histogram skumulowany na podstawie stanu wewnetrznego obiektu
        Ta metoda dodaje do obecnego obiektu zmienne skumulowane. Jesli chcemy tylko wyswietlic histogram skumulowany,
        nie musimy używać tej metody. Wystarczy: his.plot(cumulated=True).
        """
        cumul_sum = 0
        if np.ndim(self.values) == 3:
            cumul_array = np.zeros(shape=(3, 256), dtype=int)
            cumul_sub_array = np.zeros(shape=256, dtype=int)
            for dim, array in enumerate(self.values):
                for i in range(256):
                    cumul_sum += array[array == i].size
                    cumul_sub_array[i] = cumul_sum
                cumul_array[dim] = cumul_sub_array
        else:
            cumul_array = np.zeros(shape=256, dtype=int)
            for i in range(256):
                cumul_sum += self.values[self.values == i].size
                cumul_array[i] = cumul_sum
        
        self.cumulated_values = cumul_array


class ImageDiffMethod(Enum):
    mse = 0
    rmse = 1


class ImageComparison(GrayScaleTransform):
    """
    Klasa reprezentujaca obraz, jego histogram oraz metody porównania
    """
        
    def histogram(self) -> Histogram:
        """
        metoda zwracajaca obiekt zawierajacy histogram biezacego obrazu (1- lub wielowarstwowy)
        """
        return Histogram(self.data)

    def compare_to(self, other: 'Image', method: ImageDiffMethod) -> float:
        """
        metoda zwracajaca mse lub rmse dla dwoch obrazow
        """
        if self.color_model is ColorModel.rgb:
            self = self.to_gray()
            other = other.to_gray()
        his_1 = Histogram(self.data).values.astype(float)
        his_2 = Histogram(other.data).values.astype(float)
        if method is ImageDiffMethod.mse:
            return np.square(np.subtract(his_1, his_2)).mean()
        elif method is ImageDiffMethod.rmse:
            return np.sqrt(np.square(np.subtract(his_1, his_2)).mean())
        else:
            raise ValueError("Nie podano prawidlowej metody")


class ImageAligning(ImageComparison):
    """
    klasa odpowiadająca za wyrównywanie histogramu
    """
    def __init__(self, path: str = None) -> None:
        super().__init__(path)

    def align_image(self, tail_elimination: bool = True) -> 'BaseImage':
        """
        metoda zwracajaca poprawiony obraz metoda wyrownywania histogramow
        """
        aligned_image = BaseImage()
        if self.color_model is ColorModel.rgb:
            layers = np.squeeze(np.dsplit(self.data, self.data.shape[-1]))
            for dim, layer in enumerate(layers):
                if tail_elimination:
                    cumul_layer = np.zeros(shape=256, dtype=int)
                    cumul_sum = 0
                    for pixel in range(256):
                        cumul_sum += layer[layer == pixel].size
                        cumul_layer[pixel] = cumul_sum
                    min_value = np.where(cumul_layer == cumul_layer[np.abs(cumul_layer - np.percentile(cumul_layer, 5)).argmin()])[0][0]
                    max_value = np.where(cumul_layer == cumul_layer[np.abs(cumul_layer - np.percentile(cumul_layer, 95)).argmin()])[0][-1]
                else:
                    min_value = np.min(layer.data)
                    max_value = np.max(layer.data)
                
                aligned_layer = (layer.astype(float) - min_value) * (255 / (max_value - min_value))
                aligned_layer[(aligned_layer > 255)] = 255
                aligned_layer[(aligned_layer < 0)] = 0
                layers[dim] = aligned_layer

            aligned_image.color_model = ColorModel.rgb
            aligned_image.data = np.dstack([layers[i] for i in range(3)])

        elif self.color_model is ColorModel.gray:
            min_value = np.min(self.data)
            max_value = np.max(self.data)
            aligned_image.data = (self.data.astype(float) - min_value) * (255 / (max_value - min_value))
            aligned_image.data = aligned_image.data.astype(np.uint8)
            aligned_image.color_model = ColorModel.gray

        return aligned_image

    def clahe(self, clipLimit=2.0, tileGridSize=(4, 4)) -> BaseImage:
        clahe = cv2.createCLAHE(clipLimit, tileGridSize)
        result_image = BaseImage()
        if self.color_model is ColorModel.gray:
            result_image.data = clahe.apply(self.data)
            result_image.color_model = ColorModel.gray
            return result_image
        elif self.color_model is ColorModel.rgb:
            image_lab = cv2.cvtColor(self.data, cv2.COLOR_RGB2LAB)
            image_lab[..., 0] = clahe.apply(image_lab[..., 0])
            result_image.data = cv2.cvtColor(image_lab, cv2.COLOR_LAB2RGB)
            result_image.color_model = ColorModel.rgb
        else: 
            raise ValueError("Funkcja clahe działa tylko na obrazach RGB i w skali szarości.")
        return result_image


class ImageFiltration(BaseImage):
    def conv_2d(self, kernel: np.ndarray) -> BaseImage:
        """
        kernel: filtr w postaci tablicy numpy
        prefix: przedrostek filtra, o ile istnieje; Optional - forma poprawna obiektowo, lub domyslna wartosc = 1 - optymalne arytmetycznie
        metoda zwroci obraz po procesie filtrowania
        """
        try:
            x_kernel, y_kernel = kernel.shape
            if x_kernel != y_kernel or x_kernel % 2 != 1 or x_kernel < 3:
                raise ValueError()
        except ValueError:
            raise ValueError("Możliwy kernel tylko o nieparzystych kwadratowych wymiarach od 3x3")
        # kernel = np.flipud(np.fliplr(kernel))
        
        result_image = BaseImage()

        if self.color_model is ColorModel.rgb:
            result_image.color_model = ColorModel.rgb
            layers = np.squeeze(np.dsplit(self.data, self.data.shape[-1]))
            dimensions = np.ndim(layers)
            layers = layers.astype(float)
            x_image, y_image = layers[0].shape
            x_out, y_out = x_image, y_image
            if x_image % x_kernel != 0:
                x_out = x_image - x_kernel + 1
            if y_image % y_kernel != 0:
                y_out = y_image - y_kernel + 1

            result_layers = np.zeros((dimensions, x_out, y_out))
            for i, layer in enumerate(layers):
                for y in range(y_image - y_kernel + 1):
                    for x in range(x_image - x_kernel + 1):
                        result_layers[i, x, y] = np.sum(kernel * layer[x : x + x_kernel, y : y + y_kernel])
            result_image.data = np.dstack([result_layers[i] for i in range(dimensions)])

        elif self.color_model is ColorModel.gray:
            result_image.color_model = ColorModel.gray
            layer = self.data.astype(float)
            
            x_image, y_image = self.data.shape
            x_out, y_out = x_image, y_image
            if x_image % x_kernel != 0:
                x_out = x_image - x_kernel + 1
            if y_image % y_kernel != 0:
                y_out = y_image - y_kernel + 1

            result_image.data = np.zeros((x_out, y_out))
            for y in range(y_image - y_kernel + 1):
                for x in range(x_image - x_kernel + 1):
                    result_image.data[x, y] = np.sum(kernel * layer[x : x + x_kernel, y : y + y_kernel])
        else:
            raise ValueError("Funkcja conv_2d działa tylko na obrazach RGB i w skali szarości.")
        result_image.data[result_image.data < 0] = 0
        result_image.data[result_image.data > 255] = 255
        result_image.data = result_image.data.astype(np.uint8)
        return result_image


# Filtr tożsamościowy
id_kernel = np.array([[0, 0, 0], [0, 1, 0], [0, 0, 0]])

# Filtr górnoprzepustowy - wyostrzanie obrazu. Filtr wzmacnia szczegóły.
up_kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])

# Filtr dolnoprzepustowy - rozmycie obrazu. Filtr uśrednia szczegoły.
down_kernel = 1/9 * np.array([[1, 1, 1], [1, 1, 1], [1, 1, 1]])

# Rozmycie Gaussowskie - wersja 3x3 px
gauss_3_kernel = 1/16 * np.array([[1, 2, 1], [2, 4, 2], [1, 2, 1]])

# Rozmycie Gaussowskie - wersja 5x5 px
gauss_5_kernel = 1/256 * np.array([[1, 4, 6, 4, 1], [4, 16, 24, 16, 4], [6, 24, 36, 24, 6], [4, 16, 24, 16, 4], [1, 4, 6, 4, 1]])

# Operatory Sobela - detekcja krawędzi
sobel0 = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
sobel45 = np.array([[0, 1, 2], [-1, 0, 1], [-2, -1, 0]])
sobel90 = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]])
sobel135 = np.array([[2, 1, 0], [1, 0, -1], [0, -1, -2]])


class Thresholding(BaseImage):
    def threshold(self, value: int = 128) -> BaseImage:
        """
        metoda dokonujaca operacji segmentacji za pomoca binaryzacji
        value - wartość progu
        """
        if self.color_model is not ColorModel.gray:
            raise ValueError("Funkcja threshold działa tylko na obrazach w skali szarości. Użyj metody to_gray().")
        result_data = self.data
        result_data[result_data < value] = 0
        result_data[result_data >= value] = 255
        result_image = BaseImage()
        result_image.data = result_data
        result_image.color_model = ColorModel.gray
        return result_image

    def otsu(self, thresh=0, maxval=255, type=(cv2.THRESH_BINARY + cv2.THRESH_OTSU)) -> BaseImage:
        if self.color_model is not ColorModel.gray:
            raise ValueError("Funkcja threshold działa tylko na obrazach w skali szarości. Użyj metody to_gray().")
        result_image = BaseImage()
        _, result_image.data = cv2.threshold(self.data, thresh, maxval, type)
        result_image.color_model = ColorModel.gray
        return result_image
    
    def adaptive_threshold(self, maxValue=255, adaptiveMethod=cv2.ADAPTIVE_THRESH_MEAN_C, 
                            thresholdType=cv2.THRESH_BINARY, blockSize=13, C=8) -> BaseImage:
        if self.color_model is not ColorModel.gray:
            raise ValueError("Funkcja threshold działa tylko na obrazach w skali szarości. Użyj metody to_gray().")
        result_image = BaseImage()
        result_image.data = cv2.adaptiveThreshold(self.data, maxValue, adaptiveMethod, thresholdType, blockSize, C)
        result_image.color_model = ColorModel.gray
        return result_image


class EdgeDetection(BaseImage):
    def canny(self, threshold1=16, threshold2=40, sobel=3) -> BaseImage:
        if self.color_model is not ColorModel.gray:
            raise ValueError("Funkcja działa tylko na obrazach w skali szarości. Użyj metody to_gray().")
        result_image = BaseImage()
        result_image.data = cv2.Canny(self.data, threshold1, threshold2, apertureSize=sobel)
        result_image.color_model = ColorModel.gray
        return result_image
    
    def lines_thresh(self, thresh=0, maxval=255, type=(cv2.THRESH_BINARY + cv2.THRESH_OTSU)) -> BaseImage:
        if self.color_model is not ColorModel.gray:
            raise ValueError("Funkcja działa tylko na obrazach w skali szarości. Użyj metody to_gray().")
        result_image = BaseImage()
        _, result_image.data = cv2.threshold(self.data, thresh, maxval, type)
        result_image.color_model = ColorModel.gray
        return result_image

    def hough_lines_p(self, rho=2, theta=np.pi/180, threshold=30) -> np.ndarray:
        if self.color_model is not ColorModel.gray:
            raise ValueError("Funkcja działa tylko na obrazach w skali szarości. Użyj metody to_gray().")
        return cv2.HoughLinesP(self.data, rho, theta, threshold)

    def line_detection(self, threshold1=20, threshold2=50, sobel=3, 
                        thresh=0, maxval=255, type=(cv2.THRESH_BINARY + cv2.THRESH_OTSU), 
                        rho=2, theta=np.pi/180, threshold=30, color=(0, 255, 0), thickness=5) -> BaseImage:
        if self.color_model is ColorModel.rgb:
            gray_image = self.baseimage_to_image().to_gray().baseimage_to_image()
            # gray_image = BaseImage()
            # gray_image.data = cv2.cvtColor(self.data, cv2.COLOR_RGB2GRAY)
            # gray_image.color_model = ColorModel.gray
            # gray_image = gray_image.baseimage_to_image()
        elif self.color_model is ColorModel.gray:
            gray_image = self
        lines_thresh = gray_image.lines_thresh(thresh, maxval, type).baseimage_to_image()
        lines_edges = lines_thresh.canny(threshold1, threshold2, sobel).baseimage_to_image()
        lines = lines_edges.hough_lines_p(rho, theta, threshold)
        result_image = BaseImage()
        result_image.data = cv2.cvtColor(gray_image.data, cv2.COLOR_GRAY2RGB)
        for line in lines:
            x0, y0, x1, y1 = line[0]
            cv2.line(result_image.data, (x0, y0), (x1, y1), color, thickness)
        result_image.color_model = ColorModel.rgb
        return result_image

    def hough_circles(self, method=cv2.HOUGH_GRADIENT, dp=2, minDist=60, minRadius=20, maxRadius=100, 
                        color=(0, 255, 0), thickness=4) -> BaseImage:
        result_image = BaseImage()
        if self.color_model is ColorModel.rgb:
            gray_data = cv2.cvtColor(self.data, cv2.COLOR_RGB2GRAY)
            circles = cv2.HoughCircles(gray_data, method, dp, minDist, minRadius=minRadius, maxRadius=maxRadius)
            result_image.color_model = ColorModel.rgb
        elif self.color_model is ColorModel.gray:
            circles = cv2.HoughCircles(self.data, method, dp, minDist, minRadius=minRadius, maxRadius=maxRadius)
            result_image.color_model = ColorModel.gray
        else:
            raise ValueError("Funkcja działa tylko na obrazach RGB i w skali szarości.")
        result_image.data = self.data.copy()
        for (x, y, r) in circles.astype(int)[0]:
            cv2.circle(result_image.data, (x, y), r, color, thickness)
        return result_image


class Image(ImageAligning, ImageFiltration, Thresholding, EdgeDetection):
    """
    klasa stanowiaca glowny interfejs biblioteki
    w pozniejszym czasie bedzie dziedziczyla po kolejnych klasach
    realizujacych kolejne metody przetwarzania obrazow
    """
    def __init__(self, path: str = None) -> None:
        super().__init__(path)

    def __str__(self) -> str:
        return f"Obiekt Image\nModel: {self.color_model}\nWymiary: {self.data.shape}\nDane ({self.data.dtype}):\n{self.data}"
    
    def to_gray(self) -> 'Image':
        gray = Image()
        gray.data = super().to_gray().data
        gray.color_model = ColorModel.gray
        return gray
        
    def align_image(self, tail_elimination: bool = True) -> 'Image':
        aligned = Image()
        aligned.data = super().align_image(tail_elimination).data
        if self.color_model is ColorModel.rgb:
            aligned.color_model = ColorModel.rgb
        else:
            aligned.color_model = ColorModel.gray
        return aligned
    
    def threshold(self, value: int) -> 'Image':
        result = Image()
        result.data = super().threshold(value).data
        result.color_model = ColorModel.gray
        return result
