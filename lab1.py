import cv2
import numpy as np
import argparse


# Организация работы с аргументами командной строки
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--image_path', type = str)
    parser.add_argument('-f', '--filter', type = str,
                        choices = ['grayscale', 'resize', 'sepia',
                                   'vignette', 'pixelate'])

    parser.add_argument('-r', '--resizeValue', type = float, default = 1)
    parser.add_argument('-v', '--vignetteRadius', type = float, default = 500)
    parser.add_argument('-p', '--pixelationBlockSize', type = int, default = 10)

    return parser.parse_args()


# Функция для вывода изображения на экран
def picToScreen(name: str,
                image: np.ndarray) -> None:
    cv2.imshow(name, image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


# Определение фильтра grayscale
def grayscale(image: np.ndarray) -> None:
    # Проверка корректности изображения
    if image.shape[2] != 3:
        raise RuntimeError("Image must have 3 channels or it was opened with an error.")

    B, G, R = cv2.split(image)

    greyImage = (R * 0.2989 + G * 0.5870 + B * 0.1140).astype(np.uint8)

    greyImage = cv2.merge([greyImage, greyImage, greyImage])

    picToScreen('Grey image', greyImage)


# Определение функции, изменяющей размер изображения
def resize(image: np.ndarray,
           value: float) -> None:
    if value < 0:
        raise RuntimeError('Value must be greater than zero')

    height, width = image.shape[0], image.shape[1]

    newWidth = int(width * value)
    newHeight = int(height * value)

    x = np.floor(np.arange(newWidth) / value).astype(int)
    y = np.floor(np.arange(newHeight) / value).astype(int)

    resizedImage = image[y[:, None], x]

    picToScreen('Resized image', resizedImage)


# Определение функции, дающей изображению фотоэффект сепии
def sepia(image: np.ndarray) -> None:
    if image.shape[2] != 3:
        raise RuntimeError("Image must have 3 channels or it was opened with an error.")

    B, G, R = cv2.split(image)

    newR = ((0.393 * R) + (0.769 * G) + (0.189 * B)).clip(0, 255).astype(np.uint8)
    newG = ((0.349 * R) + (0.686 * G) + (0.168 * B)).clip(0, 255).astype(np.uint8)
    newB = ((0.272 * R) + (0.534 * G) + (0.131 * B)).clip(0, 255).astype(np.uint8)

    sepiaImage = cv2.merge([newB, newG, newR])

    picToScreen('Sepia image', sepiaImage)


# Определение функции, дающей изображению фотоэффект виньетки
def vignette(image: np.ndarray,
             radius: float) -> None:
    if radius <= 0:
        raise RuntimeError("Radius must be greater then zero.")

    height, width = image.shape[0], image.shape[1]

    y, x = height // 2, width // 2

    yInd, xInd = np.indices((height, width))

    distances = np.sqrt((xInd - x) ** 2 + (yInd - y) ** 2)

    mask = np.clip(1 - distances / radius, 0, 1)

    B, G, R = cv2.split(image)
    B = (B * mask).astype(np.uint8)
    G = (G * mask).astype(np.uint8)
    R = (R * mask).astype(np.uint8)

    vignetteImage = cv2.merge([B, G, R])

    picToScreen('Vignette image', vignetteImage)


x1, y1, x2, y2 = -1, -1, -1, -1
drawing = False


# Определение пикселизации заданной прямоугольной области изображения
def pixel_filter(image: np.ndarray,
                 size: int) -> None:
    global x1, y1, x2, y2

    if x1 == y1 == x2 == y2 == -1:
        x1, y1 = 0, 0
        x2, y2 = image.shape[1], image.shape[0]

    block = image[y1:y2, x1:x2]
    blockHeight, blockWidth = block.shape[0], block.shape[1]

    height = blockHeight // size
    width = blockWidth // size

    for i in range(height):
        startY = i * size
        endY = min(startY + size, blockHeight)
        for j in range(width):
            startX = j * size
            endX = min(startX + size, blockWidth)

            block[startY:endY, startX:endX] = (block[startY:endY, startX:endX].mean(axis = (0, 1))).astype(int)

    image[y1:y2, x1:x2] = block

    picToScreen('Pixelated image', image)


# Функция, отвечающая за обработку события мыши для рисования прямоугольника и пикселизации выбранной области
def draw(event: int,
         x: int, y: int,
         flags: int, param: None) -> None:
    global drawing, x1, y1, x2, y2

    if event == cv2.EVENT_LBUTTONDOWN:
        drawing = True
        x1, y1 = x, y

    elif event == cv2.EVENT_MOUSEMOVE:
        if drawing:
            x2, y2 = x, y

    elif event == cv2.EVENT_LBUTTONUP:
        drawing = False
        x2, y2 = x, y


# Сама функция пикселизации
def pixel_image(image: np.ndarray,
                size: int) -> None:
    # Проверка корректности размера блока
    if size <= 0:
        raise RuntimeError("Block size must be greater then zero.")

    global drawing, x1, y1, x2, y2

    cv2.namedWindow('Image without pixelation')
    cv2.setMouseCallback('Image without pixelation', draw)

    while True:
        img_copy = image.copy()

        if x1 != -1 and y1 != -1 and x2 != -1 and y2 != -1:
            cv2.rectangle(img_copy, (x1, y1), (x2, y2), (0, 255, 0), 2)

        cv2.imshow('Image without pixelation', img_copy)

        key = cv2.waitKey(1)
        if key == ord('p'):
            cv2.destroyWindow('Image without pixelation')
            pixel_filter(image, size)
            break


def main():
    args = parse_args()
    image_path = args.image_path
    image = cv2.imread(image_path)

    picToScreen('Original image', image)

    if args.filter == 'grayscale':
        grayscale(image)
    elif args.filter == 'resize':
        resize(image, args.resizeValue)
    elif args.filter == 'sepia':
        sepia(image)
    elif args.filter == 'vignette':
        vignette(image, args.vignetteRadius)
    elif args.filter == 'pixelate':
        pixel_image(image, args.pixelationBlockSize)


if __name__ == '__main__':
    main()
