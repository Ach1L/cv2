import cv2
import numpy as np
import argparse


# Организация работы с аргументами командной строки
def parse():
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--imagePath', type = str)
    parser.add_argument('-c', '--confidence', type = float, default = 0.5)
    parser.add_argument('-n', '--nms', type = float, default = 0.4)
    return parser.parse_args()


# Загрузка предобученной модели YOLOv3
def initializeModel():
    modelCfg, modelWeights, classNames = "yolov3.cfg", "yolov3.weights", "coco.names"

    net = cv2.dnn.readNet(modelWeights, modelCfg)

    layerNames = net.getLayerNames()

    outputLayerNames = [layerNames[i - 1] for i in net.getUnconnectedOutLayers()]

    classes = []
    with open(classNames, "r") as f:
        classes = [line.strip() for line in f.readlines()]

    return net, outputLayerNames, classes


# Определение глобальных переменных
net, outputLayerNames, classes = initializeModel()


# Функция для анализа выходов нейронной сети
def processNetworkOutput(res: tuple, h: int, w: int,
                         confidence: float) -> list:
    classIds = []
    confidences = []
    boxes = []

    for layersRes in res:
        for detection in layersRes:
            scores = detection[5:]

            classId = np.argmax(scores)
            conf = max(scores)

            if conf > confidence:
                (centerX, centerY, width, height) = (detection[:4] * np.array([w, h, w, h])).astype(int)

                x = int(centerX - (width / 2))
                y = int(centerY - (height / 2))

                boxes.append([x, y, width, height])
                confidences.append(float(conf))
                classIds.append(classId)

    return classIds, confidences, boxes


# Функция для отрисовки результатов
def drawDetections(image: np.ndarray, boxes: list, confidences: list,
                   classIds: list, boxesIndAfterNMS: np.ndarray) -> np.ndarray:
    stat = {}
    colors = np.random.uniform(0, 255, size = (len(classes), 3))
    for i in range(len(boxes)):
        if i in boxesIndAfterNMS:
            x, y, w, h = boxes[i]

            label = str(classes[classIds[i]])

            color = colors[classIds[i]].tolist()

            cv2.rectangle(image, (x, y), (x + w, y + h), color, 2)

            cv2.putText(image, f"{label} {confidences[i]:.3f}", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

            if label in stat:
                stat[label] += 1
            else:
                stat[label] = 1
    print(stat)
    return image


def detect(image: np.ndarray, confidence: float, nms: float) -> np.ndarray:
    height, width = image.shape[0], image.shape[1]

    blob = cv2.dnn.blobFromImage(image = image, scalefactor = 1 / 255, size = (416, 416),
                                 mean = (0, 0, 0), swapRB = True, crop = False)

    net.setInput(blob)

    res = net.forward(outputLayerNames)

    classIds, confidences, boxes = processNetworkOutput(res, height, width, confidence)
    
    boxesIndAfterNMS = cv2.dnn.NMSBoxes(boxes, confidences, confidence, nms)

    image = drawDetections(image, boxes, confidences, classIds, boxesIndAfterNMS)

    return image


def main():
    args = parse()

    image = cv2.imread(args.imagePath)
    if image is None:
        print("Can't open image.")

    if args.imagePath:
        image = detect(image, args.confidence, args.nms)
        if image is not None:
            cv2.imshow("Detected Image", image)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
    else:
        print("Use -i flag to provide image path.")


if __name__ == "__main__":
    main()
