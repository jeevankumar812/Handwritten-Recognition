import cv2

def segment_characters(img_path):

    img = cv2.imread(img_path,0)

    _,th = cv2.threshold(
        img,0,255,
        cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU
    )

    contours,_ = cv2.findContours(
        th,cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    boxes = [cv2.boundingRect(c) for c in contours]

    # sort left to right
    boxes = sorted(boxes, key=lambda b:b[0])

    chars = []
    for x,y,w,h in boxes:
        if w*h > 300:
            ch = th[y:y+h, x:x+w]
            ch = cv2.resize(ch,(64,64))
            chars.append(ch)

    return chars
