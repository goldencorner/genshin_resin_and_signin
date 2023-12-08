import os
import subprocess

import time

from paddleocr import PaddleOCR

def get_screenshot():
    os.system("adb shell screencap -p /sdcard/screen.png")
    os.system("adb pull /sdcard/screen.png .")

get_screenshot()
'''离线百度文本识别'''
ocr = PaddleOCR(use_angle_cls=True, lang="ch")  # need to run only once to download and load model into memory
img_path = 'screen.png'
result = ocr.ocr(img_path, cls=True)
result = result[0]

# debug   显示结果
# image = Image.open(img_path).convert('RGB')
# boxes = [line[0] for line in result]
# txts = [line[1][0] for line in result]
# scores = [line[1][1] for line in result]
# im_show = draw_ocr(image, boxes, txts, scores, font_path='./fonts/simfang.ttf')
# im_show = Image.fromarray(im_show)
# im_show.save('result.jpg')

for i in result:
    if "回顶部" in i[1][0]:
        center= i[0][0]
        break
os.system(
    "adb shell input tap {} {}".format(center[0], center[1]))


'''联网百度文本识别'''
# result = subprocess.check_output("paddleocr --image_dir screen.png", shell=True)
# result = result.decode('utf-8')