import pickle
import subprocess

# adb connect 192.168.137.16:40803
import datetime
import traceback

import cv2
import time
import os
import subprocess
from PIL import Image
import pytesseract
import requests
import tkinter as tk
from tkinter import messagebox

# 定义截图函数
from matplotlib import pyplot as plt


def get_screenshot():
    os.system("adb shell screencap -p /sdcard/screen.png")
    os.system("adb pull /sdcard/screen.png .")


# 定义模板匹配函数
def match_template(template, image):
    # 转为灰度图像
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # 模板匹配
    result = cv2.matchTemplate(gray, template, cv2.TM_CCOEFF_NORMED)
    # 找到匹配位置
    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)
    top_left = max_loc
    h, w = template.shape
    bottom_right = (top_left[0] + w, top_left[1] + h)
    center = (int((top_left[0] + bottom_right[0]) / 2), int((top_left[1] + bottom_right[1]) / 2))
    return center, max_val


def match_and_click(template_path):
    # 加载模板
    template = cv2.imread(template_path, cv2.IMREAD_GRAYSCALE)
    # 截图
    get_screenshot()
    # 加载截图
    screenshot = cv2.imread("screen.png")
    # 匹配模板
    match_loc, confidence = match_template(template, screenshot)

    # 创建一个图形窗口并显示截图
    # fig, ax = plt.subplots()
    # ax.imshow(screenshot)
    # ax.plot(match_loc[0], match_loc[1], 'ro')
    # plt.show()
    # plt.close()
    print(f"confidence:{confidence}")
    if confidence > 0.9:
        # 点击模板中心位置
        os.system("adb shell input tap {} {}".format(match_loc[0], match_loc[1]))
        time.sleep(2)
        return True
    else:
        print("NOT FOUND")
        return False


def turn2resin_page():
    # 启动应用程序
    package_name = 'com.mihoyo.hyperion'
    activity_name = '.main.HyperionMainActivity'
    subprocess.call(['adb', 'shell', 'am', 'start', '-n', f'{package_name}/{package_name + activity_name}'])
    time.sleep(2)
    match_and_click("./templates/myself.png")
    match_and_click("./templates/my_roles.png")
    time.sleep(8)


def monitor_resin():
    # 加载模板
    template = cv2.imread("./templates/resin_position.png", cv2.IMREAD_GRAYSCALE)
    # 截图
    get_screenshot()
    # 加载截图
    screenshot = cv2.imread("screen.png")
    # 转为灰度图像
    gray = cv2.cvtColor(screenshot, cv2.COLOR_BGR2GRAY)
    # 模板匹配
    result = cv2.matchTemplate(gray, template, cv2.TM_CCOEFF_NORMED)
    # 找到匹配位置
    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)
    top_left = max_loc
    confidence=max_val
    print(f"confidence:{confidence}")
    # 打开图像文件
    img = Image.open("screen.png")
    # 指定需要识别文本的区域坐标
    box = (106, 1824, top_left[0], 1916)  # (左上角 x, 左上角 y, 右下角 x, 右下角 y)

    # 裁剪指定区域
    crop_img = img.crop(box)

    # 将图像转换为灰度图像
    gray_img = crop_img.convert('L')

    # 将灰度图像转换为文本
    text = pytesseract.image_to_string(gray_img, lang='eng')
    current_resin = int(text)
    time_till_full = (160-current_resin) * 8
    t = datetime.datetime.now()
    delta = datetime.timedelta(minutes=time_till_full)

    t_full = t + delta
    print(current_resin,", ",t_full, "完全恢复")
    print()

    return current_resin


def sign_in():
    print("正在签到")
    # 启动应用程序
    package_name = 'com.mihoyo.hyperion'
    activity_name = '.main.HyperionMainActivity'
    subprocess.call(['adb', 'shell', 'am', 'start', '-n', f'{package_name}/{package_name + activity_name}'])
    time.sleep(2)
    match_and_click("./templates/sign_in.png")
    time.sleep(8)
    result = match_and_click("./templates/draw.png")
    return result


def pop_up_windows(str):
    # 创建一个Tk对象
    root = tk.Tk()
    root.withdraw()
    # 获取屏幕的宽度和高度
    screen_width = root.winfo_screenwidth()
    screen_height = root.winfo_screenheight()

    # 创建一个Toplevel窗口，并将它置顶
    top = tk.Toplevel(root)
    top.title("Title")
    top.lift()
    top.attributes("-topmost", True)

    # 计算Toplevel窗口的位置，使其居中显示
    top_width = 200
    top_height = 100
    x = (screen_width - top_width) // 2
    y = (screen_height - top_height) // 2
    top.geometry('{}x{}+{}+{}'.format(top_width, top_height, x, y))

    # 在Toplevel窗口中显示一段字符串
    label = tk.Label(top, text=str)
    label.pack()

    # 设置Toplevel窗口关闭时，同时关闭root窗口
    def on_closing():
        root.destroy()

    top.protocol("WM_DELETE_WINDOW", on_closing)

    # 进入Tk事件循环，等待事件处理
    root.mainloop()


def send_wechat(text):
    url = "https://sctapi.ftqq.com/SCT205640T7uk4aHGd7sNje9MwcreSHWcA.send"
    params = {
        "title": text
    }
    response = requests.post(url, data=params, proxies=None, timeout=10)
    print(response.text)


fault_num = 0
reset_threshold = 4 * 60
time_tolerance = 5 * 60 * 60
os.system("adb devices")
# 调用adb shell命令将亮度设置为0
subprocess.run(["adb", "shell", "settings", "put", "system", "screen_brightness", "0"])
while True:
    # 检查今天是否已经签到
    # 加载上次签到的日期
    try:
        with open("last_sign_in_day.pkl", "rb") as f:
            last_sign_in_day = pickle.load(f)
    except FileNotFoundError:
        last_sign_in_day = None
    # 获取当前时间
    now = datetime.datetime.now()
    # 如果当前时间是今天，并且上次签到不是今天，则执行签到
    if now.day != last_sign_in_day:
        result = sign_in()
        if result:
            last_sign_in_day = now.day
            try:
                send_wechat("签到成功")
            except:
                pop_up_windows("签到成功")
            # 保存签到日期到磁盘上
            with open("last_sign_in_day.pkl", "wb") as f:
                pickle.dump(last_sign_in_day, f)

    turn2resin_page()
    start_time = time.time()
    while time.time() - start_time <= reset_threshold:
        try:
            current_resin = monitor_resin()
            if current_resin >= 159:
                try:
                    send_wechat(f"current_resin:{current_resin}")
                except:
                    pop_up_windows("请求错误")
                # 进入死循环，每10分钟查看一次，直到体力被消耗
                while monitor_resin() >= current_resin:
                    turn2resin_page()
                    time.sleep(reset_threshold)
            fault_num = 0
        except Exception as e:
            traceback.print_exc()
            match_and_click("./templates/i_get_it.png")
            fault_num += 1

        time.sleep(reset_threshold)

    if fault_num * reset_threshold > time_tolerance:
        try:
            send_wechat("出现异常界面")
        except:
            pop_up_windows("出现异常界面")
