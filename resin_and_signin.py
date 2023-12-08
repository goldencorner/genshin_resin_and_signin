import pickle
import subprocess
import warnings

from paddleocr import PaddleOCR, draw_ocr
import datetime
import traceback
from copy import deepcopy
import re

import cv2
import time
import os

import numpy as np
from PIL import Image, ImageDraw

import requests
import tkinter as tk
from tkinter import messagebox

# 定义截图函数
from jmetal.algorithm.multiobjective import NSGAII
from jmetal.core.problem import FloatProblem
from jmetal.core.solution import FloatSolution
from jmetal.operator import PolynomialMutation, SBXCrossover
from jmetal.util.comparator import DominanceComparator
from jmetal.util.termination_criterion import StoppingByTime
from matplotlib import pyplot as plt
from tqdm import tqdm

std_confidence = 0.9
package_name = 'com.mihoyo.hyperion'


def get_soc():
    """获取电量"""
    # 使用 ADB 命令获取电池电量
    adb_command = "adb shell dumpsys battery | findstr \"level\""
    battery_info = os.popen(adb_command).read()

    # 提取电量百分比
    battery_level = None
    for line in battery_info.splitlines():
        if "level" in line:
            parts = line.strip().split(":")
            if len(parts) == 2:
                battery_level = int(parts[1].strip())
                break

    if battery_level is not None:
        print("电池电量：{}%".format(battery_level))
    else:
        raise "无法获取电池电量信息"
    return battery_level


def get_resolution():
    # 运行ADB命令以获取屏幕分辨率
    adb_command = "adb shell wm size"
    result = subprocess.check_output(adb_command, shell=True)

    # 解析输出以获取分辨率
    output_str = result.decode('utf-8')
    lines = output_str.strip().split("\n")
    resolution = None

    for line in lines:
        if "Physical size:" in line:
            resolution = line.split(":")[1].strip()
    if resolution:
        print(f"设备分辨率: {resolution}")
    else:
        raise "未能获取设备分辨率"
    return resolution


class ImageResizer:
    def __init__(self, scale0, scale1, direction):
        self.scale0 = scale0
        self.scale1 = scale1
        self.direction = direction

    def resize(self, *args):
        scale0 = self.scale0
        scale1 = self.scale1
        if self.direction == 0:
            shape = args[0].shape
            resized_img = cv2.resize(args[0], (int(shape[1] * scale1), int(shape[0] * scale0)))
            return resized_img, args[1]
        elif self.direction == 1:
            shape = args[1].shape
            resized_img = cv2.resize(args[1], (int(shape[1] * scale1), int(shape[0] * scale0)))
            return args[0], resized_img

    def restore_coordinates(self, coord):
        if self.direction == 1:
            return [coord[0] / self.scale1, coord[1] / self.scale0]
        else:
            return coord


class ResolutionScaleProblem(FloatProblem):
    """Class representing problem Srinivas."""

    def __init__(self, template, screenshot, direction):
        """
        :param direction:
            0: 缩小template
            1：缩小screenshot
        """
        super().__init__()
        self.number_of_variables = 2
        self.number_of_objectives = 1
        self.number_of_constraints = 0

        self.obj_directions = [self.MINIMIZE]
        self.obj_labels = ["similarity"]

        self.lower_bound = [0.1, 0.1]
        self.upper_bound = [1, 1]

        self.template = template
        self.screenshot = screenshot
        self.direction = direction

    def evaluate(self, solution: FloatSolution) -> FloatSolution:
        template = self.template
        screenshot = self.screenshot

        scale0 = solution.variables[0]
        scale1 = solution.variables[1]

        template_ = deepcopy(template)
        screenshot_ = deepcopy(screenshot)
        image_resizer = ImageResizer(scale0, scale1, self.direction)
        resized_tuple = image_resizer.resize(template_, screenshot_)
        # 模板匹配
        try:
            result = cv2.matchTemplate(resized_tuple[0], resized_tuple[1], cv2.TM_CCOEFF_NORMED)
            # 找到匹配位置
            min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)
            solution.objectives[0] = -max_val
        except:
            solution.objectives[0] = 0

        return solution

    def get_name(self):
        return "solve_resolution_scale"


def solve_resolution_scale(template, screenshot, direction, max_seconds):
    problem = ResolutionScaleProblem(template, screenshot, direction)
    algorithm = NSGAII(
        problem=problem,
        population_size=20,
        offspring_population_size=20,
        mutation=PolynomialMutation(probability=1.0 / problem.number_of_variables, distribution_index=20.0),
        crossover=SBXCrossover(probability=0.9, distribution_index=20.0),
        termination_criterion=StoppingByTime(max_seconds=max_seconds),
        dominance_comparator=DominanceComparator(),
    )

    print(f"正在适配分辨率，持续{max_seconds}s")

    algorithm.start_computing_time = time.time()
    algorithm.solutions = algorithm.create_initial_solutions()
    algorithm.solutions = algorithm.evaluate(algorithm.solutions)
    algorithm.init_progress()

    with tqdm(total=max_seconds, desc="Time Progress") as pbar:
        start_time = time.time()
        while not algorithm.stopping_condition_is_met():
            elapsed_time = time.time() - start_time
            pbar.update(int(elapsed_time - pbar.n))  # 更新进度条
            algorithm.step()
            front = algorithm.get_result()
            best_objective_val = np.inf
            best_solution = front[0]
            for solution in front:
                if solution.objectives[0] < best_objective_val:
                    best_solution = solution
                    best_objective_val = -solution.objectives[0]
            algorithm.update_progress()
            pbar.set_description(f"best confidence {best_objective_val:.2f}")
            if best_objective_val > 0.95:
                break

    algorithm.total_computing_time = time.time() - algorithm.start_computing_time
    print("Algorithm (continuous problem): " + algorithm.get_name())
    print("Problem: " + problem.get_name())
    print("Computing time: " + str(algorithm.total_computing_time))
    # print("optimization result: cycle=", cycle, "best_objective_val=", best_objective_val)

    return ImageResizer(best_solution.variables[0], best_solution.variables[1], direction), best_objective_val


def get_screenshot():
    os.system("adb shell screencap -p /sdcard/screen.png")
    datetime_now = datetime.datetime.now()
    formatted_now = datetime_now.strftime('%d_%H_%M_%S')
    screenshot_path = f"./screenshots/screen_{formatted_now}.png"
    screenshot_path=f"./screenshots/screen.png"
    os.system(f"adb pull /sdcard/screen.png {screenshot_path}")

    # todo 直接读取到内存
    return screenshot_path


# 定义模板匹配函数
def match_and_click(template_path, save_coordinates=True):
    global std_confidence
    # 加载模板
    template = cv2.imread(template_path, cv2.IMREAD_GRAYSCALE)
    # 截图
    screenshot_path = get_screenshot()
    # 加载截图
    screenshot = cv2.imread(screenshot_path)
    # 转为灰度图像
    gray = cv2.cvtColor(screenshot, cv2.COLOR_BGR2GRAY)
    resolution = get_resolution()

    def calibration():
        # 尺度校准
        for desctiption, direction in {"缩小截图": 1, "缩小模板": 0}.items():
            # if direction ==0:continue
            print("尝试", desctiption)
            image_resizer, confidence = solve_resolution_scale(template, gray, direction, 60)
            std_confidence = confidence - 0.05  # 增加0.05的容错率
            std_resolution = get_resolution()
            if confidence > 0.95:
                break
        if confidence < 0.95:
            raise "分辨率适配失败，重试有概率解决问题"
        if not os.path.exists("scale_info.pkl"):
            with open("scale_info.pkl", "wb") as f:
                scale_dict = {"default": [std_resolution, image_resizer, std_confidence]}
                pickle.dump(scale_dict, f)
        else:
            with open("scale_info.pkl", "rb") as f:
                scale_dict = pickle.load(f)
            for desctiption, direction in {"缩小截图": 1, "缩小模板": 0}.items():
                print("尝试", desctiption)
                image_resizer, confidence = solve_resolution_scale(template, gray, direction, 60)
                std_confidence = confidence - 0.05  # 增加0.05的容错率
                std_resolution = get_resolution()
                if confidence > 0.95:
                    break
            if confidence < 0.95:
                raise "分辨率适配失败，重试有概率解决问题"
            print(f"为模板{template_path}添加专用分辨率参数")
            scale_dict[template_path] = [std_resolution, image_resizer, std_confidence]
            with open("scale_info.pkl", "wb") as f:
                pickle.dump(scale_dict, f)
        return image_resizer, std_confidence

    '''第一次机会'''
    if os.path.exists("scale_info.pkl"):
        with open("scale_info.pkl", "rb") as f:
            scale_dict = pickle.load(f)
            if template_path in scale_dict.keys():
                [std_resolution, image_resizer, std_confidence] = scale_dict[template_path]
            else:
                [std_resolution, image_resizer, std_confidence] = scale_dict["default"]
        if std_resolution != resolution:
            image_resizer, std_confidence = calibration()
    else:
        image_resizer, std_confidence = calibration()
    resized_tuple = image_resizer.resize(template, gray)
    # 模板匹配
    result = cv2.matchTemplate(resized_tuple[0], resized_tuple[1], cv2.TM_CCOEFF_NORMED)
    # 找到匹配位置
    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)
    confidence = max_val
    print("CONFIDENCE1", confidence)
    top_left = list(max_loc)

    # debug 创建一个图形窗口并显示截图
    # fig, ax = plt.subplots()
    # ax.imshow(screenshot)
    # ax.plot(max_loc[0], max_loc[1], 'ro')
    # plt.savefig("test.jpg")
    # plt.show()
    # plt.close()

    '''第二次机会'''
    if max_val < std_confidence:
        print("默认参数失效，尝试更新参数")
        calibration()
        if os.path.exists("scale_info.pkl"):
            with open("scale_info.pkl", "rb") as f:
                scale_dict = pickle.load(f)
                if template_path in scale_dict.keys():
                    [std_resolution, image_resizer, std_confidence] = scale_dict[template_path]
                else:
                    [std_resolution, image_resizer, std_confidence] = scale_dict["default"]
            if std_resolution != resolution:
                image_resizer, std_confidence = calibration()
        else:
            image_resizer, std_confidence = calibration()
        resized_tuple = image_resizer.resize(template, gray)
        # 模板匹配
        result = cv2.matchTemplate(resized_tuple[0], resized_tuple[1], cv2.TM_CCOEFF_NORMED)
        # 找到匹配位置
        min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)
        confidence = max_val
        print("CONFIDENCE2", confidence)
        top_left = list(max_loc)
        if max_val < std_confidence:
            raise "匹配失败，未知错误"

    # debug
    # fig, ax = plt.subplots()
    # ax.imshow(gray)
    # ax.scatter(top_left[0], top_left[1], s=100, c='red', marker='o')
    # fig.savefig("test.jpg")
    # plt.show()
    # plt.close()
    h, w = template.shape
    bottom_right = (top_left[0] + w, top_left[1] + h)
    center = [(top_left[0] + bottom_right[0]) / 2, (top_left[1] + bottom_right[1]) / 2]
    center = image_resizer.restore_coordinates(center)

    if confidence > std_confidence:
        # debug 创建一个图形窗口并显示截图
        if save_coordinates:
            fig, ax = plt.subplots()
            ax.imshow(screenshot)
            ax.plot(center[0], center[1], 'ro')
            plt.savefig("test.jpg")
            # plt.show()
            plt.close()

        # 点击模板中心位置
        os.system(
            "adb shell input tap {} {}".format(center[0], center[1]))
        time.sleep(2)
        return True
    else:
        print("NOT FOUND")
        return False


def turn2main_page():
    # 启动应用程序
    activity_name = '.main.HyperionMainActivity'
    subprocess.call(['adb', 'shell', 'am', 'start', '-n',
                     f'{package_name}/{package_name + activity_name}'])
    time.sleep(8)

    # 处理主界面可能出现的弹窗
    screenshot_path = get_screenshot()
    result = get_OCR_result(screenshot_path)

    for i in result:
        if "青少年模式" in i[1][0]:
            try:
                print("try confirming teenager mode")
                match_and_click("./templates/i_get_it.png")
            except Exception as e:
                warnings.warn("意料之外的识别错误: {}".format(str(e)))
            time.sleep(3)
            break
        if "下次再说" in i[1][0]:
            try:
                print("try skipping update")
                match_and_click("./templates/skip_update.png")
            except Exception as e:
                warnings.warn("意料之外的识别错误: {}".format(str(e)))
            time.sleep(3)
            break
        if "米游社没有响应" in i[1][0]:
            relaunch_APP()
            break
        if "回顶部" in i[1][0]:
            center = i[0][0]
            os.system(
                "adb shell input tap {} {}".format(center[0], center[1]))
            time.sleep(3)
            break
        if "发现" in i[1][0]:
            center = i[0][0]
            os.system(
                "adb shell input tap {} {}".format(center[0], center[1]))
            time.sleep(3)
            break


def turn2resin_page():
    turn2main_page()
    match_and_click("./templates/myself.png")

    screenshot_path = get_screenshot()
    result = get_OCR_result(screenshot_path)

    for i in result:
        if "成就达成数" in i[1][0]:
            center = i[0][0]
            os.system(
                "adb shell input tap {} {}".format(center[0], center[1]))
            break
    time.sleep(20)


def relaunch_APP():
    print("relaunch APP")
    subprocess.call(['adb', 'shell', 'am', 'force-stop',
                     f'{package_name}'])
    time.sleep(8)
    turn2main_page()
    time.sleep(3)

def get_OCR_result(screenshot_path):
    ocr = PaddleOCR(use_angle_cls=True, lang="ch")  # need to run only once to download and load model into memory
    result = ocr.ocr(screenshot_path, cls=True)
    result = result[0]

    # debug   显示结果
    # image = Image.open(img_path).convert('RGB')
    # boxes = [line[0] for line in result]
    # txts = [line[1][0] for line in result]
    # scores = [line[1][1] for line in result]
    # im_show = draw_ocr(image, boxes, txts, scores, font_path='./fonts/simfang.ttf')
    # im_show = Image.fromarray(im_show)
    # im_show.save('result.jpg')

    return result


def monitor_resin():
    """从截图上识别体力值"""
    # 截图
    screenshot_path = get_screenshot()

    result = get_OCR_result(screenshot_path)

    result = str(result)
    result = result.replace(" ", "")

    '''联网百度文本识别'''
    # result = subprocess.check_output("paddleocr --image_dir screen.png", shell=True)
    # result = result.decode('utf-8')

    for pattern in [r'(\d+)/160', r'(\d+)／160']:
        match = re.search(pattern, result)
        if match:
            # 获取匹配到的值
            text = match.group(1)
            print(f"The value is: {text}")
            break
        else:
            print("Pattern not found in the string.")

    print("识别结果为:", text)
    current_resin = int(text)
    time_till_full = (160 - current_resin) * 8
    t = datetime.datetime.now()
    delta = datetime.timedelta(minutes=time_till_full)

    t_full = t + delta
    print(current_resin, ", ", t_full, "完全恢复")
    print()

    return current_resin


def sign_in():
    print("正在签到")
    # 启动应用程序
    turn2main_page()
    match_and_click("./templates/sign_in.png")
    time.sleep(8)
    match_and_click("./templates/draw.png")
    time.sleep(5)
    sign_result=False
    # 处理主界面可能出现的弹窗
    screenshot_path = get_screenshot()
    result = get_OCR_result(screenshot_path)

    for i in result:
        if "签到成功" in i[1][0]:
            sign_result=True
            break


    return sign_result


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
    url = "https://sctapi.ftqq.com/SCT205640T2og2nNrP2BE8mR0H3sRbShJ4.send"
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
subprocess.run(["adb", "shell", "settings", "put", "system",
                "screen_brightness", "10"])


def balance_SOC_or_sleep(seconds):
    soc = get_soc()
    if soc > 90:
        # 增加CPU负载
        print(f"stree cpu {seconds}s")
        t0 = time.time()
        turn2main_page()
        while time.time() - t0 < seconds:
            os.system("adb shell input swipe 500 1800 500 1000")
            time.sleep(0.2)
        time.sleep(3)

        turn2main_page()

    else:
        print(f"sleep {seconds}s")
        time.sleep(seconds)


while True:
    # 检查今天是否已经签到
    # 加载上次签到的日期
    try:
        with open("last_sign_in_day.pkl", "rb") as f:
            last_sign_in_day = pickle.load(f)
    except FileNotFoundError:
        last_sign_in_day = None
    # 获取当前时间
    now = datetime.datetime.now() + datetime.timedelta(hours=-3)  # 米游社在零点会跳出一个弹窗 三点钟再签到避免这种情况
    # 如果当前时间是今天，并且上次签到不是今天，则执行签到
    if now.day != last_sign_in_day:
        try:
            result = sign_in()
            if result:
                last_sign_in_day = now.day
                try:
                    send_wechat(f"签到成功; soc:{get_soc()}")
                except:
                    pop_up_windows("签到成功")
                # 保存签到日期到磁盘上
                with open("last_sign_in_day.pkl", "wb") as f:
                    pickle.dump(last_sign_in_day, f)
        except Exception as e:
            traceback.print_exc()

    start_time = time.time()
    while time.time() - start_time <= reset_threshold:
        try:
            turn2resin_page()
            current_resin = monitor_resin()
            if current_resin >= 159:
                try:
                    send_wechat(f"current_resin:{current_resin}; soc:{get_soc()}")
                except:
                    pop_up_windows("请求错误")
                # 进入死循环，每10分钟查看一次，直到体力被消耗
                while monitor_resin() >= current_resin:
                    balance_SOC_or_sleep(reset_threshold)
                    turn2resin_page()

            fault_num = 0
        except Exception as e:
            traceback.print_exc()
            fault_num += 1
            relaunch_APP()

        balance_SOC_or_sleep(reset_threshold)

    if fault_num * reset_threshold > time_tolerance:
        try:
            send_wechat("出现异常界面")
        except:
            pop_up_windows("出现异常界面")
        fault_num = 0
