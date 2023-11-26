# genshin_resin_and_signin_tool
米游社自动签到和原神树脂提醒。所有功能通过脚本实现，无需cookie，很少出现验证码，目前真正实用的签到工具  
**2023年11月26日**最新版本米游社V2.63.1实测可用

<img src="https://github.com/goldencorner/dataset/blob/main/_images/demo_resin.png" alt="demo_resin" width="388" height="400">

## 功能
- 每天一次在原神米游社签到
- 树脂快满时发送提醒到你的微信
## 环境配置(windows x64)
1.安装百度OCR库(参考https://github.com/PaddlePaddle/PaddleOCR/blob/release/2.7/doc/doc_ch/quickstart.md#22)
```bash
pip install paddlepaddle -i https://mirror.baidu.com/pypi/simple
pip install "paddleocr>=2.0.1" -i https://mirror.baidu.com/pypi/simple
```
2.安装其他必要的库
```bash
pip install paddleocr opencv-python pillow requests jmetalpy==1.5.5 matplotlib tqdm -i https://mirror.baidu.com/pypi/simple
```
## 用法
1. USB连接一台安装米游社(测试版本：V2.63.1)的手机，并确认adb可用
```c
>>>adb devices
List of devices  attached
4b439028         device
```
2. 从server酱（https://sct.ftqq.com/after ）获取SendKey并替换resin_and_signin.py文件中的相应代码
```python
def send_wechat(text):
    url = "https://sctapi.ftqq.com/SCT205640T2og2nNrP2BE8mR0H3sRbShJ4.send"  # 替换为自己的SendKey
    params = {
        "title": text
    }
    response = requests.post(url, data=params, proxies=None, timeout=10)
    print(response.text)
```
3. 首次启动脚本时会自动进行分辨率适配，这个过程中出现弹窗和异常界面将导致分辨率适配失败，因此启动前需要手动检查APP能否正常进入树脂界面和签到界面（需要保持未签到的状态，否则要第二天才能运行）
4. 运行resin_and_signin.py
## 特性
- 分辨率自适应：模板匹配时支持自动缩放，从而适配不同的手机分辨率
- **电量控制**：电量高于50%时，此脚本会自动增加CPU负载。此功能需要配合低功率的充电口使用（推荐使用主板上USB2.0的口再连接USB延长线）。若长期挂机使用，务必确保手机电量能够稳定处于100%以下，否则电池过充可导致**鼓包和其他危险**
- 基于前沿的OCR识别工具，容错率较高
