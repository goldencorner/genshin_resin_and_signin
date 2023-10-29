# genshin_resin_and_signin_tool
米游社自动签到和原神树脂提醒。所有功能通过脚本实现，无需cookie，很少出现验证码，目前真正实用的签到工具

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
1. usb连接一台安装米游社的手机，并确认adb可用
```c
>>>adb devices
List of devices  attached
4b439028         device
```
2. 从server酱（https://sct.ftqq.com/after ）获取SendKey并替换.py文件中的相应代码
```python
                    url = "https://sctapi.ftqq.com/{SendKey}.send"
```
3. 首次启动脚本时会自动进行分辨率适配，这个过程中出现弹窗和异常界面将导致分辨率适配失败，因此启动前需要手动检查APP能否正常进入树脂界面和签到界面（保持未签到的状态，否则要第二天才能运行）
4. 运行resin_and_signin.py
## Tips
- 目前米游社app可以在两个手机上登录同一个账号而互不影响
- 此脚本可以持续运行
