# genshin_resin_and_signin_tool
米游社自动签到和原神树脂提醒。所有功能通过脚本实现，无需cookie，很少出现验证码，目前真正实用的签到工具
![demo_resin](https://github.com/goldencorner/dataset/blob/main/_images/demo_resin.png)

## 功能
- 每天一次在原神米游社签到
- 树脂快满时发送提醒到你的微信
## 用法
1. usb连接一台安装米游社的手机，并确认adb可用
```c
>>>adb devices
List of devices attached
4b439028        device
```

2. 根据自己的手机截图替换templates文件夹中的图片
3. 从server酱（https://sct.ftqq.com/after ）获取SendKey并替换
```python
                    url = "https://sctapi.ftqq.com/SCT205640T7uk4aHXd7sNje9MwcreSHWcA.send"
```
4. 运行
