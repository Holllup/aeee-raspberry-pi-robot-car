# 巡线 V1.4 技术文档

## 版本定位

- 版本号: `V1.4`
- 归档目录: `巡线V1.3——alarm`
- 当前归档代码: `line_following_minimal_v1_4.py`
- 配套模板文件: `alarm_template.png`
- 当前树莓派运行文件: `/home/jacob/line_following_minimal.py`

## 本版本主要功能

1. 保留基础巡线功能。
2. 保留中部横向颜色触发逻辑。
3. 颜色触发后:
   - 小车停止
   - 舵机转到 `120` 度
   - 进入牌子识别模式
   - 在 LCD 上显示识别结果
   - 等待 `5` 秒
   - 舵机回到 `180` 度
   - 小车恢复运行
4. 牌子识别从原来的规则判断升级为 `Template Matching`。
5. 当前已知模板只配置了一个模块: `ALARM`
6. 若匹配不到已知模板，则显示 `UNKNOWN`
7. 当识别到 `ALARM` 后:
   - 小车恢复巡线
   - 基准速度提升到 `45`
   - 持续 `5` 秒
   - LCD 显示加速运行状态
8. 丢线后不再直接停住，而是继续按记忆方向自动找线。
9. 在 `ALARM RUN` 这 `5` 秒内，额外驱动 `GPIO20` 和 `GPIO21` 做交替闪烁提示。

## 为什么从规则判断改成 Template Matching

前一版的 `ALARM` 识别是基于“上面有几个块、下面有几个块”的粗略布局规则实现的。

实际测试发现:

- 正确的 `ALARM` 牌子有时会因为透视、虚焦、连通域粘连而被判错。
- 其他几何图形牌子也可能因为布局碰巧接近，被误判成 `ALARM`。

因此本版本改用课件中提到的 `Template matching` 思路:

- 先准备一个标准 `ALARM` 模板
- 将实时检测到的牌子裁正、预处理
- 与模板做相关性匹配
- 匹配分数高于阈值才判定为 `ALARM`

## Template Matching 实现流程

### 1. 外框检测

- 利用粉色外框的 HSV 阈值定位整块牌子
- 只保留牌子区域，忽略外部背景

### 2. 透视校正

- 对检测到的牌子四边形做透视变换
- 将牌子拉正到统一尺寸: `240 x 240`

### 3. 内部图案预处理

- 按固定比例裁掉边框附近区域
- 对内部图案做二值化预处理
- 生成统一大小的符号图案图

### 4. 模板匹配

- 加载 `alarm_template.png`
- 使用 OpenCV 的 `cv2.matchTemplate(..., cv2.TM_CCOEFF_NORMED)`
- `TM_CCOEFF_NORMED` 本质上是归一化相关性匹配，更适合亮度存在波动的情况

### 5. 判定

- 模板匹配分数 `>= 0.70` 判定为 `ALARM`
- 否则判定为 `UNKNOWN`

## 当前关键参数

- 舵机识别角度: `CENTER_PURPLE_STOP_ANGLE = 120`
- 停车等待时间: `CENTER_PURPLE_STOP_WAIT_SECONDS = 5.0`
- 模板匹配阈值: `SIGN_TEMPLATE_MATCH_THRESHOLD = 0.70`
- 模板文件名: `ALARM_TEMPLATE_FILENAME = "alarm_template.png"`
- `ALARM` 后恢复巡线基准速度: `ALARM_RESUME_BASE_SPEED = 45`
- `ALARM` 后高速持续时间: `ALARM_RESUME_SECONDS = 5.0`
- `ALARM` 高速阶段转向增益倍率: `ALARM_RESUME_TURN_GAIN_SCALE = 1.25`
- 丢线后持续找线转向速度: `LOST_LINE_CONTINUOUS_TURN_SPEED = 32`
- 丢线后持续找线前进速度: `LOST_LINE_CONTINUOUS_FORWARD_SPEED = -2`
- 报警灯 GPIO:
  - `ALARM_RED_PIN = 20`
  - `ALARM_BLUE_PIN = 21`
- 报警灯交替闪烁周期: `ALARM_LED_BLINK_SECONDS = 0.5`

## 巡线直角弯优化参数

本版本保留了前面已经验证过的直角弯优化:

- `SHARP_ENTRY_BASE_SPEED = 16`
- `SHARP_ENTRY_STEERING_GAIN = 60.0`
- `SHARP_CORNER_PIVOT_SPEED = 40`
- `SHARP_RECOVERY_TURN_SPEED = 34`

目的:

- 在检测到锐角进入趋势时提前减速
- 丢线后用更大的左右轮差完成转弯恢复
- 在 `ALARM` 高速阶段，转向比例进一步放大
- 若冲出线外，不直接停住，而是继续转向自动搜索

## GPIO 提示灯逻辑

在识别到 `ALARM` 并进入 `ALARM RUN` 状态时:

- `GPIO20` 与 `GPIO21` 交替输出高电平
- 实现红/蓝交替闪烁效果
- 状态结束后自动全部关闭

实现方式为 Python 中直接使用 `RPi.GPIO` 控制，不依赖外部 C++ 程序。

## 调试视频

运行时可保存调试视频:

```bash
python3 /home/jacob/line_following_minimal.py --debug-video-output /home/jacob/v14_debug.mp4
```

调试视频中会显示:

- 巡线 ROI
- 中线参考线
- 颜色检测带
- 牌子外框与内部框
- `SIGN ALARM xx.xx` 或 `SIGN UNKNOWN xx.xx`

其中 `xx.xx` 为模板匹配分数，方便分析:

- 分数高且稳定: 说明模板匹配成功
- 分数低: 说明当前牌子与 `ALARM` 模板不匹配

## 常用运行命令

### 普通运行

```bash
python3 /home/jacob/line_following_minimal.py
```

### 带调试视频运行

```bash
python3 /home/jacob/line_following_minimal.py --debug-video-output /home/jacob/v14_debug.mp4
```

## 当前版本说明

- 本版本对应 `V1.4`
- 已经将 `ALARM` 识别改为模板匹配方式
- 识别逻辑相较上一版更不容易把其他牌子误判为 `ALARM`
- 识别到 `ALARM` 后会进入短时高速巡线状态
- 当前版本已经针对高速后直角弯和丢线卡住问题做了修复
- 当前版本已经加入 `GPIO20/21` 的 `ALARM RUN` 提示灯输出
- 如果后续还要识别更多模块，可以继续为每个模块新增模板文件，并扩展模板匹配逻辑
