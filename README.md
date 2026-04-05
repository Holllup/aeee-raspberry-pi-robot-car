# 树莓派小车项目 GitHub 整理版

这是一个面向 GitHub 展示与归档的整理版目录。当前结构按课程周次组织，先展示前期基础工作，再展示后续独立检测与比赛任务。

## 项目说明

- 项目主题：基于树莓派的小车巡线与任务扩展
- 当前整理逻辑：先按课程阶段分类，再在第 6-7 周内按比赛任务顺序展示
- 当前仓库定位：精选版，不包含调试视频、原始录像、缓存文件和测试输出

## 目录结构

```text
.
├── docs/
│   ├── hardware/
│   ├── usage/
│   ├── tasks/
│   ├── offline_detection/
│   └── benchmark/
├── support/
│   ├── offline_detection/
│   ├── remote_control/
│   └── benchmark/
├── week_05_aeee_project_5_foundation/
│   ├── code/
│   ├── task2/
│   ├── task3/
│   ├── task4/
│   ├── task5/
│   └── task6/
└── week_06_07_competition_tasks/
    ├── 01_general_challenge_description/
    ├── 02_line_following/
    ├── 03_alarm/
    ├── 04_traffic_light/
    ├── 05_play_music/
    ├── 06_color_shape_sorting/
    ├── 07_obstacle_detour/
    ├── 08_kick_football_bonus/
    └── 09_maze_navigation_bonus/
```

## 阶段说明

### 第 5 周：前期基础工作

`week_05_aeee_project_5_foundation/` 对应 `aeee-project-5`，是这门课第 5 周的前期工作，也是后续任务的基础阶段。

- `code/`：前期基础代码
- `task2/` 到 `task6/`：保留原始周任务资料、图片、文档和表格

### 第 6-7 周：后续独立检测与比赛任务

`week_06_07_competition_tasks/` 是后续主线，内部继续按比赛任务 1-9 排列。

- `01_general_challenge_description/`：总任务说明预留目录
- `02_line_following/`：基础巡线
- `03_alarm/`：报警牌识别
- `04_traffic_light/`：红绿灯识别
- `05_play_music/`：播放音乐
- `06_color_shape_sorting/`：颜色与形状分拣
- `07_obstacle_detour/`：避障绕行
- `08_kick_football_bonus/`：踢足球 Bonus
- `09_maze_navigation_bonus/`：迷宫导航 Bonus

比赛任务目录内部统一保留：

- `code/`：当前任务的代表性主代码
- `templates/`：当前任务运行所需的模板素材

## 通用资料与支撑目录

- `docs/hardware/`：接线、车辆控制、旧相机相关硬件说明
- `docs/usage/`：LCD、舵机、相机参数和使用文档
- `docs/tasks/`：任务功能对应的阶段性技术文档
- `docs/offline_detection/`：离线检测说明与调参文档
- `docs/benchmark/`：性能测试说明和结果记录
- `support/offline_detection/`：离线检测与分析代码
- `support/remote_control/`：远程启动/停止辅助脚本
- `support/benchmark/`：性能测试与采样脚本

## 当前整理原则

- `aeee-project-5` 不再作为 `02_line_following` 的附属内容，而是独立作为第 5 周基础阶段存在
- 第 6-7 周目录保留 1-9 的任务顺序，方便 Finder 和 GitHub 中稳定排序
- 每个比赛任务目录只保留代表性的主线代码，不把所有历史版本全部复制进来
- 已将当前主线代码依赖的模板素材放入对应任务目录
- 调试视频、原始录像、分析结果文件保留在原始资料目录，不纳入这个 GitHub 整理版

## 后续建议

上传 GitHub 前，建议在这个目录内执行：

```bash
git init
git add .
git commit -m "Prepare curated Raspberry Pi car repository"
```

如果你后续补充模板素材，只需要放到对应比赛任务的 `templates/` 目录即可。
