# pairwise-demosaic-badpixel

## 领域

游戏、图形与多媒体

## 主题

去马赛克坏点

## 说明

坏点检测、Bayer 相位与并行确定性。

初始状态下 `imaging/demosaic_badpixel.py` 中所有函数抛出 `NotImplementedError`，测试全部失败。
修复实现后运行：

```bash
python -m pytest -q
```

期望全部通过。
