---
name: archive-to-risedb
description: >
  将 RISE 项目的检查点或数据集归档到 /mnt/databackup/RISE/ 备份盘。
  触发条件：用户提到"归档"、"archive"、"备份到 databackup"、"移到备份盘"，
  以及任何涉及将 checkpoints/ 或 outputs/s1/ 下的数据移动到 /mnt/databackup 的操作。
  仅适用于 RISE 项目，目标固定为 /mnt/databackup/RISE/。
---

# RISE 数据归档到 /mnt/databackup

## 核心流程

```
理解需求 → 读取规范 → 确定数据日期 → 检查已有命名 → 执行移动 → 更新 README
```

## 1. 理解需求

向用户确认三个关键信息：
- **归档对象**：哪个目录/文件？（checkpoint 还是数据集？）
- **数据日期**：训练/采集数据最终日期（MMDD 格式），不是归档日期
- **保留规则**：归档后本地保留哪些？（用户当场指定，无默认规则）

如果用户未明确提供数据日期，从文件名中提取（如 `bi_s1_0612_*` → `0612`）。

## 2. 读取现行规范

**必须先读取** `/mnt/databackup/RISE/README.md`，了解：
- 当前目录结构（`data/` 和 `checkpoint/` 下有哪些 MMDD 目录）
- 命名规则：`data/<MMDD>/` 或 `checkpoint/<MMDD>/`，MMDD 为数据日期
- 已有索引条目，避免冲突

## 3. 确定目标路径和命名修正

目标路径 = `/mnt/databackup/RISE/<category>/<MMDD>/`

其中 `<category>` 为 `data` 或 `checkpoint`。

**关键检查**：`README.md` 中已有的 MMDD 目录是否以"数据日期"命名？如果发现已有目录使用归档日期而非数据日期命名（如 README 中日期范围和数据日期不一致），需要先修正（rename 目录 + 更新 README），再新增归档。

## 4. 执行步骤

### Step A: 修正已有命名（如果需要）
```bash
mv /mnt/databackup/RISE/<category>/<旧MMDD> /mnt/databackup/RISE/<category>/<新MMDD>
```

### Step B: 创建目标目录并移动
```bash
mkdir -p /mnt/databackup/RISE/<category>/<MMDD>/<子路径>
mv <本地路径> /mnt/databackup/RISE/<category>/<MMDD>/<子路径>/
```

### Step C: 移回本地保留项
```bash
mv /mnt/databackup/RISE/<category>/<MMDD>/<保留项> <本地原路径>/
```

### Step D: 更新 README.md

更新内容：
1. **目录规范** ASCII 图 — 添加新目录节点
2. **索引表** — 新增/修改对应 MMDD 条目，包含：

```
### <category>/<MMDD>/ — <简短描述>

| 项目 | 内容 |
|------|------|
| 机器人 | bi_s1_follower |
| 任务 | hang cloths |
| 训练数据 | <MMDD> （仅 checkpoint 需要此行） |
| 数据日期 | <日期范围或单日> |
| 归档日期 | <今天日期> |

<子集/checkpoint 列表>
```

**checkpoint 条目必须注明训练数据来源**，格式：`| 训练数据 | <MMDD> |`。

## 5. 验证

归档完成后：
```bash
echo "=== 备份 ===" && ls /mnt/databackup/RISE/<category>/<MMDD>/
echo "=== 本地 ===" && ls <本地原路径>/
```

确认备份完整、本地仅有指定保留项、README 描述准确。

## 注意事项

- 只归档到 `/mnt/databackup/RISE/`，不要归档到其他位置
- 命名始终使用**数据采集/使用日期**，不是归档操作日期
- checkpoint 条目在 README 中必须标注训练数据来源
- 所有 `mv` 操作在同一文件系统内进行（瞬时完成，不跨盘）
- 涉及到删除/覆盖的操作，先列出计划让用户确认
