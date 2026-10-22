# 智鉴：伪造人脸检测系统

智鉴是一款面向图片、视频的伪造人脸检测作品。系统以 FADR-CLIP 最佳主模型为核心，提供文件上传、CPU/GPU 选择、人脸预处理、异步检测、实时逐帧可视化、注意力展示、分析记录管理和报告导出等功能。

当前功能版本：**0.16.0**。

作品仓库：<https://github.com/V1nce23/zhijian-fadr-clip>

发布资源：<https://github.com/V1nce23/zhijian-fadr-clip/releases/tag/v0.16.0>

GitHub 源码包含演示样例、人脸对齐模型和系统实现。首次运行前，请从上述 Release 下载 `model.safetensors`，放到 `system/models/fadr_clip/model.safetensors`，并按下文核对 SHA-256。主权重使用独立附件分发，不包含在 GitHub 自动生成的源码 ZIP 中。

[下载 FADR-CLIP 最佳主模型权重](https://github.com/V1nce23/zhijian-fadr-clip/releases/download/v0.16.0/model.safetensors)

## 1. 作品功能

| 功能 | 说明 |
| --- | --- |
| 图片检测 | 提取并对齐最大人脸，输出检测分数与判定结果 |
| 视频检测 | 最多均匀抽取 32 帧，对有效人脸帧分别检测并汇总 |
| 设备选择 | 每个任务可选择 CPU 或 GPU；GPU 不可用时禁用对应选项 |
| 判断模式 | 支持均衡、低误报、低漏报和自定义阈值 |
| 过程展示 | 展示抽帧、人脸对齐、逐帧检测和报告生成进度 |
| 逐帧可视化 | 使用重叠卡片浏览视频画面、人脸与单帧分数 |
| 注意力展示 | 展示路由加权空间注意力，可调节叠加效果 |
| 记录管理 | 查看详情、取消任务、单独或批量删除分析记录 |
| 报告导出 | 支持 PDF、Word、HTML、CSV、JSON，可选择包含技术详情 |
| 用户管理 | 支持管理员与分析用户，报告访问受权限控制 |

本作品仅处理图片和视频，视频声音不参与检测。

## 2. 材料目录

```text
作品材料/
├─ README.md                         作品介绍、部署和评测说明
├─ 提交与模型资源说明.md             500 MB 限制下的资源组织说明
├─ 文件校验清单.json                 本材料文件的大小及 SHA-256
├─ test-data/
│  ├─ photo/fake/                    3 张伪造图片样例
│  ├─ photo/real/                    3 张真实图片样例
│  ├─ video/fake/                    1 个伪造视频样例
│  └─ video/real/                    1 个真实视频样例
└─ system/
   ├─ apps/web/                     前端源码与前端测试
   ├─ services/api/                 API、异步任务、检测及报告导出
   │  ├─ app/
   │  ├─ alembic/                   数据库结构迁移
   │  └─ tests/                     后端自动化测试
   ├─ models/
   │  ├─ face/                      人脸关键点模型
   │  └─ fadr_clip/                 检测模型权重、配置和校验信息
   ├─ tools/                        模型导出及一致性验证工具
   ├─ benchmarks/deepfake30/        评测辅助工具
   ├─ .env.example                  环境配置示例
   ├─ compose.local-cpu.yaml        CPU 部署配置
   └─ compose.local.yaml            CPU/GPU 部署配置
```

完整材料包含全部模型。若使用面向 500 MB 上传限制的主体压缩包，需先按《提交与模型资源说明.md》补齐 FADR-CLIP 主权重；其他所需模型和配置已在主体包内。

材料不包含历史 Git 仓库、原环境账号密码、数据库记录、用户上传文件或构建缓存。第三方组件的许可证及必要署名保留在对应目录。

## 3. 运行环境

- 支持 Docker Desktop 或 Docker Engine 与 Compose 插件的环境。
- 浏览器可访问本机的 3000 端口；本机 8100、5433、6380、9000、9001 端口也需可用。
- CPU 可以完成全部检测流程；GPU 加速需要可被 Docker 识别的 NVIDIA 显卡及相应驱动环境。
- 模型权重约 1.23 GB，请预留足够的内存和磁盘空间用于推理、镜像构建及文件存储。
- 首次构建需要联网下载基础镜像与软件依赖。模型及镜像准备完成后，正常检测不依赖外部推理服务，可离线使用。

本材料不包含 Docker 镜像离线包。

## 4. 首次部署

### 4.1 检查模型文件

部署前确认以下文件存在：

```text
system/models/fadr_clip/model.safetensors
system/models/fadr_clip/model_config.json
system/models/fadr_clip/manifest.json
system/models/face/shape_predictor_81_face_landmarks.dat
```

FADR-CLIP 主权重的 SHA-256 为：

```text
9da257ce6a69cd22371c7f36a3ea5702b860da2354e942ea3d42689fd3caf0e7
```

在 PowerShell 中可执行：

```powershell
Get-FileHash .\system\models\fadr_clip\model.safetensors -Algorithm SHA256
```

无需重新训练或再次导出权重。

### 4.2 配置账号与密码

启动 Docker Desktop，等待引擎就绪，在 `system` 目录打开 PowerShell：

```powershell
Copy-Item .env.example .env
notepad .env
```

设置 `POSTGRES_PASSWORD`、`MINIO_ROOT_PASSWORD`、`ADMIN_EMAIL`、`ADMIN_PASSWORD` 等参数。`.env.example` 中的账号和密码是配置占位值，运行时应自行填写。管理员账户会在首次启动时创建。

### 4.3 启动 CPU 版本

没有 NVIDIA GPU 或先进行基础试用时，在 `system` 目录执行：

```powershell
docker compose -f compose.local-cpu.yaml up --build -d
```

### 4.4 启动可选择 CPU/GPU 的版本

已配置 Docker GPU 透传的 NVIDIA GPU 环境执行：

```powershell
docker compose -f compose.local.yaml up --build -d
```

该部署方式支持在网页中逐任务选择 CPU 或 GPU，不需要每次切换部署配置。GPU 是否可用以页面显示的检测服务状态为准。

### 4.5 打开系统

浏览器访问 <http://localhost:3000>，使用 `.env` 中设置的管理员账号登录。

API 地址为 <http://localhost:8100/api/v1>，API 文档为 <http://localhost:8100/api/v1/docs>。

## 5. 样例测试流程

1. 登录系统，在“提交检测”中选择 CPU 或 GPU。
2. 保持默认“均衡”模式，或选择其他判断模式。
3. 上传 `test-data/photo/` 中的图片或 `test-data/video/` 中的视频。
4. 提交后查看预处理、逐帧检测与报告生成进度。
5. 在报告中浏览人脸卡片、单帧分数、总体结论及注意力图。
6. 需要模型版本与层权重时展开“技术详情”。
7. 选择 PDF、Word、HTML、CSV 或 JSON 导出；研究分析可勾选“包含技术详情”。
8. 在“分析记录”中重新打开报告，或删除相应记录。

`fake`、`real` 为样例来源标签，便于对照，并不保证模型对每个样例都判定正确。样例仅用于演示功能，不构成完整性能评测集。

## 6. 检测方法与结果解释

系统仅运行 FADR-CLIP 最佳主模型。图片直接进入人脸处理；视频按原始总帧数均匀采样最多 32 帧，短视频不通过复制帧凑足数量。

图片和视频帧均进行最大人脸提取、五点对齐及裁剪，得到 **256×256 RGB** 人脸，再按模型要求转换为 **224×224 RGB**，执行 CLIP 归一化：

```text
mean = [0.48145466, 0.4578275, 0.40821073]
std  = [0.26862954, 0.26130258, 0.27577711]
```

模型的伪造类别 softmax 输出作为单帧分数；视频总体分数取有效人脸帧分数的算术平均。无人脸帧不参与平均，全部没有有效人脸时检测失败。

| 判断模式 | 视频总体阈值 | 图片/单帧阈值 |
| --- | ---: | ---: |
| 均衡（默认） | 0.67 | 0.85 |
| 低误报 | 0.93 | 0.99 |
| 低漏报 | 0.50 | 0.50 |
| 自定义 | 用户填写 | 用户填写 |

分数**严格大于**对应阈值时判为“疑似伪造”，否则判为“疑似真实”。判断模式改变阈值，不改变模型输出。分数未经过概率校准，百分比显示不等于已经证实的伪造概率。

注意力图由模型路由描述器的空间关注与实际层路由权重组合得到，仅用于辅助观察模型关注区域，不是像素级伪造定位。颜色经过单帧归一化，不宜跨帧比较颜色强弱。层权重表示特征融合比例，不是各层的准确率。

## 7. 日常启动与停止

首次部署成功后，可在 Docker Desktop 的 Containers 页面启动或停止 `deeptrace-local` 整组服务，再打开系统网页。

也可在 `system` 目录使用命令行管理 CPU 配置：

```powershell
# 启动已创建的服务
docker compose -f compose.local-cpu.yaml start

# 查看状态
docker compose -f compose.local-cpu.yaml ps

# 查看主要服务日志
docker compose -f compose.local-cpu.yaml logs --tail 100 api worker-cpu web

# 停止服务并保留数据
docker compose -f compose.local-cpu.yaml stop
```

GPU 配置使用 `compose.local.yaml`，对应推理服务名为 `worker`。停止前应等待任务结束或先取消任务。停止服务会保留数据库和对象存储中的数据，避免删除数据卷。

## 8. 技术实现与验证

前端采用 Next.js、React、TypeScript；后端采用 FastAPI、SQLAlchemy 和 Alembic；Celery 与 Redis 处理异步任务；PostgreSQL 保存业务记录，MinIO 保存媒体和可视化证据，Docker Compose 编排服务。

本次材料整理后，后端自动化测试 **172 项通过**；2 项依赖 PostgreSQL 专用环境的测试未在本次隔离测试中执行。四份 Compose 文件通过配置检查，模型和 8 个演示样例均通过文件一致性校验。上述检查是功能与完整性验证，不代表检测准确率。

代码验证入口：

- 前端：在 `system/apps/web` 中执行 `npm ci`、`npm run lint`、`npm run build`。
- 后端：在具备 FFmpeg、libmagic 等依赖的 Linux/容器环境中，进入 `system/services/api`，执行 `uv sync --frozen --extra inference` 和 `uv run pytest`。

源码修改后，需要重新运行对应的 `docker compose ... up --build -d` 更新运行镜像。
