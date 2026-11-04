
    # 拾级AI 平台说明 v1.8

## 一、产品定位
面向开发者的 AI 应用开发平台，帮助开发者快速构建 LLM 应用。

## 二、目标客户
- 开发者、创业团队、企业技术部门
- 典型场景：RAG 应用、Agent 开发、AI 助手

## 三、核心功能

### 3.1 模型接入
- 支持 DeepSeek、通义千问、OpenAI
- 统一 API 接口
- 支持流式输出

### 3.2 RAG 引擎
- 文档加载：PDF、Word、Markdown、TXT
- 文本切分：多种策略可选
- 向量化：支持多种 Embedding 模型
- 向量库：内置 Chroma、支持 Milvus
- 检索：向量检索、混合检索、重排序

### 3.3 Agent 框架
- 工具调用：Function Calling
- 多轮对话：记忆管理
- 工作流：LangGraph 支持
- 多 Agent 协作

### 3.4 部署方式
- SaaS：开箱即用
- 私有化部署：支持本地服务器
- 混合部署：核心数据本地，计算上云

## 四、技术架构
- 前端：Streamlit / React
- 后端：Python + FastAPI
- 框架：LangChain + LangGraph
- 向量库：Chroma / Milvus

## 五、版本与定价

| 版本 | 价格 | 功能 |
|:---|:---|:---|
| 免费版 | 0 元 | 1000 次调用/月 |
| 开发者版 | 199 元/月 | 10000 次调用/月 |
| 企业版 | 999 元/月 | 不限调用 + 私有化 |

## 六、常见问题

### Q：支持哪些 Embedding 模型？
A：text-embedding-v4、text-embedding-3-small、bge-m3。

### Q：RAG 效果如何评估？
A：提供评估工具，支持 Recall、MRR、幻觉率等指标。

### Q：支持自定义 Prompt 吗？
A：支持，可在控制台配置 Prompt 模板。
    