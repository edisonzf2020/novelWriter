# **第二部分：P1 - 核心功能与写入 API**

**阶段目标:** 将 AI 助手从“建议者”升级为“协作者”。核心是为 `NWAiApi` 赋予安全的写入能力，并引入事务和上下文管理机制。

## **史诗 2: AI 安全写入与事务化操作**

> **作为一名作者，** 我希望 AI 助手能够安全地执行写入操作，并且所有关键变更都是可回滚的，
> **以便** 我可以放心地授权 AI 对我的稿件进行实质性修改。

**用户故事 2.0: 为 `NWAiApi` 集成事务与审计功能**
> **作为 AI Agent，** 我需要在执行一系列写入操作前开启一个事务，并在操作完成后提交或回滚，
> **以便** 保证复杂操作的原子性和数据一致性。
> **验收标准:**
> * 实现 `begin_transaction()`, `commit_transaction()`, `rollback_transaction()` 接口。
> * 实现 `get_audit_log()` 接口，用于追踪 AI 发起的关键操作。

**用户故事 2.1: 实现 API 的安全写入与建议应用接口**
> **作为 AI Agent，** 我需要能够通过 `NWAiApi` 安全地修改文档内容或应用一个建议，
> **以便** 我可以在用户的授权下，对文学作品进行实质性的修改。
> **验收标准:**
> * 实现 `setDocText(..., apply=True)` 和 `applySuggestion()` 等写入方法。
> * 所有写入方法都必须在事务块内执行。
> * 实现安全栅栏，如对大量内容变更进行二次确认。

## **史诗 3: 增强的上下文理解与会话记忆**

> **作为一名创作者，** 我希望 AI 助手能够理解更广泛的上下文并记住我们之前的对话，
> **以便** AI 能够提供更贴切、更连贯的建议。

**用户故事 3.0: 灵活的上下文选择**
> **作为一名长篇小说作者，** 我希望在与 AI 交互时，可以手动指定当前的上下文范围，
> **以便** AI 的回答和建议能更精准地聚焦于我正在处理的部分。
> **验收标准:**
> * Copilot 面板中增加一个上下文范围选择器（选项：“选区”、“当前文档”、“大纲”、“整个项目”）。
> * `NWAiApi` 的 `collectContext` 方法能根据所选范围正确收集上下文。
> * AI Copilot 具备基础的会话记忆能力。

**用户故事 3.1: 智能的 API 端点能力检测**
> **作为一名开发者，** 我希望 AI Provider 能够自动检测所连接的 API 端点支持哪些高级功能，
> **以便** 系统能够自动选择最优的协议路径，最大化功能可用性。
> **验收标准:**
> * `OpenAICompatibleProvider` 在首次连接时进行懒检测。
> * 能自动判断并优先使用 `POST /v1/responses` 接口，否则回退到 `POST /v1/chat/completions`。
> * 检测结果在会话级别缓存，并记入调试日志。

**用户故事 3.2: 引入 OpenAI 官方 Python SDK Provider**
> **作为 AI Copilot 维护者，** 我希望在现有 Provider 体系中集成官方 OpenAI Python SDK 并提供可配置切换，
> **以便** 减少协议变更导致的请求错误并提升对最新能力的兼容性。
> **验收标准:**
> * 在 `AIConfig` 和 Provider 工厂中新增 "openai-sdk" 选项，启用后 `NWAiApi` 通过官方 `openai.OpenAI` 客户端执行 `/v1/responses` 与 `/v1/chat/completions` 请求，并保留原有 `openai` 兼容实现作为回退策略。
> * 官方 SDK Provider 能正确处理 `input` 载荷（包含字符串与复合多段输入），避免日志中 `Invalid type for 'input': expected string, but got array.` 的错误；当 `/v1/responses` 不可用时自动降级到 `/v1/chat/completions` 并记录调试日志。
> * 能力探测与模型列表通过官方 SDK (`client.models.list`/`retrieve`) 实现，返回结果映射至 `ProviderCapabilities` 结构，并写入审核日志以供 UI 查询。
> * `AIConfig` 持久化模型/参数时兼容新 Provider，`novelwriter/dialogs/preferences.py` 中的 Provider 下拉和连接校验按钮支持选择、保存和回显 "OpenAI Official SDK"，未安装 SDK 时显示可用性提醒。
> * `pyproject.toml` 的 `ai` 可选依赖扩展包含兼容版本的 `openai` 包，缺失依赖时 Copilot 展示友好提示；新增单元与 GUI 测试覆盖 Provider 构建、流式输出、错误映射和 UI 切换场景，并通过 CI。

---
