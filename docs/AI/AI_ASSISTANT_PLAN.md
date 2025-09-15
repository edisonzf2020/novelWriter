# novelWriter AI 助手集成方案 v1.1

本文档描述在 novelWriter 中新增“AI 辅助创作”能力的总体设计、目录与命名策略、最小侵入式接入点、实施计划与风险控制，并给出上游同步策略。目标是在不破坏现有架构的前提下，实现右侧 Copilot 风格面板、可被 AI 安全调用的领域 API、和可插拔的 Agent 运行时。

本版本（v1.1）基于评审意见进行优化：采用 `extensions/ai_copilot/` 命名；引入独立 `AIConfig` 配置解耦方案；在 API 中加入事务与审计；补充可选依赖分组与 Anthropic Provider；采用“安全集成”薄封装，暂缓完整插件管理器至后续阶段。

## 1. 背景与目标

- 背景：novelWriter 是基于 Python + PyQt6 的长文创作应用，采用 MVC + 分层架构。我们希望在不重构的前提下，引入 AI 辅助能力。
- 目标：
  - 在 UI 右侧提供 Copilot 风格交互区（Dock 面板），支持建议、续写、摘要与批注等。
  - 抽象“AI 安全动作 API”，封装项目树、文档、导出等领域操作，支持 Dry-run/Diff 预览与事务化应用。
  - 引入可插拔 Agent 运行时（本地/云模型可切换），通过工具函数调用领域 API。
  - 与上游开源仓库保持易于同步，最小化冲突面与侵入点。

不在本期范围：
- 在线协同编辑、云同步、插件生态平台化（留作后续路线）。

## 2. 总体架构与分层

- 分层与依赖方向（单向）：
  - `ai -> core/text/formats`（AI 仅依赖领域，不依赖 GUI）
  - `extensions -> ai`（UI 薄层，调用 AI）
  - 禁止 `core/gui -> ai` 反向依赖，避免循环引用。
- 关键组件：
  - 右侧 Copilot 面板：`extensions/ai_copilot/`（Dock + 视图 + 动作）
  - 领域 API（AI 安全动作层）：`ai/api.py` + `ai/models.py` + `ai/errors.py`
  - Agent 运行时：`ai/orchestrator.py`、`ai/tools/`、`ai/providers/`

### 2.1 目录与命名

建议在 Python 包根目录新增 `ai/`，与 `core/`、`gui/`、`formats/`、`extensions/` 并列（以实际包名为准，例如 `novelwriter/ai/`）：

- novelwriter/ai/
  - config.py（AIConfig：AI 配置管理与主配置对接）
  - api.py（NWAiApi：对外动作 API，强类型、可控副作用、支持事务）
  - models.py（DTO：DocumentRef、TextRange、Suggestion、BuildResult 等）
  - errors.py（AI 层异常类型）
  - tools/（把 api.py 的动作包装为“工具函数”，供模型调用）
  - providers/（OpenAI/Ollama/Anthropic/本地 LLM 的统一接口与实现）
  - orchestrator.py（对话/上下文拼装，工具调用循环，安全栅栏）
- novelwriter/extensions/ai_copilot/
  - plugin.py（可选：插件主类，后续阶段）
  - integration.py（安全集成薄封装，供主窗体调用）
  - dock.py（右侧 QDockWidget）
  - chat_widget.py（聊天界面/消息区）
  - context_selector.py（上下文选择器）
  - preferences.py（配置页面）
- tests/
  - test_ai/（API/Config/Providers/Tools 单测）
  - test_extensions/test_ai_copilot/（Dock/Chat UI 冒烟）

说明：仓库中未发现 `novelwriter/extensions/assistant/providers/` 目录；为避免潜在命名混淆，同时表达功能意图，UI 路径统一使用 `ai_copilot`。

## 3. 最小侵入式接入点

- 主窗口安全集成（推荐）：在主窗体中调用 `MainWindowIntegration.integrate_ai_dock(main_window)`，由 `extensions/ai_copilot/integration.py` 完成 Dock 的创建与注册；失败时记录日志并静默跳过。
- 设置页增加 AI 选项卡：复用 `extensions/configlayout.py` 体系，读取 `CONFIG.ai`。
- 启动时延迟初始化 Provider：懒加载，缺依赖时显示友好提示。
- 复用现有信号/槽（文档切换、光标/选区变更、项目变更）；缺失再补充必要信号。
- 可选方案（后续）：在 `core/plugins.py` 引入通用插件管理器，当前阶段不强制，以降低核心入侵面。
- 决议：已确认 P0–P2 不引入通用插件管理器，采用集成薄封装；插件化将在 P3 再推进。

集成示例（integration）：
```python
# novelwriter/extensions/ai_copilot/integration.py
from PyQt6.QtCore import Qt
import logging

logger = logging.getLogger(__name__)

class MainWindowIntegration:
    @staticmethod
    def integrate_ai_dock(main_window) -> bool:
        try:
            if not getattr(main_window, '_ai_dock_integrated', False):
                from .dock import AICopilotDock
                ai_dock = AICopilotDock(main_window)
                main_window.addDockWidget(Qt.DockWidgetArea.RightDockWidgetArea, ai_dock)
                main_window._ai_dock_integrated = True
            return True
        except Exception as e:
            logger.error(f"Failed to integrate AI dock: {e}")
            return False
```

## 4. UI：右侧 Copilot 面板

- 面板形态：可折叠/可分离的 `QDockWidget`，停靠编辑器右侧；遵循主题与 i18n。
- 区域组成：
  - 消息区：流式显示模型输出、工具执行日志、错误提示
  - 输入区：多行输入、系统提示选择、上下文范围开关（选区/当前文档/大纲/全项目）
  - 动作区：快捷命令（续写、摘要、标题建议、风格转换、批注）
  - 控制区：中断/重试、Dry-run 开关、应用前 Diff 预览按钮
- 性能与线程：
  - 模型/网络调用放入 `QThread` 或 `QtConcurrent`；提供取消机制与超时
  - 结果用信号投递到主线程，避免 GUI 阻塞

## 5. 领域 API（AI 安全动作层）

位置：`novelwriter/ai/api.py` + `models.py` + `errors.py`

设计原则：
- 强类型、完整 docstring、异常可分层捕获
- 默认无副作用：读操作与 Dry-run 为主；写操作需要显式 `apply=True`
- 重要变更前支持 Diff 预览；关键操作（删除/批量改写）需要用户确认
- 支持事务与操作审计：可回滚、可追溯

### 5.1 数据模型（节选，models.py）

```python
from dataclasses import dataclass
from typing import Optional

@dataclass
class DocumentRef:
    handle: str
    name: str
    parent: Optional[str]

@dataclass
class TextRange:
    start: int  # 文档内偏移
    end: int

@dataclass
class Suggestion:
    id: str
    handle: str
    preview: str           # 建议文本
    diff: Optional[str]    # 可选：统一 diff 文本

@dataclass
class BuildResult:
    format: str
    outputPath: str
    success: bool
    message: Optional[str] = None
```

### 5.2 接口清单（节选，api.py）

```python
class NWAiApi:
    """AI 可调用的安全动作 API 层。仅依赖 core/text/formats，不依赖 GUI。"""

    # 事务与审计
    def begin_transaction(self) -> str: ...
    def commit_transaction(self, transaction_id: str) -> bool: ...
    def rollback_transaction(self, transaction_id: str) -> bool: ...
    def get_audit_log(self) -> list[dict]: ...

    # 项目/文档
    def getProjectMeta(self) -> dict: ...
    def listDocuments(self, scope: str = "all") -> list[DocumentRef]: ...
    def getCurrentDocument(self) -> Optional[DocumentRef]: ...

    # 文本读取与写入（写入需 apply 明确开启）
    def getDocText(self, handle: str) -> str: ...
    def setDocText(self, handle: str, text: str, apply: bool = False) -> bool: ...

    # 建议与应用
    def previewSuggestion(self, handle: str, rng: TextRange, newText: str) -> Suggestion: ...
    def applySuggestion(self, suggestionId: str) -> bool: ...

    # 上下文与搜索
    def collectContext(self, mode: str) -> str: ...
    def search(self, query: str, scope: str = "document", limit: int = 50) -> list[str]: ...

    # 结构化操作（P1+）
    def createChapter(self, name: str, parent: Optional[str]) -> Optional[DocumentRef]: ...
    def createScene(self, name: str, parent: str, after: Optional[str] = None) -> Optional[DocumentRef]: ...
    def duplicateItem(self, handle: str, parent: Optional[str] = None, after: bool = True) -> Optional[DocumentRef]: ...

    # 导出
    def build(self, fmt: str, options: Optional[dict] = None) -> BuildResult: ...
```

### 5.3 错误分型（errors.py）

```python
class NWAiError(Exception):
    pass

class NWAiProviderError(NWAiError):
    pass

class NWAiApiError(NWAiError):
    pass

class NWAiConfigError(NWAiError):
    pass
```

## 6. Agent 运行时

- Providers（`ai/providers/`）
  - 统一接口：`generate(prompt, tools=None, stream=False, timeout=...)`
  - 实现：`OpenAICompatibleProvider`（P0 采用）、`OpenAIProvider`、`OllamaProvider`、`AnthropicProvider`、`LocalProvider`
  - 配置：`CONFIG.ai.provider`, `CONFIG.ai.model`, `CONFIG.ai.temperature` 等
- Tools（`ai/tools/`）
  - 将 `NWAiApi` 方法以“工具”形式暴露（带参数校验/JSON schema）
  - 工具例：`insert_at_cursor`, `replace_range`, `create_scene`, `build_project`
- Orchestrator（`ai/orchestrator.py`）
  - 负责对话管理、上下文拼装、调用 Provider、路由工具调用
  - 安全栅栏：
    - 白名单工具与参数范围
    - 最大变更阈值（字数/比例）
    - 敏感操作需确认（UI 二次确认）
    - 所有写操作支持回滚点（事务）

### 6.1 P0 Provider 选择：OpenAI API 兼容栈

- 决议：P0 优先实现“OpenAI API 标准格式”的兼容 Provider（`OpenAICompatibleProvider`）。在不改动上层调用的前提下，同时支持：
  - 官方 OpenAI（api.openai.com）
  - 兼容 OpenAI API 的服务（例如自建/第三方/本地代理，提供 `chat/completions` 与 `responses` 端点，遵循工具调用与流式协议）
- 兼容策略：
  - Base URL 可配置（`CONFIG.ai.openai_base_url`）；默认为官方。
  - 优先使用通用 HTTP 客户端（`httpx`）按 OpenAI 协议访问；如检测到官方 SDK 可用且开启，则可走 SDK（支持 `base_url` 覆盖）。
  - 支持 `chat.completions` 与 `responses` 两种模式，首选 `responses`（新接口），回退至 `chat.completions`。
  - 支持流式输出（server‑sent events），以及工具调用（function/tool calls）参数格式。
  - Token 限额由配置项控制（`max_tokens`），超限自动截断/提示。

接口要点（示例）：
```python
provider = OpenAICompatibleProvider(
  api_key=...,                 # 必填或从环境变量读取
  base_url=...,                # 可选，默认官方；用于兼容端点
  model=...,                   # 模型名，保持透传
  timeout=30.0,                # 请求超时
  use_sdk=False                # 可选：是否使用 openai 官方 SDK
)
resp = provider.generate(
  messages=[{"role":"user","content":"..."}],
  tools=[...],                 # 可选：工具定义
  stream=True                  # 流式
)
```

注意：为覆盖“兼容栈”，采用 `httpx` 实现的 HTTP 客户端作为默认通道；当 `openai` SDK 可用且未设置 `base_url`（或明确允许），可切换为 SDK 路径。

### 6.2 端点能力检测（P1）

- 目标：在首次调用前检测兼容端点支持的能力并缓存结果，自动选择最佳协议路径与特性。
- 能力项：
  - 是否支持 `POST /v1/responses`（含工具调用与流式）
  - 是否仅支持 `POST /v1/chat/completions`
  - 是否支持流式（SSE）、最大 tokens、工具调用参数格式
- 策略：
  - 懒检测：首次请求或显式刷新时检测，结果缓存于 Provider 会话级。
  - 回退：检测失败或不支持时自动回退到较低能力端点。
  - 可观测：将能力检测结果写入调试日志，便于问题排查。

## 7. 配置与可选依赖

### 7.1 AI 配置解耦（AIConfig）

- 位置：`novelwriter/ai/config.py`
- 设计：将 AI 配置独立为 `AIConfig`，在主配置加载/保存阶段，通过 `Config.ai` 属性进行懒加载对接，减少对核心 `Config` 的侵入修改。

示例：
```python
# novelwriter/ai/config.py
class AIConfig:
    def __init__(self):
        self.enabled = False
        self.provider = "openai"
        self.model = "gpt-4"
        self.temperature = 0.7
        self.max_tokens = 2000
        self.dry_run_default = True
        self.ask_before_apply = True
        self.max_context_length = 8000

    def load_from_main_config(self, conf):
        sec = "AI"
        self.enabled = conf.rdBool(sec, "enabled", self.enabled)
        self.provider = conf.rdStr(sec, "provider", self.provider)
        self.model = conf.rdStr(sec, "model", self.model)
        self.temperature = conf.rdFloat(sec, "temperature", self.temperature)

    def save_to_main_config(self, conf):
        if "AI" not in conf:
            conf["AI"] = {}
        conf["AI"]["enabled"] = str(self.enabled)
        conf["AI"]["provider"] = str(self.provider)
        conf["AI"]["model"] = str(self.model)
        conf["AI"]["temperature"] = str(self.temperature)
```

在 `novelwriter/config.py` 中的最小改动（示例）：
```python
class Config:
    def __init__(self):
        # ...
        self._ai_config = None  # 延迟加载

    @property
    def ai(self):
        if self._ai_config is None:
            try:
                from novelwriter.ai.config import AIConfig
                self._ai_config = AIConfig()
            except Exception:
                class DisabledAIConfig:
                    enabled = False
                self._ai_config = DisabledAIConfig()
        return self._ai_config

    def loadConfig(self, splash=None):
        # ... 现有加载
        try:
            if getattr(self, '_ai_config', None):
                self._ai_config.load_from_main_config(conf)
        except Exception as e:
            logger.warning(f"Failed to load AI config: {e}")
            # 不影响主程序启动

    def saveConfig(self):
        # ... 现有保存
        try:
            if getattr(self, '_ai_config', None):
                self._ai_config.save_to_main_config(conf)
        except Exception as e:
            logger.warning(f"Failed to save AI config: {e}")
            # 不影响主程序保存
```

### 7.2 可选依赖（pyproject.toml）

建议将 AI 依赖分层为多个 extras，按需安装：

```toml
[project.optional-dependencies]
ai = [
  "httpx>=0.24.0",
  "pydantic>=2.0.0"
]
ai-openai = [
  "openai>=1.0.0",
  "tiktoken>=0.5.0"
]
ai-local = [
  "ollama>=0.1.0"
]
ai-anthropic = [
  "anthropic>=0.8.0"
]
```

说明：将 `tiktoken` 放在 `ai-openai` 而非通用 `ai`，以避免对非 OpenAI 场景的额外依赖。

### 7.3 OpenAI 兼容配置项（P0）

- 基础配置（建议）：
  - `CONFIG.ai.provider = "openai"`
  - `CONFIG.ai.model = "gpt-4o-mini"`（示例，保持可配置）
  - `CONFIG.ai.openai_base_url = "https://api.openai.com/v1"`（可指向兼容服务）
  - `CONFIG.ai.timeout = 30.0`
  - `CONFIG.ai.temperature = 0.7`
  - `CONFIG.ai.max_tokens = 2000`
  - `CONFIG.ai.enable_stream = True`
- 凭据读取：
  - 优先从环境变量读取：`OPENAI_API_KEY`；若配置中显式提供 `CONFIG.ai.api_key` 则覆盖。
  - 兼容 Azure/OpenAI‑兼容服务可在 `openai_base_url` 中体现；若使用 Azure 还需 `api_version` 与 `deployment` 等参数（后续 P1/P2 支持）。
- 端点策略：
  - 首选 `POST /v1/responses`（若兼容端点支持），否则回退 `POST /v1/chat/completions`。
  - 流式使用 `stream=True`，采用 SSE 逐步刷新到 UI。

## 8. 与上游同步策略（GitHub）

- 模式：Fork + Upstream
  - 添加 upstream 远程；在 fork 中开发 AI 目录与少量接入点
  - 周期性合并 upstream（按钮或命令行）
- 自动化（可选）：GitHub Actions 定时同步
  - 方案 A：纯 Git 步骤
```yaml
name: Sync Upstream
on:
  schedule:
    - cron: "0 3 * * 1,4"
  workflow_dispatch: {}
jobs:
  sync:
    runs-on: ubuntu-latest
    permissions:
      contents: write
    steps:
      - uses: actions/checkout@v4
        with:
          fetch-depth: 0
      - name: Add upstream
        run: |
          git remote add upstream https://github.com/<upstream_owner>/<upstream_repo>.git || true
          git fetch upstream
      - name: Merge upstream/main
        run: |
          git checkout main
          git merge --ff-only upstream/main || git merge --no-edit upstream/main
      - name: Push
        run: git push origin main
```
  - 方案 B：使用现成 Action（repo-sync 类），效果等同

- 冲突控制：
  - 新增目录几乎无冲突
  - 少量接入点在升级后手工审阅即可

## 9. 实施计划与里程碑（修订）

- P0 基础（2–3 周）
  - 目录结构创建：`novelwriter/ai/`、`extensions/ai_copilot/`
  - 配置系统解耦：`AIConfig` 与主配置对接（最小入侵）
  - 安全集成：`integration.py` 与 Dock 雏形，i18n/主题接入
  - Provider：实现 `OpenAICompatibleProvider` 最小可用版（支持 base_url、api_key、model、responses/completions、流式、超时）
  - API：只读 + 建议预览（Diff），手动应用（无事务写入）

- P1 核心（3–4 周）
  - API：写操作 + 事务/回滚 + 审计
  - 事务机制考量：嵌套事务策略、事务超时（自动回滚）、跨文档一致性边界
  - 上下文收集与会话记忆
  - Provider：OpenAI 与本地（Ollama）实现
  - 端点能力检测：实现 6.2 中的懒检测与缓存，自动选择 responses/completions
  - UI：聊天界面、上下文选择器、配置页面
  - 安全栅栏与变更阈值

- P2 完善（2–3 周）
  - Provider：Anthropic 接入与切换
  - 性能优化：异步、缓存、取消、超时；日志与指标
  - 构建/导出前校对；更友好的 Diff/回滚体验
  - 测试完善：API/Tools 单测、UI 冒烟、集成测试

- P3 插件化（可选）
  - 评估与实现通用 `core/plugins.py` 插件管理器
  - 将 AI 集成切换为插件注册，减少主窗体逻辑

## 10. 任务拆解（Backlog）

- 文档与设计
  - [ ] 最终确认目录与命名（novelwriter/ai/, extensions/ai_copilot/）
  - [ ] NWAiApi 接口签名与类型文档（含事务/审计）
  - [ ] Dock 面板交互稿与状态图
  - [ ] Provider 接口规范与最小实现选择
- 基础实现（P0）
  - [ ] 创建目录与空文件骨架
  - [ ] AIConfig 与主配置对接（延迟加载 + 守卫）
  - [ ] Integration 薄封装与 Dock 雏形
  - [ ] 只读 API 与建议预览（Diff）
  - [ ] 设置页挂载 AI 选项卡（可选依赖守卫）
- 强化与扩展（P1+）
  - [ ] 事务/回滚/审计落地
  - [ ] 上下文选择器与会话记忆
  - [ ] 安全栅栏（阈值/白名单/确认）
  - [ ] 构建/导出集成与校对建议
- 同步与质量
  - [ ] 配置 upstream 与手动合并流程
  - [ ] 添加定时同步 Actions（可选）
  - [ ] 单测与 CI 基础流水线
  - [ ] 可选依赖安装与运行指引文档

## 11. 风险与缓解

- UI 阻塞：全部模型/网络调用放后台线程；提供取消与超时；结果信号返回主线程
- 大上下文成本：检索式拼接与分块；摘要缓存；限制上下文预算
- 误改写：默认 Dry-run + Diff；敏感操作二次确认；支持事务回滚与审计
- 上游漂移：侵入点极少化；扩展点/Hook 化；定期合并并跑冒烟测试
- 隐私与合规：默认本地优先；云模式需用户提供 key 与明确提示

## 12. 验收标准（P0）

- 应用在未安装 AI 依赖时正常运行，AI 面板不加载或提示友好
- 右侧 Dock 正常显示，流式输出不卡 UI，国际化与主题一致
- 至少 3 个快捷动作可用；建议前可预览 Diff；应用后文档内容正确
- 配置系统改造后，主配置加载/保存无回归
- Provider 验收：
  - 使用官方 OpenAI（默认 base_url）可正常生成与流式输出
  - 将 `openai_base_url` 指向任一兼容 OpenAI API 的服务（本地/代理）仍可正常工作
  - 支持 `responses` 或 `chat.completions` 至少一种协议路径；token 限制与超时生效

## 14. 成功指标与性能要求

- UI 响应：常规 UI 操作响应时间 < 100ms；长任务提供进度与可取消。
- AI 调用：支持取消与超时；超时默认 30s（可配置）。
- 内存：避免泄漏，及时释放大对象（文本/图像/网络缓冲）。
- 并发：支持多个 AI 请求并发处理（队列/节流/取消机制）。
- 稳定性与满意度（发布后跟踪）：
  - 用户满意度 > 85%
  - 功能稳定性 > 99%
  - 性能影响 < 5%
  - 上游同步冲突 < 1 次/月

## 13. 变更记录

- v1.1：采纳评审意见（目录 `ai_copilot`、AIConfig 解耦、事务/审计、依赖分组与 Anthropic）、补充安全集成方式与计划修订
- v1.0：初稿，明确分层、目录、API 草案、实施与同步策略
