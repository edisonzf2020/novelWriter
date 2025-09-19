好的，这是 `docs/brownfield-architecture.md` 的完整内容，您可以直接复制保存。
# novelWriter Brownfield 架构文档 (修订版)

## 简介

本文档捕获了 `novelWriter` 代码库的 **当前状态**，其分析核心基于项目自带的 `docs/technical` 目录下的详细技术文档，并结合了对 `flattened-codebase.xml` 中实际代码结构的验证。本文档旨在作为 AI 代理在现有代码库中实施 AI Copilot 增强功能时的核心技术参考，**确保所有新增功能严格遵守已建立的设计标准**。

### 文档范围

本文档的范围严格限定于实施 `novelWriter AI 助手` 所需了解的领域。它首先总结了从现有技术文档中学到的核心架构原则，然后详细说明了新的 AI 功能将如何作为模块化扩展与现有系统安全地集成。

### 变更日志

| 日期 | 版本 | 描述 | 作者 |
| :--- | :--- | :--- | :--- |
| 2025年9月16日 | 2.0 | **重大修订**: 基于 `docs/technical` 目录下的权威技术文档重写了架构分析。 | Winston (Architect) |
| 2025年9月16日 | 1.0 | 基于代码库、PRD 和集成方案的初始 brownfield 分析 | Winston (Architect) |

## 核心架构原则 (源自 /docs/technical)

经过对作者技术文档的详细研读，我们确认了以下必须遵守的核心设计原则：

* **分层架构**: 系统严格遵循 `core` (非 UI 逻辑), `gui` (UI 组件), 和 `formats` (文件格式处理) 之间的分离。**新的 `ai` 包必须作为另一个核心层，不得直接依赖于 `gui`**。
* **信号/槽机制**: PyQt6 的信号和槽是组件间通信的首选机制。新的 AI Copilot UI 必须通过这种方式与主应用交互，以保持松耦合。
* **配置解耦**: `novelwriter.core.config.Config` 是配置的唯一来源。新的 AI 配置 (`AIConfig`) 必须以最小侵入的方式与之集成，如规划文档中所述，通过懒加载属性进行对接。
* **可选依赖**: 系统设计支持通过 `extras_require` (在 `setup.cfg` 中定义) 实现可选功能。AI 助手的功能必须作为一个或多个可选依赖组提供，确保在未安装相关库时，主程序能正常运行。

## 快速参考 - 关键文件和入口点

### 理解系统的关键文件

* **主程序入口**: `novelwriter.py`
* **主窗口 UI**: `novelwriter/gui/main_window.py` (这是 AI Copilot 面板的主要集成点)
* **核心配置**: `novelwriter/core/config.py` (将进行最小化修改以集成 `AIConfig`)
* **技术文档**: `docs/technical/*.md` (**架构和编码标准的最终来源**)

### 增强功能影响区域

* **新增 `ai` 包**: `novelwriter/ai/`
* **新增 `ai_copilot` 扩展**: `novelwriter/extensions/ai_copilot/`
* **主窗口 (`main_window.py`)**: 将被修改以调用 `MainWindowIntegration.integrate_ai_dock`。
* **设置窗口 (`config_dialog.py`)**: 将新增一个 "AI" 选项卡。
* **依赖管理 (`setup.cfg`)**: 将新增 `[options.extras_require]`。

## 高层架构

### 技术摘要

`novelWriter` 的现有架构是基于 Python 和 PyQt6 的经典 MVC 模式桌面应用，并拥有完善的技术文档。本次增强将严格遵循其既定模式，通过引入独立的 `ai` 核心包和 `extensions/ai_copilot` UI 包来集成 AI 助手。核心设计 `NWAiApi` 将作为 AI 功能与 `novelwriter.core` 之间的安全桥梁。UI 层面，将新增一个可停靠的右侧面板。

### 当前技术栈 (已验证)

| 类别 | 技术 | 版本/说明 |
| :--- | :--- | :--- |
| 语言 | Python 3 | `setup.cfg` 中指定 `python_requires >= 3.7` |
| UI 框架 | PyQt6 | `setup.cfg` 中指定 `PyQt6>=6.4.0` |
| 架构 | MVC + 分层 | **在 `docs/technical` 中有详细说明** |
| 开发环境 | 项目专用 virtualenv | `source /Users/fanmac/AI/novelWriter/writer/bin/activate` 激活默认运行/测试环境 |

## 源码树与模块组织

### 项目结构 (集成后)

```plaintext
novelwriter/
├── docs/
│   ├── technical/         # 核心技术文档 (Source of Truth)
│   └── ...
├── core/
├── gui/
├── ai/                      # 新增：AI 核心逻辑 (遵循分层原则)
│   ├── api.py               # NWAiApi (安全动作领域 API)
│   └── ...
├── extensions/
│   ├── ai_copilot/          # 新增：AI Copilot UI 与集成
│   │   ├── integration.py   # 主窗口集成逻辑
│   │   └── ...
│   └── ...
└── ...
```

## 技术债与已知问题 (基于代码和文档分析)

  * **`FIXME` / `TODO` 注释**: 代码库中存在多处 `FIXME` 和 `TODO` 注释，表明作者已意识到一些待办事项和需要改进的区域。新的 AI 功能应避免在这些区域引入更多复杂性。
  * **潜在的紧耦合**: `MainWindow` 类非常庞大。`integration.py` 的薄封装设计是**符合现有架构原则的正确选择**，因为它将新 UI 的创建和管理责任隔离，而不是直接在 `MainWindow` 中实现。

## 集成点与外部依赖

### 内部集成点

  * **主窗口 UI**: AI Copilot 将作为一个 `QDockWidget` 添加到 `MainWindow`。**这符合 `docs/technical` 中关于可扩展 UI 的描述**。
  * **配置系统**: `AIConfig` 的集成方案**符合现有配置系统的设计**。
  * **信号/槽机制**: AI 面板将监听 `MainWindow` 和 `core` 模块发出的现有信号，**这是与现有系统交互的首选方式**。

## 增强功能影响分析

### 需要修改的文件

  * `novelwriter/gui/main_window.py`: 需要添加代码来调用 `MainWindowIntegration.integrate_ai_dock()`。
  * `novelwriter/core/config.py`: 需要添加 `@property` 来懒加载 `AIConfig`。
  * `novelwriter/gui/config_dialog.py`: 需要添加逻辑来加载 `ai_copilot/preferences.py` 中定义的 UI。
  * `setup.cfg`: 需要添加 `[options.extras_require]` 来定义 `ai`, `ai-openai` 等可选依赖组。
      * **所有修改都将是最小化的，以尊重现有代码的稳定性。**

### 新增功能必须遵守的设计标准

根据 `docs/technical`，所有为 AI Copilot 新增的代码都必须遵守以下标准：

1.  **严格遵守分层**: `ai` 包不能有任何 `from novelwriter.gui import ...` 的语句。
2.  **使用信号/槽**: `ai_copilot` 与 `gui` 之间的通信必须通过信号/槽，而不是直接方法调用。
3.  **遵循现有编码风格**: 新代码的风格（命名、文档字符串等）必须与现有代码库保持一致。
4.  **可配置与可禁用**: 整个 AI 功能必须可以通过 `AIConfig` 完全禁用，并且在禁用时不应加载任何相关模块或影响性能。

## 部署与运维指引

- **依赖安装**：AI 功能依赖 `httpx` 等可选库，通过 `pip install novelWriter[ai]` 或 `pip install .[ai]` 进行安装；上线前运行 `python -m pytest tests/test_ai/test_ai_suggestions.py` 验证环境满足可选依赖。
- **上线检查**：在预生产环境中执行 `CONFIG.ai.create_provider()` 或 `novelwriter --info`，确认能力检测成功，并保存 `CONFIG.ai.getProviderCapabilitiesSummary()` 输出以备诊断。
- **回滚策略**：若需禁用 Copilot，可在偏好设置中关闭 AI 功能并恢复 `novelWriter.conf` 中的 `AI` 段落快照；必要时回退至 Story 1.1 标签并重新发布。
- **监控与支持**：持续关注日志中的 `provider.request.failed`、`suggestion.apply_failed` 事件；若发生长时间故障，由 Feature Owner 协调回滚并向支持团队同步状态。

## 增强功能影响分析
