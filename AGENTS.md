# First Rule
- Always respond in Chinese.
- Whenever you are confused about the specific use of the technology, always be sure to call the MCP tool `context7` to determine the latest technical details.
- Whenever you need to get the current time, be sure to call the MCP tool `time-mcp`.

# novelWriter 项目 AI 代理开发指南

## 项目概述

novelWriter 是一个专业的小说创作软件，采用 Python + PyQt6 构建。这是一个功能完整的桌面应用程序，专为创作长篇小说、剧本和其他创意写作项目设计。

### 核心特性
- **项目管理**：完整的写作项目管理系统
- **文档编辑**：专业的文档编辑器，支持 Markdown 语法
- **大纲管理**：章节、场景的层级结构管理
- **多格式导出**：支持 HTML、ODT、DOCX、PDF、Markdown 等格式
- **写作统计**：详细的写作进度和统计分析
- **主题系统**：完整的明暗主题支持
- **国际化**：多语言支持

## 项目架构

### 核心架构模式
novelWriter 采用了经典的 **MVC + 分层架构** 模式：

```
novelWriter/
├── core/           # 核心业务逻辑层
├── gui/            # 图形用户界面层
├── formats/        # 文档格式处理层
├── extensions/     # GUI扩展组件层
├── dialogs/        # 对话框层
├── tools/          # 工具功能层
└── text/           # 文本处理层
```

### 关键设计模式

#### 1. 单例模式 (Singleton)
```python
# CONFIG 和 SHARED 作为全局单例
CONFIG = Config()
SHARED = SharedData()
```

#### 2. 工厂模式 (Factory)
```python
# 项目构建器
class ProjectBuilder:
    def buildProject(self, data: dict) -> bool:
        # 根据数据创建不同类型的项目
```

#### 3. 观察者模式 (Observer)
```python
# 广泛使用 PyQt 信号槽机制
class NWItem:
    itemChanged = pyqtSignal(str, str)
```

#### 4. 策略模式 (Strategy)
```python
# 不同格式的构建策略
if bFormat == nwBuildFmt.HTML:
    makeObj = ToHtml(project)
elif bFormat == nwBuildFmt.ODT:
    makeObj = ToOdt(project)
```

## 核心模块详解

### 1. 数据管理层 (`core/`)

#### 关键文件及职责

**`project.py`** - 项目核心管理器
```python
class NWProject:
    """项目的总控制器，管理所有项目相关操作"""
    def openProject(self, path: Path) -> bool
    def saveProject(self) -> bool
    def newFile(self, name: str, parent: str) -> str | None
```

**`storage.py`** - 存储系统抽象
```python
class NWStorage:
    """负责文件系统操作的抽象层"""
    def getDocument(self, handle: str) -> NWDocument
    def getDocumentText(self, handle: str) -> str
```

**`item.py`** - 项目元素模型
```python
class NWItem:
    """项目树中的单个元素（文件夹或文档）"""
    def isDocumentLayout(self) -> bool
    def isNoteLayout(self) -> bool
```

**`tree.py`** - 项目树结构
```python
class NWTree:
    """管理项目的树状结构"""
    def __iter__(self) -> Iterator[NWItem]
    def duplicate(self, handle: str, parent: str, after: bool) -> NWItem | None
```

### 2. 用户界面层 (`gui/`)

#### 主要组件

**`doceditor.py`** - 文档编辑器
- 基于 QTextEdit 的富文本编辑器
- 支持语法高亮和自动完成
- 集成拼写检查

**`projtree.py`** - 项目树视图
- 文件/文件夹的可视化管理
- 拖拽操作支持
- 上下文菜单

**`outline.py`** - 大纲视图
- 章节结构可视化
- 快速导航功能

### 3. 格式处理层 (`formats/`)

#### 格式转换系统
```python
class Tokenizer:
    """文本标记化的基类"""
    def tokenizeText(self) -> None
    def doConvert(self) -> None

class ToHtml(Tokenizer):
    """HTML 格式输出"""
    
class ToOdt(Tokenizer):
    """ODT 格式输出"""
```

### 4. 扩展组件层 (`extensions/`)

#### 自定义 UI 组件
- **`switch.py`** - 动画切换开关
- **`progressbars.py`** - 自定义进度条
- **`configlayout.py`** - 配置表单系统
- **`modified.py`** - 增强标准组件

## 开发指导

### 编码规范

#### 1. 代码风格
```python
# 类命名：PascalCase
class NWProject:

# 方法命名：camelCase
def openProject(self) -> bool:

# 常量命名：UPPER_CASE
MAX_SEARCH_RESULT = 200

# 私有方法：前缀 _
def _setupBuild(self) -> None:
```

#### 2. 类型注解
```python
# 强制使用类型注解
def processDocument(self, handle: str) -> bool:
    return True

# 复杂类型使用 typing
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from novelwriter.core.project import NWProject
```

#### 3. 文档字符串
```python
def buildProject(self, data: dict) -> bool:
    """从数据字典构建新项目
    
    Args:
        data: 包含项目设置的字典
        
    Returns:
        构建成功返回 True，失败返回 False
    """
```

### 常见开发任务

#### 1. 添加新的导出格式

1. 在 `formats/` 目录创建新的格式类：
```python
class ToNewFormat(Tokenizer):
    def __init__(self, project: NWProject) -> None:
        super().__init__(project)
    
    def doConvert(self) -> None:
        # 实现格式转换逻辑
```

2. 在 `enum.py` 中添加新格式枚举：
```python
class nwBuildFmt(Enum):
    NEW_FORMAT = "new_format"
```

3. 在 `docbuild.py` 中添加构建逻辑：
```python
elif bFormat == nwBuildFmt.NEW_FORMAT:
    makeObj = ToNewFormat(self._project)
```

#### 2. 创建新的 GUI 组件

1. 在 `extensions/` 目录创建组件文件
2. 继承适当的 PyQt6 基类
3. 实现自定义绘制（如需要）
4. 添加信号和槽
5. 在主界面中集成

#### 3. 扩展项目数据结构

1. 修改 `projectdata.py` 中的数据模型
2. 更新 `projectxml.py` 中的序列化逻辑
3. 考虑向后兼容性
4. 添加数据验证

### 依赖管理

#### 核心依赖
```python
# GUI框架
PyQt6 >= 6.2.0

# 文档处理
lxml >= 4.6.0  # XML处理
enchant >= 3.0.0  # 拼写检查（可选）

# 开发工具
pytest >= 6.0.0  # 测试框架
flake8 >= 4.0.0  # 代码检查
```

#### 可选依赖
```python
# 导出功能增强
python-docx  # DOCX 支持
```

### 测试策略

#### 测试目录结构
```
tests/
├── test_core/      # 核心功能测试
├── test_gui/       # GUI 测试
├── test_formats/   # 格式转换测试
└── fixtures/       # 测试数据
```

#### 测试示例
```python
def test_project_creation(mockGUI):
    """测试项目创建功能"""
    project = NWProject()
    assert project.createProject("/tmp/test") is True
    assert project.projPath.exists()
```

## 关键系统组件

### 配置系统
```python
# 全局配置访问
CONFIG.textFont  # 文本字体
CONFIG.guiTheme  # 界面主题
CONFIG.spellCheck  # 拼写检查设置
```

### 主题系统
```python
# 主题管理
SHARED.theme.getIcon(iconKey, color)
SHARED.theme.getColor(colorKey)
```

### 国际化系统
```python
# 翻译函数
self.tr("Text to translate")
```

## 性能考虑

### 1. 大文档处理
- 使用懒加载机制
- 避免全文档重新渲染
- 实现增量更新

### 2. 内存管理
```python
# PyQt 对象的正确删除
def softDelete(self) -> None:
    self.setParent(None)  # 避免 C++ 对象提前删除
```

### 3. 文件操作
- 异步文件 I/O（大文件）
- 文件锁定机制
- 备份和恢复策略

## 调试和诊断

### 日志系统
```python
import logging
logger = logging.getLogger(__name__)

logger.debug("调试信息")
logger.info("一般信息")
logger.warning("警告信息")
logger.error("错误信息")
```

### 常见问题排查

1. **GUI 无响应**
   - 检查主线程是否被阻塞
   - 使用 QApplication.processEvents()

2. **内存泄漏**
   - 检查 Qt 对象的父子关系
   - 使用 softDelete() 方法

3. **文件访问错误**
   - 检查文件权限
   - 验证路径有效性

## AI 代理特殊指导

### 代码生成建议

1. **遵循现有模式**：新代码应当遵循项目中已建立的设计模式
2. **类型安全**：始终使用类型注解
3. **错误处理**：实现完整的异常处理
4. **测试覆盖**：为新功能编写测试
5. **文档更新**：更新相关文档和注释

### 架构决策

1. **保持分层**：不要跨层直接调用
2. **信号槽优先**：使用 Qt 信号槽而不是直接方法调用
3. **配置驱动**：通过配置而非硬编码实现可变行为
4. **国际化考虑**：所有用户可见文本都要支持翻译

### 常见陷阱

1. **Qt 对象生命周期**：注意 Python 和 C++ 对象的生命周期差异
2. **线程安全**：GUI 操作必须在主线程进行
3. **循环引用**：避免创建循环引用，特别是在信号槽连接中
4. **资源管理**：及时释放文件句柄、图像等资源

## 项目路线图

### 当前版本重点
- 性能优化
- 用户体验改进
- 国际化完善

### 未来发展方向
- 云同步功能
- 协作编辑
- 插件系统
- 移动端支持

---

**注意**：在为此项目贡献代码时，请始终参考现有代码的风格和架构模式。如有疑问，请查阅项目的测试用例和现有实现作为参考。

本指南基于对 novelWriter 项目源码的深度分析编写，旨在帮助 AI 代理更好地理解和协助项目开发。
