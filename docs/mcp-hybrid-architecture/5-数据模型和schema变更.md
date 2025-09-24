# 5. 数据模型和Schema变更

## 5.1 新增数据模型

### 模型1：HybridToolCall
**目的**：统一本地和外部工具调用的数据表示

```python
class HybridToolCall(BaseModel):
    tool_name: str
    tool_type: Literal["local", "external"]
    parameters: Dict[str, Any]
    call_id: str
    timestamp: datetime = datetime.now()
```

### 模型2：ToolExecutionResult
**目的**：标准化工具执行结果格式

```python
class ToolExecutionResult(BaseModel):
    call_id: str
    success: bool
    result: Optional[Dict[str, Any]] = None
    error_message: Optional[str] = None
    execution_time_ms: int
```

### 模型3：ExternalMCPConnection
**目的**：管理外部MCP工具连接信息

```python
class ExternalMCPConnection(BaseModel):
    connection_id: str
    server_url: str
    transport_type: Literal["streamable-http", "sse"] = "streamable-http"
    auth_config: Optional[Dict[str, str]] = None
    available_tools: List[str] = []
    health_status: Literal["healthy", "degraded", "offline"] = "offline"
```

## 5.2 Schema集成策略

**无数据库Schema变更**：
- 使用JSON配置文件而非数据库
- 运行时数据存储，无持久化依赖
- 现有模型复用，最大化复用`ai/models.py`中的数据类型

---
