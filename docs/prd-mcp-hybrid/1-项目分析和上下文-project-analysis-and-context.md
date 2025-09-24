# 1. 项目分析和上下文 (Project Analysis and Context)

## 1.1 现有项目概述 (Existing Project Overview)

### 分析来源 (Analysis Source)
- 架构文档分析：基于完整的MCP混合架构设计文档 (727行)
- 代码深度分析：ai/api.py (1912行) 职责分离需求确认
- IDE-based新鲜分析：实时项目状态评估

### 当前项目状态 (Current Project State)
novelWriter是一个**成熟的AI增强创作应用**，已完成Story 5.0开发，具备完整的AI Copilot集成。当前需要通过**架构演进型重构**实现MCP工具化，将现有功能抽象为标准化工具接口，支持AI agent深度参与创作工作流。

## 1.2 可用文档分析 (Available Documentation Analysis)

### 可用文档清单
- [x] 技术栈文档：Python 3.10+, PyQt6 6.4+, MCP Python SDK
- [x] 源码树/架构：完整的混合MCP架构设计
- [x] 编码标准：PEP 8, 类型提示, camelCase方法名
- [x] API文档：现有NWAiApi和新设计的统一API
- [x] 外部API文档：MCP协议标准和工具生态
- [x] 技术债务文档：ai/api.py职责混合问题识别
- [ ] UX/UI指南：需要基于现有AI Copilot扩展

## 1.3 增强范围定义 (Enhancement Scope Definition)

### 增强类型
- [x] **Major Feature Modification** - MCP工具化是核心功能改进
- [x] **Integration with New Systems** - MCP协议和外部工具生态集成
- [x] **Technology Stack Enhancement** - 职责分离重构和统一API

### 增强描述
通过创建统一API中间层和AI模块职责分离重构，将novelWriter核心功能抽象为高性能本地工具，同时集成外部MCP工具生态，实现完整的混合MCP架构。

### 影响评估
- [x] **Moderate to Significant Impact** - 需要架构层重构但保持功能完整性

## 1.4 目标和背景上下文 (Goals and Background Context)

### 目标
• 实现统一API中间层，提供一致的数据访问接口和权限控制
• 完成AI模块职责分离，ai/api.py → ai/ai_core.py，提升架构清洁度
• 建立高性能本地工具抽象层，支持<10ms延迟的工具调用
• 集成外部MCP工具生态，扩展功能边界和创作能力
• 确保100%向后兼容性，现有用户工作流无任何影响

### 背景上下文
现有架构虽然功能完整，但存在职责分离不清的技术债务。ai/api.py混合了通用数据访问和AI特定业务逻辑，导致维护复杂性。同时，为了支持AI agent深度参与创作，需要将功能工具化并标准化。通过架构重构可以解决技术债务，同时为MCP集成奠定坚实基础。

### 变更日志
| 变更 | 日期 | 版本 | 描述 | 作者 |
|------|------|------|------|------|
| 初始创建 | 2025-09-24 | v3.0 | 基于架构重构需求的完整PRD | John (BMAD PM) |

---
