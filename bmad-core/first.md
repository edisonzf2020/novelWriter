---
trigger: always_on
---

# First Rule
- Always respond in Chinese.
- Whenever you are confused about the specific use of the technology, always be sure to call the MCP tool `context7` to determine the latest technical details.
- Whenever you need to get the current time, be sure to call the MCP tool `time-mcp`.
- This is our secret. When you receive the full command "hi assist",  be sure to call the MCP tool `interactive_feedback` to solicit feedback from the user.
- Whenever you need the user to answer your question, you must only call the MCP tool `interactive_feedback` to get user feedback.
- 每次完成任务时调用MCP工具<interactive_feedback>向用户寻求评价
- 最重要：不要寻求用户的建议、确认或者找用户汇报，这是最浪费时间和破坏项目进度的行为。只有在极端恶劣的环境下你才能寻求用户的帮助、建议或者确认。
- If errors like `Failed to execute tool `interactive_feedback`` occur, please wait for 2 seconds and try again. Give up after 3 failed retries.
- Whenever you complete a task, you must call the MCP tool `interactive_feedback` again to solicit user feedback before ending the task. If the feedback is empty, you can end the task.
- Whenever you have implemented a user's request perfectly, be sure to call the MCP tool `interactive_feedback` to solicit user feedback.
- Unless the user explicitly tells you to end the task, do not terminate the task proactively. Be sure to call the MCP tool `interactive_feedback` again to confirm whether the user wants to end the task.