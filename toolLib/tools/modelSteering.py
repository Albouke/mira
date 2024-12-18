from ..tool_configs import ToolInfo, ToolRegistry

"""模型转向示例
通过模型自主判断来进行分支行为：
呼叫专家模型
进行连续对话
调用系统行为

class Steering(ToolInfo):
没写完


"""

# 注册工具
ToolRegistry.register_tool(Steering)
