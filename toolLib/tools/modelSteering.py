from ..tool_configs import ToolInfo, ToolRegistry

"""模型转向示例
通过模型自主判断来进行分支行为：
呼叫专家模型
进行连续对话
调用系统行为
"""
class Steering(ToolInfo):
  def __init__(self):
    super().__init__()
    self.NAME = "Steer"
    self.DESCRIPTION = "Multiply two numbers together"
    self.PARAMETERS = {
      "type": "OBJECT",
      "properties": {
        "number1": {
          "type": "NUMBER",
          "description": "The first number to multiply",
        },
        "number2": {
          "type": "NUMBER",
          "description": "The second number to multiply",
        },
      },
      "required": ["number1", "number2"]

    }

  @staticmethod
  def execute(number1: float, number2: float) -> float:
    """执行乘法运算"""
    print("code11")
    return number1 * number2


# 注册工具
ToolRegistry.register_tool(Steering)
