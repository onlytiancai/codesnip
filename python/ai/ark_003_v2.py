from arkitect.core.component.context.context import Context
from pydantic import Field
import asyncio
import logging

# 配置日志，输出详细信息
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler()  # 输出到控制台
    ]
)
logger = logging.getLogger(__name__)

# 模拟数据库：姓名到学号的映射
name_to_id = {
    "张三": "2023001",
    "李四": "2023002",
    "王五": "2023003"
}

# 模拟数据库：学号对应的分数
student_scores = {
    "2023001": 80,
    "2023002": 75,
    "2023003": 90
}

def get_student_id(name: str = Field(description="同学的姓名，用于查询对应的学号")) -> str:
    """
    根据同学姓名查询对应的学号
    """
    logger.info(f"开始查询姓名为「{name}」的学号")
    
    if not name:
        logger.error("查询失败：姓名不能为空")
        raise ValueError("姓名不能为空")
        
    student_id = name_to_id.get(name)
    if not student_id:
        logger.error(f"查询失败：未找到姓名为「{name}」的学生信息")
        raise KeyError(f"未找到姓名为「{name}」的学生")
        
    logger.info(f"查询成功：「{name}」的学号为「{student_id}」")
    return student_id

def add_score(student_id: str = Field(description="学生的学号"), 
              points: int = Field(description="要增加的分数，必须是正整数")) -> str:
    """
    根据学号给学生增加指定分数
    """
    logger.info(f"开始给学号「{student_id}」增加「{points}」分")
    
    # 验证输入
    if not student_id:
        logger.error("加分失败：学号不能为空")
        raise ValueError("学号不能为空")
        
    if not isinstance(points, int) or points <= 0:
        logger.error(f"加分失败：分数「{points}」必须是正整数")
        raise ValueError("分数必须是正整数")
        
    # 检查学号是否存在
    if student_id not in student_scores:
        logger.error(f"加分失败：未找到学号为「{student_id}」的学生")
        raise KeyError(f"未找到学号为「{student_id}」的学生")
        
    # 执行加分操作
    old_score = student_scores[student_id]
    student_scores[student_id] += points
    new_score = student_scores[student_id]
    
    logger.info(f"加分成功：学号「{student_id}」分数从「{old_score}」变为「{new_score}」")
    return f"学号「{student_id}」加分成功，当前分数：{new_score}"

async def process_score_addition():
    # 初始化上下文，注册工具函数
    ctx = Context(
        model="doubao-seed-1-6-251015",
        tools=[get_student_id, add_score]
    )
    await ctx.init()
    
    # 用户输入（可以改为input()函数获取实际输入）
    user_input = "给张三加5分，给李四加3分，给小明加2分"
    logger.info(f"收到用户输入：{user_input}")

    messages=[
        {"role": "system", "content": """
        你需要处理用户的加分指令，规则如下：
        1. 必须先通过get_student_id函数查询学生姓名对应的学号，不可直接猜测。
        2. 调用add_score函数时，分数必须是正整数，且需使用查询到的学号。
        3. 若有多个学生，需逐个处理，前一个失败不影响后一个。
        4. 最终返回所有处理结果，包括成功和失败的详情。
        """},
        {"role": "user", "content": user_input}
    ]

    # 处理请求
    completion = await ctx.completions.create(
        messages=messages,
        stream=False
    )
    
    return completion

if __name__ == "__main__":
    try:
        result = asyncio.run(process_score_addition())
        logger.info("处理结果：")
        print(result.choices[0].message.content)
        # 打印最终分数状态
        logger.info("当前所有学生分数：")
        for student_id, score in student_scores.items():
            # 查找对应的姓名
            name = next((k for k, v in name_to_id.items() if v == student_id), "未知")
            logger.info(f"{name}（{student_id}）：{score}分")
    except Exception as e:
        logger.error(f"系统错误：{str(e)}", exc_info=True)