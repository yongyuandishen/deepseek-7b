
import json
import time
import os
from openai import OpenAI

client = OpenAI(
    api_key="sk-fdafaba4c18e4afb8d7828981dd3a9c1",
    base_url="https://api.deepseek.com/v1"
)

MODEL_NAME = "deepseek-reasoner"
OUTPUT_PATH = "deepseek_emotion_master_style.jsonl"
ERROR_LOG_PATH = "emotion_generation_errors.log"
TOTAL_SAMPLES = 20000  # 修改为 10000 即可生成1万条

few_shot_messages = [
    {
        "role": "user",
        "content": "请生成一个带有幽默风格的情感问题及回答，并给出自然简洁的推理，使用JSON格式输出。"
    },
    {
        "role": "assistant",
        "content": json.dumps({
            "input": "小李为朋友付出了很多，但对方却开始疏远他，这种情况他可能会怎么想？",
            "output": "感情不是余额宝，投得多不一定有回报。",
            "rationale": "人在关系里最怕的就是高投入低反馈。他付出多，期待也高，但朋友不领情，就会让他开始怀疑这段关系甚至自我价值。"
        }, ensure_ascii=False)
    },
    {
        "role": "user",
        "content": "再来一个关于失恋后社交行为的例子，风格保持轻松幽默。",
    },
    {
        "role": "assistant",
        "content": json.dumps({
            "input": "小王失恋后把朋友圈全清空了，他是怎么想的？",
            "output": "清空的不是动态，是过去的自己。",
            "rationale": "这是很多人处理情绪的方式。他不是不痛，是太痛了，朋友圈就像回忆的展柜，清掉才不那么容易反刍过去。"
        }, ensure_ascii=False)
    }
]

system_prompt = {
    "role": "system",
    "content": (
        "你是一位拥有情绪洞察力、哲理性幽默感和共情能力的情感大师，"
        "擅长用简洁、生动、有温度的语言来分析人类的情绪与心理反应，"
        "融合网络高赞评论风格，避免长篇累牍或列条目式输出。"
        "每次输出请用 JSON 格式，包含 'input'（问题）、'output'（回答）、'rationale'（推理解释）。"
    )
}

mode = "a"
start_index = 0
if os.path.exists(OUTPUT_PATH):
    with open(OUTPUT_PATH, "r", encoding="utf-8") as f:
        start_index = sum(1 for _ in f)
    mode = "a"
else:
    start_index = 0
    mode = "w"

print(f"从第 {start_index + 1} 条开始生成，共 {TOTAL_SAMPLES} 条")

with open(OUTPUT_PATH, mode, encoding="utf-8") as fout, open(ERROR_LOG_PATH, "a", encoding="utf-8") as ferr:
    for i in range(start_index, TOTAL_SAMPLES):
        user_prompt = {
            "role": "user",
            "content": (
                "请生成一个情感相关的问题、带格局又带点网络幽默风的回答，并给出自然语言推理解释，"
                "风格参考高赞短评，不要啰嗦，输出JSON对象。"
            )
        }

        messages = [system_prompt] + few_shot_messages + [user_prompt]

        for attempt in range(5):
            try:
                response = client.chat.completions.create(
                    model=MODEL_NAME,
                    messages=messages,
                    temperature=0.7,
                    max_tokens=512
                )
                content = response.choices[0].message.content.strip()
                print(f"第 {i+1} 条返回内容：\n", repr(content))

                if not content:
                    raise ValueError("❌ 模型返回内容为空")

                if content.startswith("```json"):
                    content = content.removeprefix("```json").strip("`").strip()
                elif content.startswith("```"):
                    content = content.strip("`").strip()

                data = json.loads(content)
                json_line = json.dumps(data, ensure_ascii=False)
                fout.write(json_line + "\n")
                fout.flush()
                print(f"[{i+1}/{TOTAL_SAMPLES}] ✔ 写入完成：{data['input'][:30]}...")
                break
            except Exception as e:
                print(f"⚠️ 第 {i+1} 条失败，第 {attempt+1} 次尝试：{e}")
                time.sleep(2)
        else:
            ferr.write(json.dumps({
                "index": i + 1,
                "error": str(e)
            }, ensure_ascii=False) + "\n")
            ferr.flush()
            print(f"⛔ 跳过第 {i+1} 条（重试失败）")
            continue

        time.sleep(0.5)
        if (i + 1) % 100 == 0:
            time.sleep(2)
