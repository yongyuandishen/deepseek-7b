import torch
import numpy as np
import faiss
import gradio as gr
from datasets import load_dataset
from sentence_transformers import SentenceTransformer
from transformers import (
    AutoTokenizer, AutoModelForCausalLM,
    GenerationConfig, BitsAndBytesConfig
)

# === 中文向量模型 ===
embedder = SentenceTransformer("./text2vec-large-chinese")

# === 加载本地数据集 ===
dataset = load_dataset("json", data_files="deepseek_emotion_master_style.jsonl")["train"]
data_qa = [{"question": item["input"], "answer": item["output"]} for item in dataset]
questions = [item["question"] for item in data_qa]
answers = [item["answer"] for item in data_qa]

# === 构建检索索引 ===
question_embeddings = embedder.encode(questions, convert_to_numpy=True)
faiss.normalize_L2(question_embeddings)
dim = question_embeddings.shape[1]
index = faiss.IndexFlatIP(dim)
index.add(question_embeddings)

# === 本地加载 DeepSeek 模型（4bit） ===
model_path = "/home/zdb/deepseek-7b/deepseek-llm-7b-chat"
bnb = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4"
)

tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=True, local_files_only=True)
model = AutoModelForCausalLM.from_pretrained(
    model_path,
    device_map="auto",
    quantization_config=bnb,
    trust_remote_code=True,
    local_files_only=True
)
model.generation_config = GenerationConfig.from_pretrained(model_path, local_files_only=True)
model.generation_config.pad_token_id = model.generation_config.eos_token_id

# === 响应逻辑 ===
def respond(msg, history):
    if history is None:
        history = []

    q_emb = embedder.encode([msg], convert_to_numpy=True)
    faiss.normalize_L2(q_emb)
    D, I = index.search(q_emb, k=1)
    similarity = D[0][0]

    threshold_direct = 0.8
    threshold_hint = 0.5

    if similarity > threshold_direct:
        ans = answers[I[0][0]].strip()

    elif similarity > threshold_hint:
        ref_q = questions[I[0][0]]
        ref_a = answers[I[0][0]]
        ref_r = dataset[I[0][0]]["rationale"]
        prompt = (
            f"请参考以下范例并模仿其风格，用简洁中文回答用户问题（不超过50字）：\n\n"
            f"范例问题：{ref_q}\n范例回答：{ref_a}\n范例理由：{ref_r}\n\n"
            f"用户问题：{msg}\n回答："
        )
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        out = model.generate(
            **inputs,
            max_new_tokens=64,
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
            pad_token_id=model.generation_config.eos_token_id
        )
        ans = tokenizer.decode(out[0, inputs["input_ids"].shape[1]:], skip_special_tokens=True).strip()

    else:
        prompt = f"请用简洁自然的中文，最多50字内回答用户问题：\n用户：{msg}\n助手："
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        out = model.generate(
            **inputs,
            max_new_tokens=64,
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
            pad_token_id=model.generation_config.eos_token_id
        )
        ans = tokenizer.decode(out[0, inputs["input_ids"].shape[1]:], skip_special_tokens=True).strip()

    history.append({"role": "user", "content": msg})
    history.append({"role": "assistant", "content": ans})
    return history, history

# === 本地头像 ===
USER_AVATAR = "question.jpg"
BOT_AVATAR  = "answer.jpg"

# === UI 样式 ===
css = """
body { background: #e8f5e9 !important; }
#chat-window {
  background: #ffffff;
  width: 540px;
  height: 700px;
  margin: 30px auto;
  padding: 12px;
  border-radius: 12px;
  box-shadow: 0 2px 10px rgba(0,0,0,0.1);
}
#chatbot .message.user {
  display: flex; justify-content: flex-end; margin-bottom: 10px;
}
#chatbot .message.user .bubble {
  background-color: #dcf8c6; color: #000; max-width: 70%;
  padding: 10px; border-radius: 15px 15px 0 15px;
  position: relative;
}
#chatbot .message.user .bubble::after {
  content: ''; position: absolute; right: -10px; bottom: 0;
  border-width: 10px 10px 0 0; border-style: solid;
  border-color: #dcf8c6 transparent transparent transparent;
}
#chatbot .message.assistant {
  display: flex; justify-content: flex-start; margin-bottom: 10px;
}
#chatbot .message.assistant .bubble {
  background-color: #f1f0f0; color: #000; max-width: 70%;
  padding: 10px; border-radius: 15px 15px 15px 0;
  position: relative;
}
#chatbot .message.assistant .bubble::after {
  content: ''; position: absolute; left: -10px; bottom: 0;
  border-width: 10px 0 0 10px; border-style: solid;
  border-color: transparent transparent transparent #f1f0f0;
}
textarea, input { background: #fff !important; color: #000 !important; }
"""

# === Gradio App ===
with gr.Blocks(css=css) as demo:
    with gr.Column(elem_id="chat-window"):
        gr.Markdown("### 🤖 DeepSeek-7B 微信风格聊天")
        chatbot = gr.Chatbot(
            elem_id="chatbot",
            label="",
            type="messages",
            avatar_images=(USER_AVATAR, BOT_AVATAR)
        )
        state = gr.State([])

        with gr.Row():
            txt   = gr.Textbox(placeholder="请输入...", show_label=False, lines=1, scale=8)
            send  = gr.Button("发送", variant="primary", scale=1)
            clear = gr.Button("清空", variant="secondary", scale=1)

        send.click(respond, [txt, state], [chatbot, state])
        txt.submit(respond, [txt, state], [chatbot, state])
        clear.click(lambda: ([], []), None, [chatbot, state])

demo.launch()
