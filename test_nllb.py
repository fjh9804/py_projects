import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline

# 1. 指定你的本地模型文件夹路径 (建议使用绝对路径防止出错)
model_path = "./nllb_model_local" 

print("正在初始化本地翻译引擎...")

try:
    # 2. 加载分词器和模型 (设置 local_files_only=True 确保完全离线)
    tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_path, local_files_only=True)

    # 3. 检查是否有可用 GPU (如果有 NVIDIA 显卡加速会非常快)
    device = 0 if torch.cuda.is_available() else -1
    if device == 0:
        print("检测到 GPU，正在使用显卡加速...")
    else:
        print("未检测到 GPU，使用 CPU 运行（处理较长文本可能会慢一些）")

    # 4. 创建翻译流水线
    # src_lang: zho_Hans (简体中文), tgt_lang: eng_Latn (英文)
    translator = pipeline(
        "translation",
        model=model,
        tokenizer=tokenizer,
        src_lang="zho_Hans",
        tgt_lang="eng_Latn",
        max_length=400,
        device=device
    )

    # 5. 测试用例
    test_texts = [
        "系统登录超时",
        "用户账户余额不足",
        "请确认您的订单信息"
    ]

    print("\n--- 开始测试翻译 ---")
    for text in test_texts:
        result = translator(text)
        translated_text = result[0]['translation_text']
        print(f"原文: {text}")
        print(f"译文: {translated_text}\n")
    print("--- 测试完成 ---")

except Exception as e:
    print(f"\n❌ 出错了！错误原因可能是：")
    print(f"1. 路径不对：请检查 {model_path} 是否包含所有文件。")
    print(f"2. 缺少依赖：请运行 pip install sentencepiece torch transformers")
    print(f"\n错误详情: {e}")
