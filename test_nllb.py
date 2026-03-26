import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

model_path = "./nllb_model_local"

print("正在初始化本地翻译引擎...")

try:
    tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_path, local_files_only=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    print(f"使用设备: {device}")

    # 设置源语言
    tokenizer.src_lang = "zho_Hans"

    test_texts = [
        "系统登录超时",
        "用户账户余额不足",
        "请确认您的订单信息"
    ]

    print("\n--- 开始测试翻译 ---")

    for text in test_texts:
        inputs = tokenizer(text, return_tensors="pt").to(device)

        outputs = model.generate(
            **inputs,
            forced_bos_token_id = tokenizer.convert_tokens_to_ids("eng_Latn"),
            max_new_tokens=128
        )

        translated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)

        print(f"原文: {text}")
        print(f"译文: {translated_text}\n")

    print("--- 测试完成 ---")

except Exception as e:
    print(f"\n❌ 出错: {e}")
