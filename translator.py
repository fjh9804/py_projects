import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline
from tqdm import tqdm

def offline_translate_excel(file_path):
    # 【关键修改】：指向你拷贝过来的本地文件夹路径
    local_model_path = "./nllb_model_local" 
    
    print(f"正在从本地路径加载模型: {local_model_path}...")
    
    # 这里的所有加载操作现在都是 100% 离线的
    tokenizer = AutoTokenizer.from_pretrained(local_model_path)
    model = AutoModelForSeq2SeqLM.from_pretrained(local_model_path)

    device = 0 if torch.cuda.is_available() else -1
    translator = pipeline(
        "translation", 
        model=model, 
        tokenizer=tokenizer, 
        src_lang="zho_Hans", 
        tgt_lang="eng_Latn", 
        max_length=400,
        device=device
    )

    # ... 后续读取 Excel 和翻译的逻辑与之前一致 ...
    # (此处省略重复的 pandas 处理逻辑)




import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline
from tqdm import tqdm

def offline_translate_excel(file_path):
    # 1. 加载本地模型 (第一次运行会自动下载)
    model_name = "facebook/nllb-200-distilled-600M"
    print(f"正在加载离线模型: {model_name}...")
    
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

    # 创建翻译管道，如果有 GPU 会自动使用 GPU (device=0)
    device = 0 if torch.cuda.is_available() else -1
    translator = pipeline(
        "translation", 
        model=model, 
        tokenizer=tokenizer, 
        src_lang="zho_Hans", # 简体中文
        tgt_lang="eng_Latn", # 英文
        max_length=400,
        device=device
    )

    try:
        # 2. 读取 Excel
        df = pd.read_excel(file_path)
        col_b_name = df.columns[1]  # B列
        
        print(f"开始离线翻译，共 {len(df)} 行...")
        
        # 3. 执行翻译
        translated_results = []
        for text in tqdm(df[col_b_name]):
            if pd.isna(text) or str(text).strip() == "":
                translated_results.append("")
            else:
                # 执行翻译并提取文本
                output = translator(str(text))
                translated_results.append(output[0]['translation_text'])

        # 4. 写入 C 列并保存
        df['英文描述(离线)'] = translated_results
        output_file = file_path.replace(".xlsx", "_offline_translated.xlsx")
        df.to_excel(output_file, index=False)
        
        print(f"\n✅ 翻译完成！结果已保存至: {output_file}")

    except Exception as e:
        print(f"❌ 运行中出现错误: {e}")

if __name__ == "__main__":
    # 修改为你的文件名
    target_file = 'your_data.xlsx'
    offline_translate_excel(target_file)
