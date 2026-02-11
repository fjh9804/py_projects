import pandas as pd


def generate_cpp_struct(file_path):
    # 1. 读取 Excel 文件
    # 确保已安装 openpyxl: pip install openpyxl
    df = pd.read_excel(file_path)

    struct_lines = []

    # 2. 逐行进行逻辑处理
    for index, row in df.iterrows():
        # 获取各列的值（注意：这里的列名需与 Excel 中完全一致）
        data_format = str(row['dataformat']).strip()
        size = str(row['size']).strip()
        param_name = str(row['参数名']).strip()

        # 3. 核心需求判断：如果 dataformat 列为 "int"
        if data_format.lower() == "int":
            # 拼接逻辑：将 "int"、"size"列、"参数名"列进行拼接
            # 例如生成：int32_t myParam; 或者 int myParam[10];
            # 这里假设拼接成：int {size} {param_name};
            line = f"    UT{size} {param_name};"
            struct_lines.append(line)
        elif data_format.lower() == "float":
            line = f"    FT{size} {param_name};"
            struct_lines.append(line)
        else:
            # 其他类型的处理逻辑（可选）
            pass

    # 4. 组装成最终的结构体格式
    result = "struct {\n"
    result += "\n".join(struct_lines)
    result += "\n};"

    return result

# 使用示例
struct_cpp = generate_cpp_struct('config.xlsx')
print(struct_cpp)
