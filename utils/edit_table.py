import pandas as pd
import os


# folder_path = '/root/zc/tgn_base_iot_nids/datasets/Bot-IoT'
def merge_table(folder_path, output_file):
    # 指定包含CSV文件的文件夹路径

    # 获取文件夹中的所有CSV文件列表
    csv_files = [f for f in os.listdir(folder_path) if f.endswith('.csv')]

    # 创建一个空的DataFrame，用于存储合并后的数据
    merged_data = pd.DataFrame()

    # 逐个读取并合并CSV文件
    for csv_file in csv_files:
        file_path = os.path.join(folder_path, csv_file)
        df = pd.read_csv(file_path)
        merged_data = pd.concat([merged_data, df], ignore_index=True)

    # 将合并后的数据保存为新的CSV文件
    merged_data.to_csv(output_file, index=False)

    print(f'Merged data saved to {output_file}')


def move_column_to_position_in_data(data, target_column_name, target_column_position):
    # 获取目标列的索引位置
    target_column_index = data.columns.get_loc(target_column_name)
    # 确保目标列号在有效范围内
    if target_column_position < 0 or target_column_position > len(data.columns):
        print("Invalid target column position.")
        return
    # 重新排列列的顺序
    columns = list(data.columns)
    columns.pop(target_column_index)
    columns.insert(target_column_position, target_column_name)
    
    # 根据新的列顺序创建新的DataFrame
    data = data[columns]
    print(f"Column '{target_column_name}' moved to position {target_column_position}.")
    return data

def move_column_to_position(input_csv, target_column_name, target_column_position, output_csv):
    # 读取CSV文件
    df = pd.read_csv(input_csv)
    
    # 检查目标列是否存在于DataFrame中
    if target_column_name not in df.columns:
        print(f"Column '{target_column_name}' not found in the CSV.")
        return
    
    # 获取目标列的索引位置
    target_column_index = df.columns.get_loc(target_column_name)
    
    # 确保目标列号在有效范围内
    if target_column_position < 0 or target_column_position > len(df.columns):
        print("Invalid target column position.")
        return
    
    # 重新排列列的顺序
    columns = list(df.columns)
    columns.pop(target_column_index)
    columns.insert(target_column_position, target_column_name)
    
    # 根据新的列顺序创建新的DataFrame
    df = df[columns]
    
    # 将新的DataFrame保存为CSV文件
    df.to_csv(output_csv, index=False)
    print(f"Column '{target_column_name}' moved to position {target_column_position} and saved to '{output_csv}'.")


if __name__ == "__main__":
    folder_path = '/root/zc/tgn_base_iot_nids/datasets/BoT-IoT'
    file = '/root/zc/tgn_base_iot_nids/datasets/BoT-IoT/BoT-IoT.csv'
    # move_column_to_position(file, 'saddr', 1, file)
    # move_column_to_position(file, 'sport', 2, file)
    # move_column_to_position(file, 'daddr', 3, file)
    # move_column_to_position(file, 'dport', 4, file)
    # move_column_to_position(file, 'stime', 5, file)
    # move_column_to_position(file, 'attack', 6, file)
    data = pd.read_csv(file)
    move_column_to_position_in_data(data, 'saddr', 1)
    # merge_table(folder_path, file)
    