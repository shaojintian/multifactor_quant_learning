import csv

# 原始CSV文件的路径
input_csv_path = '/Users/wanting/Downloads/multifactor_quant_learning/data/commodities_data/csi500_futures_1d_data.csv'
# 新CSV文件的路径，其中包含英文头部
output_csv_path = '/Users/wanting/Downloads/multifactor_quant_learning/data/commodities_data/csi500_futures_1d.csv'

header_mapping = {
    "日期": "open time",
    "开盘价": "open",
    "最高价": "high",
    "最低价": "low",
    "收盘价": "close",
    "成交量": "volume",
    "持仓量": "holding_volume",
    "动态结算价": "dynamic_cleaning_price",
    # 新列不需要在映射中，因为我们会在代码中直接添加它们
}

with open(input_csv_path, mode='r', encoding='utf-8') as infile, \
     open(output_csv_path, mode='w', newline='', encoding='utf-8') as outfile:
    
    # 读取CSV文件
    reader = csv.DictReader(infile)
    
    # 获取翻译后的头部
    translated_headers = [header_mapping[header] for header in reader.fieldnames]
    
    # 添加新列到头部
    new_columns = ['quote asset volume', 'number of trades', 'taker buy base asset volume', 'taker buy quote asset volume']
    translated_headers += new_columns
    
    # 使用翻译后的头部创建新的DictWriter
    writer = csv.DictWriter(outfile, fieldnames=translated_headers)
    
    # 写入头部行
    writer.writeheader()
    
    # 写入数据行，翻译键
    for row in reader:
        translated_row = {header_mapping[ch_header]: row[ch_header] for ch_header in row}
        
        # 添加新列，并初始化为0
        for new_col in new_columns:
            translated_row[new_col] = 0  # 使用0作为默认值
        
        writer.writerow(translated_row)