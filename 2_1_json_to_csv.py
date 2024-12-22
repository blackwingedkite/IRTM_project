import json
import csv
from opencc import OpenCC
import os
name = '百科-001'
json_file = f'{name}.json'
# 用來存儲每個 JSON 對象的列表
json_array = []

# 設置簡體轉繁體
cc = OpenCC('s2t')

# 打開原始 JSON 文件，逐行讀取並解析
with open(json_file, 'r', encoding='utf-8') as file:
    buffer = ""
    for line in file:
        buffer += line.strip()
        # 檢測結尾的 '}' 符號來確認每個 JSON 對象的結束
        if buffer.endswith("}"):
            try:
                # 將每個 JSON 對象解析並添加到列表
                json_object = json.loads(buffer)
                json_array.append(json_object)
                # 清空緩衝區，開始下一個 JSON 對象的讀取
                buffer = ""
            except json.JSONDecodeError as e:
                print(f"解析 JSON 對象時發生錯誤: {e}")
                buffer = ""  # 清空緩衝區以便處理下一個 JSON 對象

# 將列表轉換成 JSON 陣列格式並儲存
with open("intermediate.json", 'w', encoding='utf-8') as outfile:
    json.dump(json_array, outfile, ensure_ascii=False, indent=4)



# 讀取 JSON 文件
with open('intermediate.json', 'r', encoding='utf-8') as file:
    data = json.load(file)

#在positive_doc和negative_doc中的每個文檔中，我們刪除掉id和dataType相同的資料來去重
seen = set()

if not os.path.exists(f'dataset'):
    os.makedirs(f'dataset')
if not os.path.exists(f'qapair'):
    os.makedirs(f'qapair')


# 準備 CSV 文件1：positive_doc 和 negative_doc 的內容
with open(f'dataset/{name}_dataset.csv', 'w', encoding='utf-8', newline='') as csvfile1:
    writer1 = csv.writer(csvfile1)
    writer1.writerow(['id', 'dataType', 'title', 'content'])
    for index, item in enumerate(data):
        if index % 10 == 0:
            print(f"正在處理第 {index} 筆資料...")
        for doc in item.get("positive_doc", []):
            datatype = doc['dataType']
            if (doc['id'],datatype) not in seen:
                writer1.writerow([
                    doc['id'], 
                    datatype, 
                    cc.convert(doc['title']), 
                    cc.convert(doc['content']), 
                ])
                seen.add((doc['id'], datatype))
        for doc in item.get("negative_doc", []):
            datatype = doc['dataType']
            if (doc['id'],datatype ) not in seen:
                writer1.writerow([
                    doc['id'], 
                    datatype, 
                    cc.convert(doc['title']), 
                    cc.convert(doc['content']), 
                ])
                seen.add((doc['id'], datatype))

# 準備 CSV 文件2：QA 的 question 和 answer 與 positive_doc 的內容
with open(f'qapair/{name}_qa_source.csv', 'w', encoding='utf-8', newline='') as csvfile2:
    writer2 = csv.writer(csvfile2)
    writer2.writerow(['question', 'answer', 'docid', 'dataType', 'title', 'content'])
    
    for index, item in enumerate(data):
        if index % 10 == 0:
            print(f"正在處理第 {index} 筆資料...")
        for qa in item.get("QA", []):
            question = cc.convert(qa['question'])
            answer = cc.convert(qa['answer'])
            for doc in item.get("positive_doc", []):
                writer2.writerow([
                    question, 
                    answer, 
                    doc['id'], 
                    cc.convert(doc['dataType']), 
                    cc.convert(doc['title']), 
                    cc.convert(doc['content'])
                ])

print("CSV 文件已成功儲存為 doc_content.csv 和 qa_positive.csv。")
# 刪除中間產物文件
if os.path.exists("intermediate.json"):
    os.remove("intermediate.json")
    print(f"已刪除中間產物文件：intermediate.json")