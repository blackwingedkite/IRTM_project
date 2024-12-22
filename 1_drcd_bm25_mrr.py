import os
from supabase import create_client
from dotenv import load_dotenv

"""
Hit Rate: 0.9750930432293158
MRR: 0.93229315774406
"""

# 載入環境變數
load_dotenv()

# Supabase 配置
SUPABASE_URL = os.environ.get('SUPABASE_URL')
SUPABASE_API_KEY = os.environ.get('SUPABASE_API_KEY')

# 確保 Supabase 配置正確
if not SUPABASE_URL or not SUPABASE_API_KEY:
    raise ValueError("Supabase URL 或 API Key 未正確設置！請檢查 .env 檔案")

# 初始化 Supabase 客戶端
supabase = create_client(SUPABASE_URL, SUPABASE_API_KEY)

# 查詢資料
response = supabase.table('drcd_questions').select('dataset_id, similar_doc_ids_bm25_2, paragraph_id').execute()


# 資料處理
data = response.data

# 初始化計算指標
hit_data = []
mmr_data = []

# 定義函數
def find_index(predicted_ids, true_id):
    """查找正確文章 ID 在預測列表中的索引位置"""
    try:
        return predicted_ids.index(true_id)
    except ValueError:
        return "not_found"

def calculate_average(scores):
    """計算分數平均值"""
    return sum(scores) / len(scores) if scores else 0

# 遍歷資料計算指標
for row in data:
    similar_doc_ids = row['similar_doc_ids_bm25_2']
    gold_doc_id = row['paragraph_id']

    # 計算 Reciprocal Rank
    position = find_index(similar_doc_ids, gold_doc_id)
    if position == "not_found":
        score = 0
    else:
        score = 1 / (position + 1)  # Reciprocal Rank 計算公式

    # 記錄指標
    mmr_data.append(score)
    hit_data.append(gold_doc_id in similar_doc_ids)

# 計算平均命中率和 MRR
average_hit = sum(hit_data) / len(hit_data)
average_mrr = calculate_average(mmr_data)

# 輸出結果
print(f"Hit Rate: {average_hit}")
print(f"MRR: {average_mrr}")
