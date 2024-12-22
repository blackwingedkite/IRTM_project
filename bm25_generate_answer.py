import os
import numpy as np
import pandas as pd
import time
import json
import jieba
from supabase import create_client
from dotenv import load_dotenv
from sklearn.feature_extraction.text import TfidfVectorizer
import openai
import warnings

# 載入環境變數
load_dotenv()

# Supabase 配置
SUPABASE_URL = os.environ.get('SUPABASE_URL')
SUPABASE_API_KEY = os.environ.get('SUPABASE_API_KEY')
openai_api_key = os.environ.get("OPENAI_API_KEY")

# 確保環境變數載入成功
if not SUPABASE_URL or not SUPABASE_API_KEY:
    raise ValueError("Supabase URL 或 API Key 未正確設置！請檢查 .env 檔案")

# 初始化 Supabase 客戶端
supabase = create_client(SUPABASE_URL, SUPABASE_API_KEY)

# 初始化 OpenAI API
openai.api_key = openai_api_key

# 忽略警告
warnings.filterwarnings('ignore')


def fetch_paragraphs(ids):
    """
    根据给定的段落 ID 列表从 Supabase 获取对应的段落数据。
    :param ids: 一个包含段落 ID 的列表，用于筛选需要提取的段落。
    :return: 返回包含指定段落的 DataFrame。
    """
    response = supabase.table('multidoc_sources').select('id', 'title', 'content').in_('id', ids).execute()
    paragraphs = response.data
    
    return pd.DataFrame(paragraphs)


def fetch_questions(batch_size=100):
    offset = 0
    all_questions = []  # 初始化為空列表
    while True:
        response = supabase.table('multidoc_questions').select('id', 'question', 'similar_doc_ids_bm25').range(offset, offset + batch_size - 1).execute()
        questions = response.data
        if not questions:
            break

        all_questions.extend(questions)  # 使用 extend 平坦化列表
        offset += batch_size
        print(f"Fetched {len(questions)} records... Total so far: {len(all_questions)}")

    # 按 ID 排序
    all_questions.sort(key=lambda x: x['id'])
    print(f"總共取到 {len(all_questions)} 個問題")
    return all_questions


def generate_answer(context, question):
    prompt = f"""
    基於以下文章內容回答問題：

    文章內容：{context}

    問題：{question}

    請提供準確且完整的回答。如果文章內容無法妥善回答問題，請回答「我不知道」
    Answer:"""

    # 在這裡加上打印出生成的 prompt
    #print("生成的 Prompt：")
    #print(prompt)  # 打印出 Prompt


    retries = 3  # Number of retry attempts
    rate_limit_per_minute = 15  # 每分钟最多请求次数
    delay = 60.0 / rate_limit_per_minute  # 计算每次请求之间的延迟时间

    for attempt in range(retries):
        try:
            time.sleep(delay)  # Apply rate limit delay
            response = openai.ChatCompletion.create(
                model="gpt-4o",
                messages=[
                    {"role": "system", "content": "你是一個專業的問答助手"},
                    {"role": "user", "content": prompt}
                ]
            )

            tokens_used = response['usage']['total_tokens']
            print(f"Tokens used in this request: {tokens_used}")

            # 获取并返回生成的回答
            answer = response['choices'][0]['message']['content']
            return answer.strip()

        except openai.error.RateLimitError as e:
            print(f"Rate limit error: {e}")
            if attempt < retries - 1:
                print("Retrying after a short wait...")
                time.sleep(5)  # Wait for 5 seconds before retrying
            else:
                print("Maximum retries reached, unable to process the request.")
                return None
        except Exception as e:
            print(f"Error during OpenAI API call: {e}")
            return None


def process_supabase_rag(questions, start_id=0):
    """
    處理 Supabase 問題列表，從指定的問題 ID 開始處理。
    :param questions: 問題列表
    :param start_id: 指定起始問題 ID，小於此 ID 的問題會被跳過
    """
    error_count = 0
    processed_count = 0
    total_questions = len(questions)

    for question in questions:
        question_id = question['id']
        if question_id < start_id:
            print(f"跳過問題 ID {question_id}，小於起始 ID {start_id}...")
            continue

        print(f"處理問題 ID {question_id}...")

        question_text = question['question']

        similar_doc_ids_bm25 = question['similar_doc_ids_bm25']

        if not similar_doc_ids_bm25:
            print(f"問題 {question_id} 沒有相關的文章，跳過...")
            continue

        # 提取文章
        retrieved_paragraphs = fetch_paragraphs(similar_doc_ids_bm25)

        if retrieved_paragraphs.empty:
            print(f"問題 {question_id} 無法找到相關段落，跳過...")
            continue

        # 用相關文章生成回答
        context = "\n\n".join([
            f"文章標題: {row['title']}\n內容: {row['content']}"
            for _, row in retrieved_paragraphs.iterrows()
        ])

        # 生成答案
        generated_answer = generate_answer(context, question_text)

        # 打印問題與答案
        print(f"問題 ID: {question_id}")
        print(f"問題內容: {question_text}")
        print(f"生成答案: {generated_answer}")
        print("-" * 50)  # 分隔線，方便閱讀

        # 更新 Supabase
        try:
            supabase.table('multidoc_questions').update({
                'gpt_answer_bm25': generated_answer
            }).eq('id', question_id).execute()

            processed_count += 1

        except Exception as e:
            error_count += 1
            print(f"Error updating the database for question ID {question_id}: {e}")

        # 打印已處理問題的數量及錯誤數量
        print(f"已處理問題數量: {processed_count}/{total_questions}")
        print(f"錯誤數量: {error_count}")

def process_supabase_rag_batched(questions, batch_size=10, start_id=0):
    """
    批量處理 Supabase 問題列表，從指定的問題 ID 開始處理。
    :param questions: 問題列表
    :param batch_size: 每次處理的問題數量
    :param start_id: 指定起始問題 ID，小於此 ID 的問題會被跳過
    """
    error_count = 0
    processed_count = 0
    total_questions = len(questions)

    for i in range(0, total_questions, batch_size):
        batch = questions[i:i + batch_size]
        batch = [q for q in batch if q['id'] >= start_id]  # 跳過小於起始 ID 的問題

        if not batch:
            continue

        print(f"處理批次 {i // batch_size + 1}，包含 {len(batch)} 個問題...")

        # 批量提取相關段落 ID
        all_doc_ids = set(
            doc_id for q in batch if q.get('similar_doc_ids_bm25') 
            for doc_id in q['similar_doc_ids_bm25']
        )
        retrieved_paragraphs = fetch_paragraphs(list(all_doc_ids))

        if retrieved_paragraphs.empty:
            print("無法找到任何相關段落，跳過該批次...")
            continue

        # 為每個問題生成上下文
        context_dict = {}
        for question in batch:
            doc_ids = question['similar_doc_ids_bm25']
            if not doc_ids:
                context_dict[question['id']] = None
                continue

            related_docs = retrieved_paragraphs[retrieved_paragraphs['id'].isin(doc_ids)]
            if related_docs.empty:
                context_dict[question['id']] = None
                continue

            context_dict[question['id']] = "\n\n".join([
                f"文章標題: {row['title']}\n內容: {row['content']}"
                for _, row in related_docs.iterrows()
            ])

        # 批量生成答案
        prompts = [
            f"""
            基於以下文章內容回答問題：

            文章內容：{context}

            問題：{question['question']}

            請提供準確且完整的回答。如果文章內容無法妥善回答問題，請回答「我不知道」"""
            for question in batch
            if (context := context_dict.get(question['id']))
        ]

        try:
            response = openai.ChatCompletion.create(
                model="gpt-4o",
                messages=[
                    {"role": "system", "content": "你是一個專業的問答助手"},
                ] + [{"role": "user", "content": prompt} for prompt in prompts]
            )
            answers = {batch[idx]['id']: choice['message']['content'].strip()
                       for idx, choice in enumerate(response['choices'])}
        except Exception as e:
            print(f"批量生成答案失敗: {e}")
            error_count += len(batch)
            continue

        # 準備更新的資料，排除 `id` 欄位
        updates = [
            {'gpt_answer_bm25': answers.get(question['id'], None)}
            for question in batch
        ]

        try:
            # 使用條件更新，而不是直接插入
            for question, update in zip(batch, updates):
                supabase.table('multidoc_questions').update(update).eq('id', question['id']).execute()
            processed_count += len(batch)
        except Exception as e:
            print(f"批量更新資料庫失敗: {e}")
            error_count += len(batch)

        # 打印進度
        print(f"已處理批次 {i // batch_size + 1}/{(total_questions + batch_size - 1) // batch_size}")
        print(f"已處理問題數量: {processed_count}/{total_questions}")
        print(f"錯誤數量: {error_count}")


if __name__ == "__main__":
    # 指定起始問題 ID
    start_id = int(input("請輸入起始問題 ID（預設為 0）：") or 0)

    # 獲取所有問題
    questions = fetch_questions()

    # 從指定 ID 開始處理
    process_supabase_rag(questions, start_id=start_id)
