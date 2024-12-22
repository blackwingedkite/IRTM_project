import os
import numpy as np
import pandas as pd
import json
import jieba
from supabase import create_client
from dotenv import load_dotenv
from sklearn.feature_extraction.text import TfidfVectorizer

# 載入環境變數
load_dotenv()

# Supabase 配置
SUPABASE_URL = os.environ.get('SUPABASE_URL')
SUPABASE_API_KEY = os.environ.get('SUPABASE_API_KEY')

# 確保環境變數載入成功
if not SUPABASE_URL or not SUPABASE_API_KEY:
    raise ValueError("Supabase URL 或 API Key 未正確設置！請檢查 .env 檔案")

# 初始化 Supabase 客戶端
supabase = create_client(SUPABASE_URL, SUPABASE_API_KEY)

class BM25:
    def __init__(self, corpus, k1=1.5, b=0.75):
        self.k1 = k1
        self.b = b
        self.corpus = corpus
        self.doc_lengths = np.array([len(doc) for doc in corpus])  # 文件長度
        self.avg_doc_length = np.mean(self.doc_lengths)  # 平均文件長度
        self.term_freq = self._compute_term_freq(corpus)  # TF
        self.idf = self._compute_idf(corpus)  # IDF

    def _compute_term_freq(self, corpus):
        term_freq = []
        for doc in corpus:
            freq = {}
            for word in doc:
                freq[word] = freq.get(word, 0) + 1
            term_freq.append(freq)
        return term_freq

    def _compute_idf(self, corpus):
        num_docs = len(corpus)
        doc_freq = {}
        for doc in corpus:
            for word in set(doc):
                doc_freq[word] = doc_freq.get(word, 0) + 1
        idf = {word: np.log((num_docs - freq + 0.5) / (freq + 0.5) + 1) for word, freq in doc_freq.items()}
        return idf

    def compute_score(self, query, doc_index):
        score = 0
        for word in query:
            if word in self.term_freq[doc_index]:
                freq = self.term_freq[doc_index][word]
                idf = self.idf.get(word, 0)
                numerator = freq * (self.k1 + 1)
                denominator = freq + self.k1 * (1 - self.b + self.b * (self.doc_lengths[doc_index] / self.avg_doc_length))
                score += idf * (numerator / denominator)
        return score

    def search(self, query, top_n=5):
        query = list(jieba.cut(query))
        scores = [self.compute_score(query, idx) for idx in range(len(self.corpus))]
        top_indices = np.argsort(scores)[-top_n:][::-1]
        return top_indices, [scores[i] for i in top_indices]

def fetch_paragraphs(batch_size=100):
    """
    Fetch paragraphs from multidoc_sources.
    """
    offset = 0
    all_paragraphs = []
    while True:
        print(f"Fetching paragraphs from offset {offset} to {offset + batch_size - 1}")
        
        response = supabase.table('multidoc_sources').select('id', 'datatype', 'title', 'content', 'embedding')\
            .range(offset, offset + batch_size - 1).execute()
        paragraphs = response.data

        if not paragraphs:
            print("No more paragraphs to fetch.")
            break

        all_paragraphs.extend(paragraphs)
        offset += batch_size

    print(f"Total paragraphs fetched: {len(all_paragraphs)}")
    return pd.DataFrame(all_paragraphs)


def fetch_questions(batch_size=100):
    """
    Fetch qustions from multidoc_questions.
    """
    offset = 0
    all_questions = []
    while True:
        response = supabase.table('multidoc_questions').select('id', 'question').range(offset, offset + batch_size - 1).execute()
        questions = response.data
        if not questions:
            break
        all_questions.extend(questions)
        offset += batch_size
    return pd.DataFrame(all_questions)

def process_and_update_results(questions_df, bm25, paragraphs_df):
    for ind, row in questions_df.iterrows():
        question_id = row['id']
        question = row['question']
        top_indices, _ = bm25.search(question)
        returned_doc_ids = paragraphs_df.iloc[top_indices]['id'].tolist()

        try:
            # 嘗試更新 Supabase
            response = supabase.table('multidoc_questions').update({
                'similar_doc_ids_bm25': returned_doc_ids
            }).eq('id', question_id).execute()

            print(f"成功更新問題 ID {question_id} 的相似文檔 ID：{returned_doc_ids}")

        except Exception as e:
            # 捕捉任何異常
            print(f"更新問題 ID {question_id} 時發生錯誤：{e}")

# 主程式
if __name__ == "__main__":
    # 爬取段落資料並處理斷詞
    paragraphs_df = fetch_paragraphs()
    paragraphs_df['tokenized_content'] = paragraphs_df['content'].apply(lambda x: list(jieba.cut(x)))

    # 初始化 BM25 模型
    bm25 = BM25(paragraphs_df['tokenized_content'].tolist())

    # 爬取問題資料
    questions_df = fetch_questions()

    # 搜尋相關段落並更新 Supabase
    process_and_update_results(questions_df, bm25, paragraphs_df)
