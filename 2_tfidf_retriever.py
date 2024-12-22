import os
import pandas as pd
import jieba
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from supabase import create_client
from dotenv import load_dotenv

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

class TfidfRetriever:
    def __init__(self, corpus):
        """
        初始化 TF-IDF 模型
        """
        self.vectorizer = TfidfVectorizer()
        self.corpus = corpus
        self.tfidf_matrix = self.vectorizer.fit_transform(corpus)

    def search(self, query, top_n=5):
        """
        使用 TF-IDF 進行檢索
        """
        query_vector = self.vectorizer.transform([query])
        similarities = cosine_similarity(query_vector, self.tfidf_matrix).flatten()
        top_indices = similarities.argsort()[-top_n:][::-1]
        return top_indices, similarities[top_indices]

def fetch_paragraphs(batch_size=100):
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

def process_and_update_results_tfidf(questions_df, retriever, paragraphs_df):
    """
    使用 TF-IDF 檢索並更新 Supabase
    """
    for ind, row in questions_df.iterrows():
        question_id = row['id']
        question = row['question']

        if not question or pd.isnull(question):
            print(f"跳過空問題 ID {question_id}")
            continue

        try:
            # 使用 TF-IDF 搜索
            top_indices, _ = retriever.search(" ".join(jieba.cut(question)))
            returned_doc_ids = paragraphs_df.iloc[top_indices]['id'].tolist()

            # 更新 Supabase
            response = supabase.table('multidoc_questions').update({
                'similar_doc_ids_tfidf': returned_doc_ids
            }).eq('id', question_id).execute()

            if response.data:
                print(f"成功更新問題 ID {question_id} 的相似文檔 ID：{returned_doc_ids}")
            else:
                print(f"問題 ID {question_id} 更新後沒有返回數據，可能更新失敗。")

        except Exception as e:
            print(f"更新問題 ID {question_id} 時發生錯誤：{e}")

# 主程式
if __name__ == "__main__":
    # 爬取段落資料
    paragraphs_df = fetch_paragraphs()
    paragraphs_df['processed_content'] = paragraphs_df['content'].apply(lambda x: " ".join(jieba.cut(x)))

    # 初始化 TF-IDF 檢索器
    retriever = TfidfRetriever(paragraphs_df['processed_content'].tolist())

    # 爬取問題資料
    questions_df = fetch_questions()

    # 使用 TF-IDF 搜尋並更新 Supabase
    process_and_update_results_tfidf(questions_df, retriever, paragraphs_df)
