# IRTM_project
Goal: Evaluation retrieval method on RAG project.

------2024/11/20更新------
第1部分：不做G，只做RA得到結果
第2部分：做完整的RAG獲得資料

分工給大家的內容：
1. 尋找適合的retriever並實作
2. 研究相關的內容，例如他的ensemble retriever的機制等等，但我確實不知道該怎麼分配比較好

supabase資訊：
Organization: IRTM_project
Project_name: IRTM_RAGproject
Db pwd: QwePoi!)2938
Region: Seoul

------2024/11/07更新------

## something to discuss:
我們的課程是資料檢索文字探勘，所以我們會著重在資料檢索上面
理論上的RAG流程：
ETL process: 原始檔案（舉例來說，一個銀行的法規pdf檔案，包含文字和表格）> 資料前處理（把字詞擷取下來並進行chunking-> embedding > 存進資料庫
QA process: 問題 > 根據問題獲取相關資訊 > 獲得答案

我們的內容可以最關注在根據問題獲取資訊的部分

目前這篇文章和我的想法很樣，有點擔心內容重複，但是我相信建僅不會看rag的論文所以應該沒事
https://arxiv.org/abs/2404.07220

額外可做想法1: 建立中文資料集進行效能評估 合成數據
額外可做想法2: 圖形資料庫補充、最佳化搜尋策略以及成本考量

在我們的專案中可以考慮實作中文的檢索策略，因為英文的賽道好像很捲

## 理想的資料集內容
- 繁體中文pdf檔案
- 針對這個繁體中文pdf的問題以及回答

## 可以使用的資料集內容：
1. DRCD
    - 架構：參考文字(非pdf)，問題、回答
    - 優點：很乾淨，不需要進行資料前處理（已經是json檔案），已經協助sementic chunking
    - 缺點：太乾淨了，和實際上的情況不符合
    - Thought: 如果我們只是要檢測retrieval資料檢索的能力的話，或許可以？
    - references: https://github.com/DRCKnowledgeTeam/DRCD
    - references: https://huggingface.co/datasets/MediaTek-Research/TCEval-v2
    - 1000個documents(都很短), 3400個questions documents的長度在250-950之間，所以可以直接用MRR


2. yuyijiong/Multi-Doc-QA-Chinese
    - 架構：參考document，問題、回答
    - 優點：資料集內容完整
    - 缺點：太大了（200GB）好像比較難取捨，是簡體中文 更新：不需要去下載原始檔案
    - Thought: 如果我們只是要檢測retrieval資料檢索的能力的話，或許可以限縮範圍，把簡體中文翻譯成繁體中文並且進行測試
    - references: https://huggingface.co/datasets/yuyijiong/Multi-Doc-QA-Chinese
    - 非常多個documents, 至少1000個questions=answer


3. Wiki 資料集 ... (已刪除，沒有QA，除非要自己生標準問答) （只有資料及沒有問答）
搞定

## 其他問題
### next question: How many documents should we add?

GPT, Claude: 10000 is feasible
Perplexity: 1000 is minimum, 10000 is feasible
Thought: use 百科001-002 放正確文件和錯誤文件總共一萬篇, chunking之後大概會增加到30000個？ 
繁體中文測試專家ihower: 他只有把1000個檔案放進dataset裡面
建議：先嘗試比較簡單的繁體中文資料來源，如果表現的還不錯的話（還有沒有花到太多錢），再小規模測試簡體中文的資料集，才能夠得到gpt建議的10000個資料來源

## 評量方法 

本來想要做的事情是評論gpt的答案和真實答案的落差，但後來發現如果是ihower用的方式的話，他是把一整篇文章直接丟進去embedding裡面，這樣的話retrieval之後就不用做embedding了 250-950