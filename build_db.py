from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.vectorstores import Chroma

print("1. 正在读取你的私人游戏设定...")
loader = TextLoader("game_rules.txt", encoding="utf-8")
docs = loader.load()

print("2. 图书管理员正在把规则切成小块，方便管理...")
# 把长文档切碎，每块大约 200 个字，块与块之间保留 20 个字的重叠（防止一句话被从中间劈开）
text_splitter = RecursiveCharacterTextSplitter(chunk_size=200, chunk_overlap=20)
splits = text_splitter.split_documents(docs)

print("3. 正在召唤图书管理员 (nomic-embed-text) 将文字转化为数学向量并存入数据库...")
embeddings = OllamaEmbeddings(model="nomic-embed-text")

# 这行代码极其关键：它会在你当前的文件夹下，真正创建一个名为 "chroma_db" 的数据库文件夹！
vectorstore = Chroma.from_documents(documents=splits, embedding=embeddings, persist_directory="./chroma_db")

print("4. 数据库建立成功！我们来测试一下图书管理员的精准度：")
# 模拟一次玩家提问，看看图书管理员能不能把相关的规则原话抽出来
question = "如果距离不够，我的杀该怎么处理？"
retriever = vectorstore.as_retriever()
relevant_docs = retriever.invoke(question)

print("="*40)
print("图书管理员为你翻到的原始规则是：\n")
print(relevant_docs[0].page_content)