from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_community.llms import Ollama

print("1. 正在唤醒图书管理员 (连接本地数据库)...")
embeddings = OllamaEmbeddings(model="nomic-embed-text")
# 连接到你刚才建好的 chroma_db 数据库
vectorstore = Chroma(persist_directory="./chroma_db", embedding_function=embeddings)

# 设置检索器：每次只提取最相关的 1 个规则块（因为我们现在的规则还很短）
retriever = vectorstore.as_retriever(search_kwargs={"k": 1})

print("2. 正在唤醒主策划大脑 (Gemma 3 12B)...")
llm = Ollama(model="gemma3:12b")

# AI 的海马体记忆库
chat_history = []

print("\n👑 终极形态 AI 已就绪！带有【长线数据库检索】与【短线历史记忆】。")
print("================================================================")

while True:
    user_input = input("造物主（你）：")
    if user_input == "退出":
        print("AI 已退下！赶紧去跑你的 5 公里吧，跑完记得补维 C！")
        break

    # 【核心联动 1】：让图书管理员去数据库里精准检索！
    relevant_docs = retriever.invoke(user_input)
    # 提取查到的文字内容
    retrieved_rules = relevant_docs[0].page_content if relevant_docs else "未找到相关设定"

    # 整理历史记忆
    history_text = ""
    for turn in chat_history:
        history_text += f"你问：{turn['user']}\nAI答：{turn['ai']}\n\n"

    # 【核心联动 2】：组装终极魔法阵（提示词）
    prompt = f"""
    你现在是我的私人游戏规则主策划。

    【图书管理员刚刚从数据库为你翻出的精准规则】：
    {retrieved_rules}

    【我们之前的聊天记录】：
    {history_text}

    【玩家最新的问题】：{user_input}

    请你结合上面的【精准规则】和【聊天记录】回答玩家。如果规则里没写，请说明“暂无明确规定”，然后运用你的游戏策划专业知识给出逻辑推演。
    """

    response = llm.invoke(prompt)
    print(f"\n专属 AI：{response}\n")

    # 更新记忆
    chat_history.append({"user": user_input, "ai": response})
    if len(chat_history) > 3:
        chat_history.pop(0)