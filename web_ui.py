import streamlit as st
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_community.llms import Ollama

# 1. 设置网页的标题和图标
st.set_page_config(page_title="私人游戏主策划", page_icon="🃏")
st.title("🃏 我的私人卡牌游戏主策划")
st.caption("Powered by Gemma3 12B & ChromaDB 向量检索")


# 2. 唤醒图书管理员和数据库 (加入缓存装饰器，防止网页每次点击都重新连数据库)
@st.cache_resource
def load_database():
    embeddings = OllamaEmbeddings(model="nomic-embed-text")
    vectorstore = Chroma(persist_directory="./chroma_db", embedding_function=embeddings)
    return vectorstore.as_retriever(search_kwargs={"k": 1})


retriever = load_database()


# 3. 唤醒大模型引擎 (同样加入缓存)
@st.cache_resource
def load_llm():
    return Ollama(model="gemma3:12b")


llm = load_llm()

# 4. 初始化网页的“记忆缓存”
if "messages" not in st.session_state:
    st.session_state.messages = []

# 把历史聊天记录画在网页上
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# 5. 网页底部的输入框
user_input = st.chat_input("请输入你想推演的游戏规则问题...")

if user_input:
    # 在网页上显示你的问题
    with st.chat_message("user"):
        st.markdown(user_input)

    # --- 【核心：拼装 RAG 所需的组件】 ---
    # 第一步：让图书管理员去检索规则
    relevant_docs = retriever.invoke(user_input)
    retrieved_rules = relevant_docs[0].page_content if relevant_docs else "未找到相关设定"

    # 第二步：整理过去 3 轮的历史记忆 (防止聊天太久撑爆显存)
    history_text = ""
    recent_messages = st.session_state.messages[-6:]  # 1轮包含一问一答，6条就是3轮
    for msg in recent_messages:
        role_name = "你问" if msg["role"] == "user" else "AI答"
        history_text += f"{role_name}：{msg['content']}\n\n"

    # 第三步：组装终极魔法阵
    prompt = f"""
    你现在是我的私人游戏规则主策划。

    【图书管理员刚刚从数据库为你翻出的精准规则】：
    {retrieved_rules}

    【我们最近的聊天记录】：
    {history_text}

    【玩家最新的问题】：{user_input}

    请你结合上面的【精准规则】和【聊天记录】回答玩家。如果规则里没写，请说明“暂无明确规定”，然后运用你的游戏策划专业知识给出逻辑推演。
    """

    # 记录用户的输入到网页缓存
    st.session_state.messages.append({"role": "user", "content": user_input})

    # 呼叫 AI 思考并输出
    with st.chat_message("assistant"):
        with st.spinner("主策划正在翻阅数据库并疯狂思考中..."):
            response = llm.invoke(prompt)
            st.markdown(response)

    # 记录 AI 的回答到网页缓存
    st.session_state.messages.append({"role": "assistant", "content": response})