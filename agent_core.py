from duckduckgo_search import DDGS
from langchain.agents import initialize_agent, AgentType
from langchain_openai import ChatOpenAI
from langchain.tools import Tool
from langchain.tools import WikipediaQueryRun
from langchain.utilities import WikipediaAPIWrapper
from langchain.prompts import SystemMessagePromptTemplate
from langchain.memory import ConversationBufferMemory
import requests
import os
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
# os.environ["HTTPS_PROXY"] = "http://127.0.0.1:7890"
# os.environ["HTTP_PROXY"]  = "http://127.0.0.1:7890"


AMAP_KEY = "ece164c7ec381626ae23389edd671b87"
SERPER_API_KEY = "88815065e318ada81ea2598ced19a6dc1a274429"

# ====== 路径 ======
UPLOAD_DIR = "uploads"
DB_DIR = "vector_db"
os.makedirs(UPLOAD_DIR, exist_ok=True)

def heweather(city: str) -> str:
    url = "https://restapi.amap.com/v3/weather/weatherInfo"
    params = {
        "key": AMAP_KEY,
        "city": city,
        "extensions": "base",
        "output": "JSON"
    }

    r = requests.get(url, params=params, timeout=5)
    data = r.json()

    if data.get("status") != "1":
        return f"未查询到 {city} 的天气信息"

    info = data["lives"][0]
    weather = info["weather"]
    temp = info["temperature"]
    wind = info["winddirection"]
    humidity = info["humidity"]

    return (
        f"{city} 当前天气：{weather}，温度 {temp}°C，"
        f"风向 {wind}，湿度 {humidity}%"
    )


def web_search(query: str) -> str:
    results = []
    with DDGS() as ddgs:
        for r in ddgs.text(query, max_results=5):
            results.append(f"{r['title']}：{r['body']}")
    if not results:
        return "未搜索到相关内容"
    return "\n".join(results)


def serper_search(query: str) -> str:
    url = "https://google.serper.dev/search"
    headers = {
        "X-API-KEY": SERPER_API_KEY,
        "Content-Type": "application/json"
    }
    payload = {
        "q": query,
        "hl": "zh-cn",
        "num": 5
    }

    try:
        r = requests.post(url, headers=headers, json=payload, timeout=10)
        data = r.json()
    except Exception as e:
        return f"搜索请求失败: {str(e)}"

    if "organic" not in data:
        return "未搜索到相关结果"

    results = []
    for item in data["organic"][:5]:
        title = item.get("title", "")
        snippet = item.get("snippet", "")
        results.append(f"{title}：{snippet}")

    return "\n".join(results)

# ====== 向量库构建 ======

# from langchain_community.embeddings import HuggingFaceEmbeddings
#
# embeddings = HuggingFaceEmbeddings(
#     model_name="BAAI/bge-small-zh-v1.5"
# )


def build_vector_db():
    docs = []
    for file in os.listdir(UPLOAD_DIR):
        if file.endswith(".pdf"):
            loader = PyPDFLoader(os.path.join(UPLOAD_DIR, file))
            docs.extend(loader.load())

    if not docs:
        return None

    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    split_docs = splitter.split_documents(docs)

    db = FAISS.from_documents(split_docs, OpenAIEmbeddings())
    # db = FAISS.from_documents(split_docs, embeddings)

    db.save_local(DB_DIR)
    return db

def load_db():
    if os.path.exists(DB_DIR):
        return FAISS.load_local(DB_DIR, OpenAIEmbeddings())
    return None

# ====== 知识库 Tool ======
def knowledge_search(query: str) -> str:
    db = load_db()
    if not db:
        return "本地知识库为空，请先上传PDF"
    docs = db.similarity_search(query, k=3)
    return "\n".join([d.page_content for d in docs])

# ===== LLM =====
llm = ChatOpenAI(
    model="gpt-4o-mini",
    temperature=0
)

# ===== Wikipedia Tool =====
wiki = WikipediaQueryRun(api_wrapper=WikipediaAPIWrapper())

# ===== Tools 列表 =====
tools = [
    Tool(
        name="Weather",
        func=heweather,
        description="查询城市天气（输入城市中文名，例如：上海、北京、长沙）"
    ),
    Tool(
        name="Wikipedia",
        func=wiki.run,
        description="查询百科知识"
    ),
    Tool(
        name="Search",
        func=serper_search,
        description="联网搜索最新信息"
    )
]

agent_kwargs = {
    "system_message": SystemMessagePromptTemplate.from_template(
        """
    你是中文智能助手。
    规则：
    1. 查最新网络信息 → Search
    2. 查百科 → Wikipedia
    3. 查实时天气 → Weather
    4. 普通聊天直接回答
    所有回答必须中文。
    """
    )
}

memory = ConversationBufferMemory(
    memory_key="chat_history",
    return_messages=True
)


agent = initialize_agent(
    tools=tools,
    llm=llm,
    agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    verbose=True,
    agent_kwargs=agent_kwargs,
    memory=memory  # ✅ 这一行是对话记忆
)