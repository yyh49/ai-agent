import os
import streamlit as st

# ✅ 第一行 Streamlit
st.set_page_config(page_title="智能体聊天助手 🤖")

from agent_exprt import agent, build_vector_db, UPLOAD_DIR

st.title("🤖 PDF + 搜索 + 天气 智能体")

# ===== 上传 PDF =====
uploaded_files = st.file_uploader(
    "上传PDF文件", type=["pdf"], accept_multiple_files=True
)

if uploaded_files:
    for file in uploaded_files:
        save_path = os.path.join(UPLOAD_DIR, file.name)
        with open(save_path, "wb") as f:
            f.write(file.read())
    st.success("PDF 上传成功")

if st.button("构建 / 更新知识库"):
    with st.spinner("正在建立向量索引..."):
        build_vector_db()
    st.success("知识库构建完成 ✅")

st.divider()

# ===== 聊天区 =====
if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    st.chat_message(msg["role"]).markdown(msg["content"])

prompt = st.chat_input("请输入你的问题...")

if prompt:
    st.chat_message("user").markdown(prompt)
    st.session_state.messages.append({"role": "user", "content": prompt})

    with st.chat_message("assistant"):
        with st.spinner("AI 思考中..."):
            result = agent.run(prompt)
            st.markdown(result)

    st.session_state.messages.append({"role": "assistant", "content": result})
