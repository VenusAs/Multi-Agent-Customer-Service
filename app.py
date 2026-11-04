import streamlit as st
import os
from langchain_core.messages import HumanMessage, AIMessage
from langgraph.types import Command

# 从你的 Agent 脚本中导入已编译好的 graph
# 假设你把之前的 Mulit-Agent.py 重命名为了 agent_graph.py
from agent_graph import graph 

# ============================================================
# 页面配置
# ============================================================
st.set_page_config(
    page_title="多 Agent 智能客服系统",
    page_icon="🤖",
    layout="wide"
)

st.title("🤖 多 Agent 智能客服系统")
st.caption("基于 LangGraph 路由分发、RAG 检索、工具调用与 HITL 人工审批")

# ============================================================
# 初始化状态 (Session State)
# ============================================================
if "messages" not in st.session_state:
    st.session_state.messages = []

if "thread_id" not in st.session_state:
    st.session_state.thread_id = "streamlit_user_001"

# 用于存储 LangGraph 返回的中断信息
if "interrupt_info" not in st.session_state:
    st.session_state.interrupt_info = None

# 统一的 config 配置
config = {"configurable": {"thread_id": st.session_state.thread_id}}

# ============================================================
# 侧边栏控制
# ============================================================
with st.sidebar:
    st.header("⚙️ 系统控制")
    st.info(f"当前会话 ID: {st.session_state.thread_id}")
    
    if st.button("🗑️ 清空对话", use_container_width=True):
        st.session_state.messages = []
        st.session_state.interrupt_info = None
        st.session_state.thread_id = f"streamlit_user_{os.urandom(2).hex()}"
        st.rerun()

# ============================================================
# 渲染历史消息
# ============================================================
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# ============================================================
# 核心逻辑：处理人工审批 (HITL)
# ============================================================
if st.session_state.interrupt_info:
    info = st.session_state.interrupt_info
    
    # 在聊天界面中插入一个审批卡片
    with st.chat_message("assistant", avatar="🚨"):
        st.warning(f"**人工审批拦截**：检测到高危操作！")
        st.markdown(f"- **操作类型**: `{info.get('action')}`")
        st.markdown(f"- **订单编号**: `{info.get('order_id')}`")
        st.markdown(f"- **提示信息**: {info.get('message')}")
        
        col1, col2 = st.columns(2)
        with col1:
            if st.button("✅ 批准执行", type="primary", use_container_width=True):
                # 恢复执行
                res = graph.invoke(Command(resume=True), config)
                final_answer = res['messages'][-1].content
                
                # 更新历史
                st.session_state.messages.append({"role": "assistant", "content": f"【已批准】{final_answer}"})
                st.session_state.interrupt_info = None
                st.rerun()
                
        with col2:
            if st.button("❌ 拒绝执行", type="secondary", use_container_width=True):
                # 恢复执行（传入 False）
                res = graph.invoke(Command(resume=False), config)
                final_answer = res['messages'][-1].content
                
                # 更新历史
                st.session_state.messages.append({"role": "assistant", "content": f"【已拒绝】{final_answer}"})
                st.session_state.interrupt_info = None
                st.rerun()

# ============================================================
# 输入框与 Agent 执行
# ============================================================
question = st.chat_input("请输入您的问题（如：员工年假怎么算？或者 帮我把订单12345退款）...")

if question and not st.session_state.interrupt_info:
    # 1. 显示用户输入
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    # 2. 调用 LangGraph Agent
    with st.spinner("多 Agent 正在协同处理..."):
        try:
            # 执行图
            res = graph.invoke({"messages": [HumanMessage(content=question)]}, config)
            
            # 检查是否触发了中断 (HITL)
            if "__interrupt__" in res:
                # 提取中断信息存入 state
                st.session_state.interrupt_info = res["__interrupt__"][0].value
                st.rerun() # 刷新页面，渲染审批卡片
                
            else:
                # 正常完成，提取最后一条 AI 回复
                final_answer = res["messages"][-1].content
                st.session_state.messages.append({"role": "assistant", "content": final_answer})
                st.rerun()
                
        except Exception as e:
            st.error(f"❌ Agent 运行出错: {str(e)}")