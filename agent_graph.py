import os
import operator
import requests
import re
from typing import Annotated, TypedDict, Literal
from dotenv import load_dotenv

from langchain_core.tools import tool
from langchain_core.messages import AnyMessage, HumanMessage, SystemMessage, AIMessage, ToolMessage
from langchain_deepseek import ChatDeepSeek
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.prebuilt import ToolNode, tools_condition

from langgraph.types import interrupt, Command

# 引入你的 RAG 引擎（请确保 rag_engine.py 在同一目录下）
from rag_engine import RAGEngine, get_embedding_fn

load_dotenv()

# ============================================================
# 0. 全局初始化 RAG 引擎（一定要在外部初始化，避免重复加载）
# ============================================================
print("正在加载知识库...")
rag_engine = RAGEngine(
    corpus_dir="corpus",
    persist_dir="chroma_db_v4",
    embedding_fn=get_embedding_fn("text-embedding-v4")
)
if rag_engine.vector_store_exists():
    rag_engine.load_existing_index()
else:
    docs = rag_engine.load_documents()
    chunks = rag_engine.split_documents(docs)
    rag_engine.build_index(chunks)
print("知识库加载完成！")

# ============================================================
# 1. 定义工具
# ============================================================
@tool
def search_knowledge_base(query: str) -> str:
    """查询公司规章制度、员工手册、IT支持、产品说明等私有知识库。"""
    try:
        return rag_engine.query(query)
    except Exception as e:
        return f"知识库检索失败: {str(e)}"

@tool
def calculator(expression: str) -> str:
    """计算数学表达式，例如 '128*7'。"""
    try:
        expression = expression.replace("（", "(").replace("）", ")")
        allowed_chars = set("0123456789+-*/(). ")
        if not set(expression).issubset(allowed_chars):
            return "错误：表达式包含非法字符"
        return str(eval(expression, {"__builtins__": {}}))
    except Exception as e:
        return f"计算错误: {str(e)}"

@tool
def refund_order(order_id: str) -> str:
    """【高危操作】对指定订单号执行退款。调用此工具需要人工审批。"""
    # 这是一个模拟的副作用操作
    return f"订单 {order_id} 已成功退款。"

# ============================================================
# 2. 定义全局状态 (Multi-Agent State)
# ============================================================
class AgentState(TypedDict):
    # 消息历史（累积）
    messages: Annotated[list[AnyMessage], operator.add]
    # 记录当前由哪个 Agent 处理，用于路由
    next_agent: str

# ============================================================
# 3. 初始化大模型
# ============================================================
model = ChatDeepSeek(
    model="deepseek-flash",
    temperature=0,
    extra_body={"thinking": {"type": "disabled"}}
)

# 绑定工具给专门的 Tool Agent 使用
model_with_tools = model.bind_tools([calculator, refund_order])

# ============================================================
# 4. 定义各个节点 (Agents)
# ============================================================

# ---- 4.1 Router 节点 ----
def router_node(state: AgentState):
    """分析用户意图，决定交给哪个 Agent"""
    # 获取用户最新的提问
    last_user_msg = [m for m in state["messages"] if isinstance(m, HumanMessage)][-1].content
    
    prompt = f"""你是一个客服分发中心。请判断用户的意图，只输出一个英文单词：
- 如果问公司制度、产品、部署流程、年假等内部知识，输出：rag
- 如果要求【退款】、【退货】、【算数学题】，输出：tool
- 如果是打招呼或无法判断，输出：end

用户问题：{last_user_msg}
"""
    response = model.invoke([HumanMessage(content=prompt)]).content.strip().lower()
    
    # 容错处理
    if "rag" in response: next_agent = "rag_agent"
    elif "tool" in response: next_agent = "tool_agent"
    else: next_agent = "end"
    
    print(f"🔀 [Router] 判定意图为: {next_agent}")
    return {"next_agent": next_agent}

# ---- 4.2 RAG 专家 Agent ----
def rag_agent_node(state: AgentState):
    """专门负责查知识库并回答"""
    print("📚 [RAG Agent] 正在检索知识库...")
    last_user_msg = state["messages"][-1].content
    
    # 调用工具
    answer = search_knowledge_base.invoke({"query": last_user_msg})
    
    # 将回答包装成 AIMessage 返回
    return {"messages": [AIMessage(content=answer)], "next_agent": "end"}

# ---- 4.3 工具专家 Agent (支持绑定工具) ----
def tool_agent_node(state: AgentState):
    """专门负责处理工具调用（算数、退款）"""

    # 1. 检查最后一条消息是不是工具返回的结果
    # 如果是，说明工具已经执行完了，我们现在只需要总结给用户，然后结束
    last_message = state["messages"][-1]
    if isinstance(last_message, ToolMessage):
        print("✅ [Tool Agent] 工具已执行完毕，正在生成最终总结...")
        # 直接让大模型根据历史生成最终回复，不再绑定 tools
        sys_msg = SystemMessage(content="你是一个客服助手，请根据工具执行的结果给用户一个友好的回复。")
        final_response = model.invoke([sys_msg] + state["messages"])
        return {"messages": [final_response], "next_agent": "end"}

    # 2. 如果不是工具结果，说明是用户的新提问，需要大模型决定调不调工具
    
    sys_msg = SystemMessage(content="""你是一个操作执行助手。
    如果用户提到了退款，你必须提取订单号，并且【强制】调用 `refund_order` 工具。
    不要说废话，不要问用户是否确认，直接调用工具（后续流程有人工拦截）。'
    """)
    response = model_with_tools.invoke([sys_msg] + state["messages"])
    
    # 👈 核心修改：如果模型决定调用高危工具，就在这里主动中断！
    if response.tool_calls:
        # 检查是不是高危工具
        for tc in response.tool_calls:
            if tc["name"] == "refund_order":
                # 触发中断，把操作详情传出去
                user_approval = interrupt({
                    "action": "refund_order",
                    "order_id": tc["args"].get("order_id"),
                    "message": "高危操作：请求退款，请人工审批！"
                })
                
                # 如果人工拒绝（返回 False 或 "n"）
                if user_approval not in [True, "y", "Y", "yes"]:
                    return {"messages": [AIMessage(content="用户拒绝了此次退款操作。")], "next_agent": "end"}
                # 如果人工同意，就继续往下走执行工具

        return {"messages": [response], "next_agent": "tools"}
    else:
        return {"messages": [response], "next_agent": "end"}

# ---- 4.4 结束节点（兜底闲聊） ----
def end_node(state: AgentState):
    last_msg = state["messages"][-1].content
    response = model.invoke([SystemMessage(content="你是一个友好的客服助手。"), HumanMessage(content=last_msg)])
    return {"messages": [response]}

# ============================================================
# 5. 构建图结构
# ============================================================
builder = StateGraph(AgentState)

# 添加所有节点
builder.add_node("router", router_node)
builder.add_node("rag_agent", rag_agent_node)
builder.add_node("tool_agent", tool_agent_node)
builder.add_node("tools", ToolNode([calculator, refund_order])) # 官方工具执行节点
builder.add_node("end", end_node)

# ---- 5.1 路由连边 ----
builder.add_edge(START, "router")

# Router 的路由逻辑
def router_condition(state: AgentState) -> Literal["rag_agent", "tool_agent", "end"]:
    return state["next_agent"]

builder.add_conditional_edges(
    "router",
    router_condition
)

# RAG Agent 干完活直接结束
builder.add_edge("rag_agent", END)
# 兜底结束
builder.add_edge("end", END)

# Tool Agent 的路由逻辑
def tool_condition(state: AgentState) -> Literal["tools", "end"]:
    return state["next_agent"]

builder.add_conditional_edges(
    "tool_agent",
    tool_condition
)

# 工具执行完毕后，回到 Tool Agent 总结结果
builder.add_edge("tools", "tool_agent")

# ============================================================
# 6. 编译图并加入 HITL (Human-in-the-loop)
# ============================================================
# 核心：在调用高危工具时强制暂停
checkpointer = InMemorySaver()
graph = builder.compile(checkpointer=checkpointer)

# ============================================================
# 7. 测试运行
# ============================================================
if __name__ == "__main__":
    pass
    # config = {"configurable": {"thread_id": "user-001"}}
    
    # # 测试 1：触发 RAG 专家（直接走向 END）
    # print("\n" + "="*50)
    # print("用户提问：员工入职满3年有几天年假？")
    # res1 = graph.invoke({"messages": [HumanMessage(content="员工入职满3年有几天年假？")]}, config)
    # print(f"👉 最终回复: {res1['messages'][-1].content}")

    # # 测试 2：触发 Tool 专家，并测试 HITL（高危操作拦截）
    # print("\n" + "="*50)
    # print("用户提问：帮我把订单 12345 退款。")
    
    # # 2.1 第一次执行
    # res = graph.invoke({"messages": [HumanMessage(content="帮我把订单 12345 退款。")]}, config)
    
    # # 👈 检查是否触发了 interrupt
    # if "__interrupt__" in res:
    #     interrupt_info = res["__interrupt__"][0].value
    #     print(f"\n🛑 图已暂停！拦截到高危操作: {interrupt_info['action']} 订单号: {interrupt_info['order_id']}")
    
        
    #      # 2.2 模拟人工审批
    #     approval = input("👉 是否批准执行该操作？(y/n): ")
        
    #     if approval.lower() == 'y':
    #         print("✅ 已批准，继续执行...")
    #         # 使用 Command 恢复执行，并传入 True
    #         res2 = graph.invoke(Command(resume=True), config)
    #         print(f"👉 最终回复: {res2['messages'][-1].content}")
    #     else:
    #         print("❌ 已拒绝执行。")
    #         # 使用 Command 恢复执行，并传入 False
    #         res2 = graph.invoke(Command(resume=False), config)
    #         print(f"👉 最终回复: {res2['messages'][-1].content}")
    # else:
    #     # 如果不是退款，正常结束
    #     print(f"👉 最终回复: {res[-1]['messages'].content}")