"""
RAG 核心引擎
- 加载
- 切分
- 建库：支持按 embedding 模型自动生成库目录
- 检索：混合检索 + Cross-Encoder 重排
- 生成
- debug
"""
import os
import json
from dotenv import load_dotenv
import shutil

# 文档加载
from langchain_community.document_loaders import PyMuPDFLoader,TextLoader

# 文本切分
from langchain_text_splitters import RecursiveCharacterTextSplitter,MarkdownHeaderTextSplitter

# LLM
from langchain_deepseek import ChatDeepSeek

# 向量库
from langchain_chroma import Chroma
from langchain_community.embeddings import DashScopeEmbeddings

# 检索器
from langchain_community.retrievers import BM25Retriever
from langchain_classic.retrievers import EnsembleRetriever
from langchain_classic.retrievers import ContextualCompressionRetriever
from langchain_classic.retrievers.document_compressors import CrossEncoderReranker

# Cross-Encoder
from langchain_community.cross_encoders import HuggingFaceCrossEncoder

# 从向量库取回chunks后构造成bm25模型能够接收的文档
from langchain_core.documents import Document

#bge-m3（开源、免费）
from langchain_community.embeddings import HuggingFaceEmbeddings

load_dotenv()

# ============================================================
# 向量化模型工厂
# ============================================================
def get_embedding_fn(model_name):
    """
    根据模型名返回对应的 Embedding 函数
    支持的模型：
        - text-embedding-v4（百炼）
        - bge-m3（本地开源）
    """
    if model_name == "text-embedding-v4":
        return DashScopeEmbeddings(
            model="text-embedding-v4",
            dashscope_api_key=os.environ.get("DASHSCOPE_API_KEY")
        )
    elif model_name == "bge-m3":
        from langchain_community.embeddings import HuggingFaceEmbeddings
        return HuggingFaceEmbeddings(
            model_name="BAAI/bge-m3",
            model_kwargs={"device": "cpu"},
            encode_kwargs={"normalize_embeddings": True}
        )
    else:
        raise ValueError(f"不支持的 Embedding 模型: {model_name}")


# ============================================================
# RAGEngine
# ============================================================

class RAGEngine:
    def __init__(self,corpus_dir="corpus",persist_dir=None,embedding_fn=None):
        
        # 1. 基础属性（必须最先赋值）
        self.corpus_dir = corpus_dir
        self.persist_dir = persist_dir or "chroma_db"
        self.embedding_fn = embedding_fn or get_embedding_fn("text-embedding-v4")

        # 2. 初始化LLM
        self.llm = ChatDeepSeek(
            model="deepseek-flash",
            temperature=0.1,
            extra_body={"thinking":{"type":"disabled"}}
        )

        # 3. 检索器占位
        self.vector_store = None
        self.vector_retriever = None
        self.bm25_retriever = None
        self.ensemble_retriever = None
        self.reranker = None
        self.retrieval_chain = None

    # ============================================================
    # 1. 加载文档
    # ============================================================
    def load_documents(self):
        """加载corpus目录下的所有md,pdf文档"""
        docs = []
        for filename in os.listdir(self.corpus_dir):
            filepath = os.path.join(self.corpus_dir, filename)
            if filename.endswith(".md"):
                loader = TextLoader(filepath, encoding="utf-8")
            elif filename.endswith(".pdf"):
                loader = PyMuPDFLoader(filepath)
            else:
                continue
            docs.extend(loader.load())
        print(f"✅ 加载了 {len(docs)} 个文档")
        return docs

    # ============================================================
    # 2. 切分文档
    # ============================================================
    def split_documents(self,docs,chunk_size=500, chunk_overlap=50,
                        separators=None,use_markdown_splitter=True):
        """
        统一切分方法：
        - MD 文件：先按 Markdown 标题切，再按字符数细切
        - PDF 文件：直接按字符数切
        """

        if separators is None:
            separators = ["\n\n", "\n", "。", "！", "？", "，", " ", ""]
        
        headers_to_split_on = [("#", "h1"),("##", "h2"),("###", "h3"), ]
        md_splitter = MarkdownHeaderTextSplitter(headers_to_split_on=headers_to_split_on)
    

        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
            separators=separators,
            keep_separator=False
        )

        all_chunks = []
        for doc in docs:
            source = doc.metadata.get("source", "")

            if source.endswith(".pdf"):
                chunks = text_splitter.split_documents([doc])
                all_chunks.extend(chunks)
            elif source.endswith(".md") and use_markdown_splitter:    
                # 1. 先按标题切
                md_chunks = md_splitter.split_text(doc.page_content)

                for chunk in md_chunks:
                    # 2. 继承原文档的 source 等元数据
                    chunk.metadata.update(doc.metadata)
                    # 3. 如果块还超过 chunk_size，用字符切分器再切
                    if len(chunk.page_content) > 500:
                        sub_chunks = text_splitter.split_documents([chunk])
                        all_chunks.extend(sub_chunks)
                    else:
                        all_chunks.append(chunk)
            else:
                chunks = text_splitter.split_documents([doc])
                all_chunks.extend(chunks)
                
        print(f"切分后共{len(all_chunks)}个块")
        return all_chunks

    # ============================================================
    # 3. 向量库构建
    # ============================================================
    def vector_store_exists(self):
        return os.path.exists(self.persist_dir) and len(os.listdir(self.persist_dir)) > 0

    
    # 从零建库：清除旧数据，再向量化
    def build_index(self,chunks,force_rebuild=False):
        """
        建库
        force_rebuild=False 时，如果库已存在则跳过
        """

        if not force_rebuild and self.vector_store_exists():
            print(f"✅ 向量库已存在，跳过重建: {self.persist_dir}")
            self.load_existing_index()
            return


        # 每次重建前清空旧数据
        if os.path.exists(self.persist_dir):
            shutil.rmtree(self.persist_dir)

        #建向量库
        self.vector_store = Chroma.from_documents(
            documents=chunks,
            embedding=self.embedding_fn,
            persist_directory=self.persist_dir,
            collection_name="company_FAQ"
        )
        print(f"✅向量库重新构建已存入{self.vector_store._collection.count()}条文档")

        self._build_retrieval_chain(chunks)

    # 加载已有向量库
    def load_existing_index(self):
        self.vector_store = Chroma(
        collection_name="company_FAQ",
        embedding_function=self.embedding_fn,
        persist_directory=self.persist_dir
    )
        print(f"✅已加载向量库({self.vector_store._collection.count()} 条文档）")
        
        # 取回 chunks,以便于BM25检索器重构（BM25不和向量一样，无法从磁盘恢复）
        all_data = self.vector_store._collection.get(include=["documents", "metadatas"])
        chunks = [
            Document(page_content=text, metadata=meta)
            for text, meta in zip(all_data["documents"], all_data["metadatas"])
        ]
        
        self._build_retrieval_chain(chunks)   

    # ============================================================
    # 4. 构建检索链，这个方法被 build_index 和 load_existing_index 共用
    # ============================================================
    def _build_retrieval_chain(self,chunks,top_n=5):
        """
        构建：
        - 向量检索器
        - BM25 检索器
        - 混合检索器（EnsembleRetriever）
        - Cross-Encoder 重排器
        - 完整检索链（ContextualCompressionRetriever）
        """
        # 4.1 向量检索器
        # 召回 10 个候选，为重排提供充足素材
        self.vector_retriever = self.vector_store.as_retriever(
                    search_kwarg={"k":20}
        ) 

        # 4.2 BM25 检索器
        self.bm25_retriever = BM25Retriever.from_documents(chunks)
        self.bm25_retriever.k = 20

        # 4.3 混合检索器：向量+BM25，用RRF融合
        self.ensemble_retriever = EnsembleRetriever(
            retrievers=[self.vector_retriever,self.bm25_retriever],
            weights=[0.5,0.5]  #两路权重相同
        )
        print("✅ 混合检索器已构建（向量 + BM25 + RRF）")

        # 4.4 初始化 Cross-Encoder重排器，使用 BAAI/bge-reranker-v2-m3，它对中英文支持都很好
        cross_encoder = HuggingFaceCrossEncoder(model_name="BAAI/bge-reranker-v2-m3")

        #重排后设定最终送给大模型的文档数量
        self.reranker = CrossEncoderReranker(
            model=cross_encoder,
            top_n=top_n            )
        print("✅ Cross-Encoder 重排器已构建")

        # 4.5 串联检索链，ContextualCompressionRetriever是一个包装器，接受基础的ensemble_retriever（RRF融合结果），并将结果交给self.reranker(CrossEncoderReranker)进行重排，最终输出top_n个最相关文档
        self.retrieval_chain = ContextualCompressionRetriever(
            base_compressor=self.reranker,
            base_retriever=self.ensemble_retriever
        )
        print("✅ 完整检索链已构建（混合检索 → 重排）")

    # ============================================================
    # 5. 检索
    # ============================================================
    def retrieve(self,query,k=5):
        """
        执行完整检索链：混合召回 → Cross-Encoder 重排 → 返回 top-k
        返回：[Document, Document, Document]
        """
        if self.retrieval_chain is None:
            raise RuntimeError("请先调用 build_index() 构建索引")

        docs = self.retrieval_chain.invoke(query)
        return docs[:k]

    # ============================================================
    # 6. 生成回答
    # ============================================================
    def generate(self,query,context):
        """基于上下文生成回答"""
        prompt = f"""请根据一下资料回答问题。

规则：
1.只使用资料中的信息，不要编造。
2.如果资料中没有相关信息，直接说“资料中没有提到”。
3.回答末尾必须标注来源，格式：[来源：文档名 章节名]

资料：
{context}

问题：{query}

回答：
"""
        response = self.llm.invoke(prompt)
        return response.content

    # ============================================================
    # 7. 完整查询流程
    # ============================================================
    def query(self,question):
        """
        完整 RAG 流程：
        1. 混合检索 + 重排 → top-3 文档
        2. 拼接上下文（带来源）
        3. 调用 LLM 生成回答
        """
        # 1. 检索+重排
        docs = self.retrieve(question)

        if not docs:
            return "没有找到相关文档"

        # 2. 拼接上下文
        context_parts = []
        for doc in docs:
            source = doc.metadata.get("source","未知")
            # 只取文件名，不显示完整路径
            source_name = os.path.basename(source)

            h1 = doc.metadata.get("h1")
            h2 = doc.metadata.get("h2")
            h3 = doc.metadata.get("h3")

             # 拼接章节路径
            if h3:
                location = f"{h1} → {h2} → {h3}"
            elif h2:
                location = f"{h1} → {h2}"
            elif h1:
                location = h1
            else:
                location = ""

            if location:
                context_parts.append(f"[来源：{source_name} {location}]\n{doc.page_content}")
            else:
                context_parts.append(f"[来源：{source_name}]\n{doc.page_content}")

        context = "\n\n".join(context_parts)

        # 3. 生成
        answer = self.generate(question,context)
        return answer


    # ============================================================
    #  调试：查看检索详情
    # ============================================================
    def debug_retrieve(self, query):
        """调试用：分别查看混合检索和重排的结果"""
        print("=" * 60)
        print(f"查询：{query}")
        print("=" * 60)

        # 混合检索结果
        print("\n[混合检索召回]")
        candidates = self.ensemble_retriever.invoke(query)
        for i, doc in enumerate(candidates):
            source = os.path.basename(doc.metadata.get("source", "未知"))
            print(f"  {i+1}. [{source}] {doc.page_content[:60]}...")

        # 重排后结果
        print("\n[Cross-Encoder 重排后]")
        reranked = self.retrieval_chain.invoke(query)
        for i, doc in enumerate(reranked):
            source = os.path.basename(doc.metadata.get("source", "未知"))
            print(f"  {i+1}. [{source}] {doc.page_content[:60]}...")

