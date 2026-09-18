from typing import List

from langchain_experimental.text_splitter import SemanticChunker
from langchain_unstructured import UnstructuredLoader
from unstructured.documents.ontology import Document

from model.embedding_models import embedding
from utils.log_utils import log


class PDFParser:
    def __int__(self):
        # 这里使用本地embedding模型
        self.splitter = SemanticChunker(
            embeddings=embedding,
            breakpoint_threshold_type='percentile'
        )

    def parse_pdf(self, file_path: str) -> List[Document]:
        loader = UnstructuredLoader(
            file_path=file_path,
            strategy='hi_res',
            partition_via_api=False,
            coordinates=True
        )

        docs = []
        for doc in loader.lazy_load():
            docs.append(doc)
        return docs

    def merge_docs(self, datas: List[Document]) -> List[Document]:
        pass
        return datas


    def chunk_text(self, datas: List[Document]) -> List[Document]:
        chunk_docs = []
        for data in datas:
            if len(data.page_content) > 300:
                chunk_docs.extend(self.splitter.split_documents([data]))
            else:
                chunk_docs.append(data)
        return chunk_docs

    def pdf2chunk(self, file_path: str) -> List[Document]:
        # 加载解析
        row_datas = self.parse_pdf(file_path)
        log.info(f'文件解析后docs长度：{len(row_datas)}')
        # 合并
        merged_datas = self.merge_docs(row_datas)
        log.info(f'文件合并后docs长度：{len(merged_datas)}')
        # 根据语义分割
        chunks = self.chunk_text(merged_datas)
        log.info(f'文件切片后docs长度：{len(chunks)}')
        return chunks
