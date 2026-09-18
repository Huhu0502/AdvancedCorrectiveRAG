from typing import List

from langchain_community.document_loaders import UnstructuredMarkdownLoader
from langchain_experimental.text_splitter import SemanticChunker
from unstructured.documents.ontology import Document

from model.embedding_models import embedding
from utils.log_utils import log


class MarkDownParser:

    # splitter 都约定俗成地只处理 Document.page_content 字段。
    def __init__(self):
        # 使用本地embedding模型，
        self.text_splitter = SemanticChunker(
            embeddings=embedding,
            breakpoint_threshold_type='percentile'
        )

    def text_chunker(self, datas: List[Document]) -> List[Document]:
        new_docs = []
        for data in datas:
            if len(data.page_content) > 300:
                new_docs.extend(self.text_splitter.split_documents([data]))
            else:
                new_docs.append(data)

        return new_docs

    def parse_markdown_to_documents(self, md_file) -> List[Document]:
        #  加载解析
        documents = self.parse_markdown(md_file)
        #  解析的粒度太小，Unstructured 的设计哲学是“原子化语义单元”，
        #  而 RAG 分块需要的是“完整语义上下文”
        #  所以需要合并
        merge_documents = self.merge_by_title(documents)
        log.info(f'文件合并后docs长度：{len(merge_documents)}')
        # 合并长度可能过长，需要切割
        chunk_docs = self.text_chunker(merge_documents)
        log.info(f'语义切割后的chunk长度：{len(chunk_docs)}')
        return chunk_docs

    def parse_markdown(self, md_file) -> List[Document]:
        # unstructured loader将加载和解析合并成一步，生成结构化元素Document
        loader = UnstructuredMarkdownLoader(
            file_path=md_file,
            mode='elements'
        )

        docs = []
        for doc in loader.lazy_load():
            docs.append(doc)
        log.info(f'文件解析后docs长度：{len(docs)}')
        return docs

    def merge_by_title(self, datas: List[Document]) -> List[Document]:
        merged_data = []
        parent_dict = {}
        for document in datas:
            metadata = document.metadata
            parent_id = metadata.get('parent_id', None)
            category = metadata.get('category', None)
            element_id = metadata.get('element_id', None)

            if 'languages' in metadata:
                metadata.pop('languages')

            if category == 'NarrativeText' and parent_id is None:
                merged_data.append(document)
            if category == 'Title':
                if parent_id in parent_dict:
                    document.page_content = parent_dict[parent_id].page_content + '->' + document.page_content
                parent_dict[element_id] = document
            if category != 'Title' and parent_id:
                parent_dict[parent_id].page_content = parent_dict[
                                                          parent_id].page_content + '******' + document.page_content
                parent_dict[parent_id].metadata['category'] = 'content'
        if parent_dict is not None:
            merged_data.extend(parent_dict.values())

        return merged_data


if __name__ == '__main__':
    parser = MarkDownParser()
    docs = parser.parse_markdown_to_documents('../data/md/tech_report_0tfhhamx.md')

    for doc in docs:
        print(f'元数据：{doc.metadata}')
        print(f'标题：{doc.metadata.get("title", None)}')
        print(f'内容：{doc.page_content}')
        print("---" * 10)
