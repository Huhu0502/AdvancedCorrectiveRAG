# 多进程写入海量数据
import multiprocessing
from multiprocessing import Queue
from pathlib import Path
from typing import List


from parser.md_parser import MarkDownParser
from parser.pdf_parser import PDFParser
from utils.env_utils import COLLECTION_NAME
from utils.log_utils import log
from utils.milvus_utils import OperateMilvus


def get_file_path(dir_path: str) -> list[str]:
    data_dir = Path(dir_path)
    file_paths: List[str] = []

    for file_path in data_dir.rglob('*'):
        if not file_path.is_file():
            continue

        path_str = str(file_path)
        if path_str.endswith('.md') or path_str.endswith('.pdf'):
            file_paths.append(path_str)

    return file_paths



def parse_datas(queue: Queue, batch_size: int = 20):
    # 解析文件 写队列
    file_paths = get_file_path('../data')
    md_parser = MarkDownParser()
    pdf_parser = PDFParser()

    docs_batch = []
    for path in file_paths:
        try:
            if path.endswith('.md'):
                docs_batch.extend(md_parser.parse_markdown_to_documents(path))
            else:
                docs_batch.extend(pdf_parser.pdf2chunk(path))
            log.info(f'Writer 向队列插入{path}文件的切片')
            if len(docs_batch) >= batch_size:
                # Queue.put(obj) 的语义是：将 obj 作为一个不可分割的整体（原子单元）放入队列。
                # 浅拷贝创建了新列表，持有相同的 Document 引用，clear() 清的是旧列表，新列表的引用完好无损。深拷贝有新引用指向新Document
                queue.put(docs_batch.copy())
                docs_batch.clear()
        except Exception as e:
            log.error(f'文件{path}解析失败: {str(e)}')
            log.exception(e)

    if docs_batch:
        queue.put(docs_batch)

    queue.put(None)
    log.info(f"解析完成，共处理{len(file_paths)}个文件")


def write_milvus(queue: Queue):
    # 读队列 写milvus
    tool = OperateMilvus()
    tool.create_connection()
    total_count = 0
    while True:
        try:
            datas = queue.get()
            if isinstance(datas, list):
                tool.insert_datas(datas)
                total_count += len(datas)
                log.info(f"累计已写入: {total_count} 个文档")
            if datas is None:
                break
        except Exception as e:
            log.error(f'写入数据失败 ！')
            log.exception(e)

    log.info(f"写入进程结束，总计写入 {total_count} 个文档")


# 将../data下的数据批量写入向量库
if __name__ == '__main__':
    queue = Queue(maxsize=20)
    parser_proc = multiprocessing.Process(
        target=parse_datas,
        args=(queue,)
    )
    writer_proc = multiprocessing.Process(
        target=write_milvus,
        args=(queue,)
    )

    parser_proc.start()
    writer_proc.start()

    parser_proc.join()
    writer_proc.join()

    print("系统提示：所有任务完成")