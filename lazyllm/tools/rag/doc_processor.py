from fastapi import HTTPException
from pydantic import BaseModel, Field
from typing import Dict, List, Optional, Any
from sqlalchemy import create_engine, Column, JSON, String, TIMESTAMP, Table, MetaData, inspect, delete, text
from sqlalchemy.dialects.mysql import insert as mysql_insert
from sqlalchemy.engine import Engine
from lazyllm import LOG, ModuleBase, ServerModule, UrlModule, FastapiApp as app, ThreadPoolExecutor, config

from .store import LAZY_ROOT_NAME, LAZY_IMAGE_GROUP
from .store.store_base import DEFAULT_KB_ID
from .store.document_store import _DocumentStore
from .store.utils import fibonacci_backoff, create_file_path
from .transform import (AdaptiveTransform, make_transform,)
from .readers import ReaderBase
from .doc_node import DocNode
from .utils import gen_docid, ensure_call_endpoint, BaseResponse
from .global_metadata import RAG_DOC_ID, RAG_DOC_PATH, RAG_KB_ID

import queue
import threading
import time
import requests
import uuid
import os
import traceback
import random
import signal
import subprocess

DB_TYPES = ['mysql']
ENABLE_DB = os.getenv('RAG_ENABLE_DB', 'false').lower() == 'true'
USE_TRANSFORMED_FILE = os.getenv('RAG_USE_TRANSFORMED_FILE', 'false').lower() == 'true'


class _Processor:
    def __init__(self, store: _DocumentStore, reader: ReaderBase, node_groups: Dict[str, Dict],
                 display_name: Optional[str] = None, description: Optional[str] = None,
                 server: bool = False):
        self._store = store
        self._reader = reader
        self._node_groups = node_groups
        self._display_name = display_name
        self._description = description

    def add_doc(self, input_files: List[str], ids: Optional[List[str]] = None,  # noqa C901
                metadatas: Optional[List[Dict[str, Any]]] = None, transfer_params: dict = None,
                **kwargs):
        try:
            cancel_event = kwargs.get('cancel_event', None)
            if cancel_event and cancel_event.is_set():
                LOG.info(f'[_Processor - add_doc] Task canceled! files:{input_files}')
                raise RuntimeError('Task canceled!')
            add_start_time = time.time()
            if not input_files: return
            if not ids: ids = [gen_docid(path) for path in input_files]
            if metadatas is None:
                metadatas = [{} for _ in input_files]
            for metadata, doc_id, path in zip(metadatas, ids, input_files):
                metadata.setdefault(RAG_DOC_ID, doc_id)
                metadata.setdefault(RAG_DOC_PATH, path)
                metadata.setdefault(RAG_KB_ID, DEFAULT_KB_ID)
            parse_start_time = time.time()
            if transfer_params is not None:
                # skip reader, get nodes from the store
                parse_end_time = time.time()
                self._transfer_impl(doc_ids=ids, metadatas=metadatas, **transfer_params, cancel_event=cancel_event)
                pass
            else:
                root_nodes, image_nodes = self._reader.load_data(input_files, metadatas, split_image_nodes=True)
                parse_end_time = time.time()
                if cancel_event and cancel_event.is_set():
                    LOG.info(f'[_Processor - add_doc] Task canceled! files:{input_files}')
                    raise RuntimeError('Task canceled!')
                self._store.update_nodes(self._set_nodes_number(root_nodes))
                self._create_nodes_recursive(root_nodes, LAZY_ROOT_NAME, cancel_event=cancel_event)
                if image_nodes:
                    self._store.update_nodes(self._set_nodes_number(image_nodes))
                    self._create_nodes_recursive(image_nodes, LAZY_IMAGE_GROUP, cancel_event=cancel_event)
            add_end_time = time.time()
            LOG.info(f'[_Processor - add_doc] Add documents done! files:{input_files}, '
                     f'Total Time:{add_end_time - add_start_time}s, '
                     f'Parse Time:{parse_end_time - parse_start_time}s')
        except RuntimeError as e:
            if 'Task canceled!' in str(e):
                LOG.info(f'[_Processor - add_doc] Task canceled! files:{input_files}, ids:{ids}, metadatas:{metadatas}')
                kb_id = metadatas[0].get(RAG_KB_ID, None)
                self._store.remove_nodes(doc_ids=ids, kb_id=kb_id)
            else:
                LOG.error(f'Add documents failed: {e}, {traceback.format_exc()}')
            raise e
        except Exception as e:
            LOG.error(f'Add documents failed: {e}, {traceback.format_exc()}')
            raise e

    def _set_nodes_number(self, nodes: List[DocNode]) -> List[DocNode]:
        doc_group_number = {}
        for node in nodes:
            doc_id = node.global_metadata.get(RAG_DOC_ID)
            group_name = node.group
            if doc_id not in doc_group_number:
                doc_group_number[doc_id] = {}
            if group_name not in doc_group_number[doc_id]:
                doc_group_number[doc_id][group_name] = 1
            node.metadata['lazyllm_store_num'] = doc_group_number[doc_id][group_name]
            doc_group_number[doc_id][group_name] += 1
        return nodes

    def _create_nodes_recursive(self, p_nodes: List[DocNode], p_name: str, **kwargs):
        cancel_event = kwargs.get('cancel_event', None)
        for group_name in self._store.activated_groups():
            group = self._node_groups.get(group_name)
            if group is None:
                raise ValueError(f'Node group {group_name} does not exist. Please check the group name '
                                 'or add a new one through `create_node_group`.')
            if cancel_event and cancel_event.is_set():
                LOG.info(f'[_Processor - _create_nodes_recursive] Task canceled! group_name:{group_name}')
                raise RuntimeError('Task canceled!')
            if group['parent'] == p_name:
                nodes = self._create_nodes_impl(p_nodes, group_name)
                if nodes: self._create_nodes_recursive(nodes, group_name, cancel_event=cancel_event)

    def _create_nodes_impl(self, p_nodes, group_name):
        # NOTE transform.batch_forward will set children for p_nodes, but when calling
        # transform.batch_forward, p_nodes has been upsert in the store.
        t = self._node_groups[group_name]['transform']
        transform = AdaptiveTransform(t) if isinstance(t, list) or t.pattern else make_transform(t, group_name)
        nodes = transform.batch_forward(p_nodes, group_name)
        self._store.update_nodes(self._set_nodes_number(nodes))
        return nodes

    def _get_or_create_nodes(self, group_name, uids: Optional[List[str]] = None):
        nodes = self._store.get_nodes(uids=uids, group=group_name) if self._store.is_group_active(group_name) else []
        if not nodes and group_name not in (LAZY_IMAGE_GROUP, LAZY_ROOT_NAME):
            p_nodes = self._get_or_create_nodes(self._node_groups[group_name]['parent'], uids)
            nodes = self._create_nodes_impl(p_nodes, group_name)
        return nodes

    def _transfer_impl(self, doc_ids: List[str], metadatas: List[Dict], target_kb_id: str,  # noqa C901
                       target_doc_ids: List[str], mode: str = 'cp', **kwargs):
        try:
            cancel_event = kwargs.get('cancel_event', None)
            if cancel_event and cancel_event.is_set():
                LOG.info(f'[_Processor - transfer_doc] Task canceled! doc_ids:{doc_ids}')
                raise RuntimeError('Task canceled!')
            if mode not in ['cp', 'mv']:
                raise ValueError(f'Invalid mode: {mode}')
            # copy process(from root group to group leaf)
            if len(doc_ids) != len(target_doc_ids):
                raise ValueError(f'The length of doc_ids and target_doc_ids must be the same. '
                                 f'doc_ids:{doc_ids}, target_doc_ids:{target_doc_ids}')
            # origin kb id
            kb_id = metadatas[0].get(RAG_KB_ID)
            doc_id_map = {}
            for i, target_doc_id in enumerate(target_doc_ids):
                # origin doc id --> new doc id
                doc_id_map[doc_ids[i]] = target_doc_id

            doc_id_meta_map = {}
            for i, meta in enumerate(metadatas):
                new_meta = {}
                for k, v in meta.items():
                    if k == RAG_DOC_ID:
                        new_meta[RAG_DOC_ID] = doc_id_map[v]
                    elif k == RAG_KB_ID:
                        new_meta[RAG_KB_ID] = target_kb_id
                    else:
                        new_meta[k] = v
                doc_id_meta_map[doc_ids[i]] = new_meta
            # root
            root_segs = self._store.copy_segments(doc_ids=doc_ids, group=LAZY_ROOT_NAME, kb_id=kb_id)
            root_uid_map = {}
            for seg in root_segs:
                root_uid_map[seg['copy_source']['uid']] = seg['uid']
                seg['doc_id'] = doc_id_map[seg['copy_source']['doc_id']]
                seg[RAG_KB_ID] = target_kb_id
                seg['global_meta'].update(doc_id_meta_map[seg['copy_source']['doc_id']])
            self._store.update_segments(root_segs, type='copy')

            def _copy_segments_recursive(p_uid_map: dict, p_name: str, **kwargs):
                cancel_event = kwargs.get('cancel_event', None)
                for group_name in self._store.activated_groups():
                    group = self._node_groups.get(group_name)
                    if group is None:
                        raise ValueError(f'Node group {group_name} does not exist. Please check the group name '
                                         'or add a new one through `create_node_group`.')
                    if cancel_event and cancel_event.is_set():
                        LOG.info(f'[_Processor - _create_nodes_recursive] Task canceled! group_name:{group_name}')
                        raise RuntimeError('Task canceled!')
                    if group['parent'] == p_name:
                        segs = self._store.copy_segments(doc_ids=doc_ids, group=group_name,
                                                         kb_id=kb_id)
                        uid_map = {}
                        for seg in segs:
                            uid_map[seg['copy_source']['uid']] = seg['uid']
                            seg['doc_id'] = doc_id_map[seg['copy_source']['doc_id']]
                            seg[RAG_KB_ID] = target_kb_id
                            seg['global_meta'].update(doc_id_meta_map[seg['copy_source']['doc_id']])
                            seg['parent'] = p_uid_map.get(seg['parent'], None) if seg['parent'] else None
                        self._store.update_segments(segs, type='copy')
                        if segs: _copy_segments_recursive(uid_map, group_name, cancel_event=cancel_event)
            # leaf
            _copy_segments_recursive(p_uid_map=root_uid_map, p_name=LAZY_ROOT_NAME, cancel_event=cancel_event)
            if cancel_event and cancel_event.is_set():
                LOG.info(f'[_Processor - transfer_doc] Task canceled! doc_ids:{doc_ids}')
                raise RuntimeError('Task canceled!')
            # move process
            if mode == 'mv':
                self._store.remove_nodes(doc_ids=doc_ids, kb_id=kb_id)
            return
        except RuntimeError as e:
            if 'Task canceled!' in str(e):
                LOG.info(f'[_Processor - transfer_doc] Task canceled! target doc_ids:{target_doc_ids}')
                self._store.remove_nodes(doc_ids=target_doc_ids, kb_id=target_kb_id)
            else:
                LOG.error(f'Transfer documents failed: {e}, {traceback.format_exc()}')
            raise e
        except Exception as e:
            LOG.error(f'Transfer documents failed: {e}, {traceback.format_exc()}')
            raise e

    def reparse(self, group_name: str, uids: Optional[List[str]] = None, doc_ids: Optional[List[str]] = None, **kwargs):
        cancel_event = kwargs.get('cancel_event', None)
        if cancel_event and cancel_event.is_set():
            LOG.info(f'[_Processor - reparse] Task canceled! group_name:{group_name}')
            raise RuntimeError('Task canceled!')
        if doc_ids:
            self._reparse_docs(group_name=group_name, doc_ids=doc_ids, **kwargs)
        else:
            self._get_or_create_nodes(group_name, uids)

    def _reparse_docs(self, group_name: str, doc_ids: List[str], doc_paths: List[str], metadatas: List[Dict], **kwargs):
        cancel_event = kwargs.get('cancel_event', None)
        kb_id = metadatas[0].get(RAG_KB_ID, None)
        if group_name == 'all':
            if cancel_event and cancel_event.is_set():
                LOG.info(f'[_Processor - _reparse_docs] Task canceled! group_name:{group_name}')
                raise RuntimeError('Task canceled!')
            self._store.remove_nodes(doc_ids=doc_ids, kb_id=kb_id)
            removed_flag = False
            for wait_time in fibonacci_backoff():
                nodes = self._store.get_nodes(group=LAZY_ROOT_NAME, kb_id=kb_id, doc_ids=doc_ids)
                if not nodes:
                    removed_flag = True
                    break
                time.sleep(wait_time)
            if not removed_flag:
                raise Exception(f'Failed to remove nodes for docs {doc_ids} from store')
            self.add_doc(input_files=doc_paths, ids=doc_ids, metadatas=metadatas, cancel_event=cancel_event)
        else:
            p_nodes = self._store.get_nodes(group=self._node_groups[group_name]['parent'],
                                            kb_id=kb_id, doc_ids=doc_ids)
            self._reparse_group_recursive(p_nodes=p_nodes, cur_name=group_name, doc_ids=doc_ids,
                                          cancel_event=cancel_event)

    def _reparse_group_recursive(self, p_nodes: List[DocNode], cur_name: str, doc_ids: List[str], **kwargs):
        cancel_event = kwargs.get('cancel_event', None)
        kb_id = p_nodes[0].global_metadata.get(RAG_KB_ID, None)
        self._store.remove_nodes(group=cur_name, kb_id=kb_id, doc_ids=doc_ids)
        if cancel_event and cancel_event.is_set():
            LOG.info(f'[_Processor - _reparse_group_recursive] Task canceled! cur_name:{cur_name}')
            raise RuntimeError('Task canceled!')
        removed_flag = False
        for wait_time in fibonacci_backoff():
            if cancel_event and cancel_event.is_set():
                LOG.info(f'[_Processor - _reparse_group_recursive] Task canceled! cur_name:{cur_name}')
                raise RuntimeError('Task canceled!')
            nodes = self._store.get_nodes(group=cur_name, kb_id=kb_id, doc_ids=doc_ids)
            if not nodes:
                removed_flag = True
                break
            time.sleep(wait_time)
        if not removed_flag:
            raise Exception(f'Failed to remove nodes for docs {doc_ids} group {cur_name} from store')

        t = self._node_groups[cur_name]['transform']
        transform = AdaptiveTransform(t) if isinstance(t, list) or t.pattern else make_transform(t, cur_name)
        nodes = transform.batch_forward(p_nodes, cur_name)
        # reparse need set global_metadata
        if cancel_event and cancel_event.is_set():
            LOG.info(f'[_Processor - _reparse_group_recursive] Task canceled! cur_name:{cur_name}')
            raise RuntimeError('Task canceled!')
        self._store.update_nodes(self._set_nodes_number(nodes))

        for group_name in self._store.activated_groups():
            group = self._node_groups.get(group_name)
            if group is None:
                raise ValueError(f'Node group {group_name} does not exist. Please check the group name '
                                 'or add a new one through `create_node_group`.')
            if cancel_event and cancel_event.is_set():
                LOG.info(f'[_Processor - _reparse_group_recursive] Task canceled! cur_name:{cur_name}')
                raise RuntimeError('Task canceled!')
            if group['parent'] == cur_name:
                self._reparse_group_recursive(p_nodes=nodes, cur_name=group_name, doc_ids=doc_ids,
                                              cancel_event=cancel_event)

    def update_doc_meta(self, doc_id: str, metadata: dict):
        self._store.update_doc_meta(doc_id=doc_id, metadata=metadata)

    def delete_doc(self, doc_ids: List[str] = None, kb_id: str = None) -> None:
        if isinstance(doc_ids, str):
            doc_ids = [doc_ids]
        LOG.info(f'delete_doc_ids: {doc_ids}')
        self._store.remove_nodes(doc_ids=doc_ids, kb_id=kb_id)


class TransferParams(BaseModel):
    mode: Optional[str] = 'cp'  # cp or mv
    target_algo_id: str
    target_doc_id: str
    target_kb_id: str


class FileInfo(BaseModel):
    file_path: Optional[str] = None
    transformed_file_path: Optional[str] = None
    doc_id: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = Field(default_factory=dict)
    reparse_group: Optional[str] = None
    transfer_params: Optional[TransferParams] = None


class DBInfo(BaseModel):
    db_type: str
    db_name: str
    user: str
    password: str
    host: str
    port: int
    table_name: str
    options_str: Optional[str] = None


class AddDocRequest(BaseModel):
    task_id: str
    algo_id: Optional[str] = '__default__'
    file_infos: List[FileInfo]
    db_info: Optional[DBInfo] = None
    feedback_url: Optional[str] = None


class UpdateMetaRequest(BaseModel):
    algo_id: Optional[str] = '__default__'
    file_infos: List[FileInfo]
    db_info: Optional[DBInfo] = None


class DeleteDocRequest(BaseModel):
    algo_id: Optional[str] = '__default__'
    dataset_id: str
    doc_ids: List[str]
    db_info: Optional[DBInfo] = None


class CancelDocRequest(BaseModel):
    task_id: str


class DocumentProcessor(ModuleBase):

    class Impl():
        def __init__(self, server: bool):
            self._processors: Dict[str, _Processor] = dict()
            self._server = server
            self._inited = False
            self._draining = False
            try:
                self._feedback_url = config['process_feedback_service']
                self._path_prefix = config['process_path_prefix']
            except Exception as e:
                LOG.warning(f'Failed to get config: {e}, use env variables instead')
                self._feedback_url = os.getenv('PROCESS_FEEDBACK_SERVICE', None)
                self._path_prefix = os.getenv('PROCESS_PATH_PREFIX', None)
            self._queue_get_url = os.getenv('RAG_FILE_QUEUE_GET_URL', None)

        def _init_components(self, server: bool):
            if server and not self._inited:
                self._tasks = {}    # running tasks
                self._pending_task_ids = set()  # pending task ids
                self._cancel_tokens: dict[str, threading.Event] = {}
                self._lock = threading.Lock()

                self._max_workers = int(os.getenv('RAG_PROCESS_MAX_WORKERS', '8'))
                self._task_queue = queue.Queue(maxsize=2 * self._max_workers)

                self._add_executor = ThreadPoolExecutor(max_workers=self._max_workers)
                self._delete_executor = ThreadPoolExecutor(max_workers=self._max_workers)
                self._update_executor = ThreadPoolExecutor(max_workers=self._max_workers)

                self._engines: dict[str, Engine] = {}
                self._inspectors: dict[str, inspect] = {}
                self._update_futures = {}

                self._stop_event = threading.Event()

                self._worker_thread = threading.Thread(target=self._worker, daemon=True)
                self._worker_thread.start()

                if self._queue_get_url:
                    self._poller_id = 'worker_' + subprocess.check_output(['hostname', '-i']).decode().strip()
                    self._poll_interval = float(os.getenv('RAG_PROCESS_POLL_INTERVAL', '1.0'))
                    self._poller_thread = threading.Thread(target=self._poller, daemon=True)
                    self._poller_thread.start()
                try:
                    signal.signal(signal.SIGTERM, lambda *_: self._begin_shutdown())
                except Exception as e:
                    LOG.error(f'[DocumentProcessor] Failed to register signal handler: {e}')
                    try:
                        import atexit
                        atexit.register(self._begin_shutdown)
                    except Exception as e:
                        LOG.error(f'[DocumentProcessor] Failed to register atexit handler: {e}')

            self._inited = True
            LOG.info(f'[DocumentProcessor] init done. feedback_url {self._feedback_url}, prefix {self._path_prefix}')

        def register_algorithm(self, name: str, store: _DocumentStore, reader: ReaderBase,
                               node_groups: Dict[str, Dict], display_name: Optional[str] = None,
                               description: Optional[str] = None, force_refresh: bool = False):
            self._init_components(server=self._server)
            if name in self._processors and not force_refresh:
                LOG.warning(f'There is already a processor with the same name {name}!')
                return
            self._processors[name] = _Processor(store, reader, node_groups, display_name, description)
            LOG.info(f'Processor {name} registered!')

        def drop_algorithm(self, name: str, clean_db: bool = False) -> None:
            if name not in self._processors:
                LOG.warning(f'Processor {name} not found!')
                return
            self._processors.pop(name)

        def _begin_shutdown(self):
            if self._draining:
                return
            LOG.info('[DocumentProcessor] Draining...')
            self._draining = True
            self._stop_event.set()
            threading.Thread(target=self._graceful_shutdown, daemon=True).start()

        def _get_engine(self, url) -> Engine:
            if url not in self._engines:
                engine = create_engine(url, echo=False, pool_pre_ping=True)
                self._engines[url] = engine
                self._inspectors[url] = inspect(engine)
            return self._engines[url]

        def _get_inspector(self, url):
            self._get_engine(url=url)
            return self._inspectors[url]

        def _get_url_from_db_info(self, db_info: DBInfo):
            return (f'mysql+pymysql://{db_info.user}:{db_info.password}'
                    f'@{db_info.host}:{db_info.port}/{db_info.db_name}'
                    '?charset=utf8mb4')

        def create_table(self, db_info: DBInfo):
            if db_info.db_type == 'mysql':
                try:
                    url = self._get_url_from_db_info(db_info)
                    engine = self._get_engine(url=url)
                    inspector = self._get_inspector(url=url)
                    tbl = db_info.table_name
                    schema = db_info.db_name

                    if not inspector.has_table(tbl, schema=schema):
                        metadata = MetaData()
                        table = Table(tbl, metadata, Column('document_id', String(255), primary_key=True),
                                      Column('file_name', String(255), nullable=False),
                                      Column('file_path', String(255), nullable=False),
                                      Column('description', String(255), nullable=True),
                                      Column('creater', String(255), nullable=False),
                                      Column('dataset_id', String(255), nullable=False),
                                      Column('tags', JSON, nullable=True),
                                      Column('created_at', TIMESTAMP, server_default=text('CURRENT_TIMESTAMP')))
                        metadata.create_all(engine, tables=[table])
                        LOG.info(f'Created table `{tbl}` in `{schema}`')
                except Exception as e:
                    LOG.error(f'Failed to create table `{tbl}` in `{schema}`: {e}')
                    return
            else:
                raise ValueError(f'Unsupported database type: {db_info.db_type}')

        def operate_db(self, db_info: DBInfo, operation: str,
                       file_infos: List[FileInfo] = None, params: Dict = None) -> None:
            db_type = db_info.db_type
            if db_type not in DB_TYPES:
                raise ValueError(f'Unsupported db_type: {db_type}')
            url = self._get_url_from_db_info(db_info)
            engine = self._get_engine(url=url)
            if operation == 'upsert':
                self._upsert_records(engine, db_info, file_infos)
            elif operation == 'delete':
                self._delete_records(engine, db_info, params)
            else:
                raise ValueError(f'Unsupported operation: {operation}')

        def _upsert_records(self, engine, db_info, file_infos):
            table_name = db_info.table_name
            metadata = MetaData()
            metadata.reflect(bind=engine, only=[table_name])
            table = metadata.tables[table_name]
            with engine.begin() as conn:
                for file_info in file_infos:
                    document_id = file_info.get('doc_id')
                    file_path = file_info.get('file_path')
                    if not document_id or not file_path:
                        raise ValueError(f'Invalid file_info: {file_info}')

                    raw_infos = {'document_id': document_id, 'file_name': os.path.basename(file_path),
                                 'file_path': file_path, 'description': file_info['metadata'].get('description', None),
                                 'creater': file_info['metadata'].get('creater', None),
                                 'dataset_id': file_info['metadata'].get(RAG_KB_ID, None),
                                 'tags': file_info['metadata'].get('tags', []) or []}
                    infos = {}
                    for k, v in raw_infos.items():
                        if v is None:
                            continue
                        if isinstance(v, str) and not v.strip():
                            continue
                        if isinstance(v, (list, dict)) and not v:
                            continue
                        infos[k] = v
                    if 'document_id' not in infos:
                        infos['document_id'] = document_id

                    stmt = mysql_insert(table).values(**infos)
                    update_dict = {k: stmt.inserted[k] for k in infos if k != 'document_id'}
                    upsert_stmt = stmt.on_duplicate_key_update(**update_dict)
                    conn.execute(upsert_stmt)

        def _delete_records(self, engine, db_info, params):
            table_name = db_info.table_name
            metadata = MetaData()
            metadata.reflect(bind=engine, only=[table_name])
            table = metadata.tables[table_name]

            with engine.begin() as conn:  # 自动提交或回滚事务
                doc_ids = params.get('doc_ids', [])
                for document_id in doc_ids:
                    stmt = delete(table).where(table.c.document_id == document_id)
                    conn.execute(stmt)

        @app.get('/ready')
        async def ready(self):
            if self._draining:
                return BaseResponse(code=503, msg='draining')
            return BaseResponse(code=200, msg='ok')

        @app.get('/prestop')
        async def prestop(self):
            self._begin_shutdown()
            return BaseResponse(code=200, msg='ok')

        def _poller(self):  # noqa: C901
            if not self._queue_get_url:
                LOG.warning('[DocumentProcessor - _poller] queue_get_url not set, poller disabled')
                return
            headers = {'Accept': 'application/json'}
            empty_backoff = self._poll_interval
            while not self._stop_event.is_set() and not self._draining:
                try:
                    active = sum(1 for (fut, _) in list(self._tasks.values()) if fut and not fut.done())
                    capacity = max(0, self._max_workers - active - self._task_queue.qsize())

                    if capacity <= 0:
                        LOG.warning(f'[Poller] capacity: {capacity} <= 0, sleep 0.2s')
                        time.sleep(0.2)
                        continue
                    params = {'worker_id': self._poller_id}
                    resp = requests.get(self._queue_get_url, headers=headers, timeout=10, params=params)
                    if resp.status_code == 204 or not resp.content:
                        LOG.warning(f'[Poller] get empty response, resp.status_code: {resp.status_code},'
                                    f' resp.content: {resp.content}, sleep {empty_backoff}s...')
                        time.sleep(empty_backoff)
                        empty_backoff = min(empty_backoff * 1.5, 5.0)
                        continue

                    resp.raise_for_status()
                    data = resp.json()

                    task = data.get('task', {})
                    worker_id = data.get('worker_id')

                    if not task:
                        LOG.warning(f'[Poller] resp get empty task, sleep {empty_backoff}s...')
                        time.sleep(empty_backoff)
                        empty_backoff = min(empty_backoff * 1.5, 5.0)
                        continue
                    LOG.info(f'[Poller] resp {data}')
                    if worker_id != self._poller_id:
                        LOG.warning(f'[Poller] task is not for this worker {self._poller_id} !'
                                    f' Current worker {worker_id}')
                    raw_db_info = task.get('db_info')
                    if not raw_db_info:
                        task['db_info'] = None
                    task_info = AddDocRequest(**task)
                    task_id = task_info.task_id
                    algo_id = task_info.algo_id
                    db_info = task_info.db_info
                    feedback_url = task_info.feedback_url
                    file_infos = task_info.file_infos

                    if ENABLE_DB and db_info is not None:
                        self.create_table(db_info=db_info)

                    if self._path_prefix:
                        for file_info in file_infos:
                            source_path = file_info.transformed_file_path if \
                                USE_TRANSFORMED_FILE and file_info.transformed_file_path \
                                else file_info.file_path
                            file_info.file_path = create_file_path(path=source_path, prefix=self._path_prefix)

                    params = {'file_infos': file_infos, 'db_info': db_info, 'feedback_url': feedback_url}
                    with self._lock:
                        self._pending_task_ids.add(task_id)
                    self._task_queue.put(('add', algo_id, task_id, params))
                    LOG.info(f'[Poller] task {task_id} pulled, params {params}')
                    empty_backoff = self._poll_interval
                except Exception as e:
                    LOG.error(f'[Poller] fetch failed: {e}')
                    time.sleep(min(self._poll_interval * 2, 5.0) + random.random() * 0.5)
            LOG.info('[Poller] stopped')

        def _graceful_shutdown(self, wait_sec: int = 10):
            deadline = time.time() + wait_sec

            while time.time() < deadline:
                if all(f.done() for f, _ in list(self._tasks.values())):
                    break
                time.sleep(0.5)

            for task_id, (future, callback_path) in list(self._tasks.items()):
                if future and not future.done():
                    tok = self._cancel_tokens.get(task_id)
                    if tok: tok.set()
                    LOG.warning(f'[Shutdown] cancel requested for running task {task_id}')

            while not self._task_queue.empty():
                task_type, algo_id, task_id, params = self._task_queue.get(timeout=1)
                if task_type == 'add':
                    callback_path = params.get('feedback_url')
                    if callback_path:
                        self._send_status_message(task_id, callback_path, success=False,
                                                  error_code='Shutdown', error_msg='pod is draining')

                with self._lock:
                    self._pending_task_ids.discard(task_id)
            LOG.info('[Shutdown] drain completed')

        @app.get('/algo/list')
        async def get_algo_list(self) -> None:
            res = []
            for algo_id, processor in self._processors.items():
                res.append({'algo_id': algo_id, 'display_name': processor._display_name,
                            'description': processor._description})
            return BaseResponse(code=200, msg='success', data=res)

        @app.get('/group/info')
        async def get_group_info(self, algo_id: str) -> None:
            if algo_id not in self._processors:
                return BaseResponse(code=400, msg=f'Invalid algo_id {algo_id}')
            processor = self._processors[algo_id]
            infos = []
            for group_name in processor._store.activated_groups():
                if group_name in processor._node_groups:
                    group_info = {'name': group_name, 'type': processor._node_groups[group_name].get('group_type'),
                                  'display_name': processor._node_groups[group_name].get('display_name')}
                    infos.append(group_info)
            LOG.info(f'Get group info for {algo_id} success with {infos}')
            return BaseResponse(code=200, msg='success', data=infos)

        @app.post('/doc/add')
        async def async_add_doc(self, request: AddDocRequest):
            if self._draining:
                return BaseResponse(code=503, msg='draining')
            LOG.info(f'Add doc for {request.model_dump_json()}')
            task_id = request.task_id
            algo_id = request.algo_id
            file_infos = request.file_infos
            db_info = request.db_info
            feedback_url = request.feedback_url
            if algo_id not in self._processors:
                return BaseResponse(code=400, msg=f'Invalid algo_id {algo_id}')
            if task_id in self._pending_task_ids or task_id in self._tasks:
                return BaseResponse(code=400, msg=f'The task {task_id} already exists in queue', data=None)
            if self._path_prefix:
                for file_info in file_infos:
                    source_path = file_info.transformed_file_path if file_info.transformed_file_path \
                        else file_info.file_path
                    file_info.file_path = create_file_path(path=source_path, prefix=self._path_prefix)

            params = {'file_infos': file_infos, 'db_info': db_info, 'feedback_url': feedback_url}
            if ENABLE_DB and db_info is not None:
                self.create_table(db_info=db_info)
            self._pending_task_ids.add(task_id)
            self._task_queue.put(('add', algo_id, task_id, params))
            return BaseResponse(code=200, msg='task submit successfully', data={'task_id': task_id})

        @app.post('/doc/meta/update')
        async def async_update_meta(self, request: UpdateMetaRequest):
            if self._draining:
                return BaseResponse(code=503, msg='draining')
            LOG.info(f'update doc meta for {request.model_dump_json()}')
            algo_id = request.algo_id
            file_infos = request.file_infos
            db_info = request.db_info

            if algo_id not in self._processors:
                return BaseResponse(code=400, msg=f'Invalid algo_id {algo_id}')

            for file_info in file_infos:
                doc_id = file_info.doc_id
                metadata = file_info.metadata
                old_fut = self._update_futures.get(doc_id)
                if old_fut and not old_fut.done():
                    cancelled = old_fut.cancel()
                    LOG.info(f'Canceled previous update for {doc_id}: {cancelled}')

                new_fut = self._update_executor.submit(self._processors[algo_id].update_doc_meta, doc_id=doc_id,
                                                       metadata=metadata)

                self._update_futures[doc_id] = new_fut

                def _cleanup(fut, doc_id=doc_id):
                    if self._update_futures.get(doc_id) is fut:
                        del self._update_futures[doc_id]
                new_fut.add_done_callback(_cleanup)
                if ENABLE_DB and db_info is not None:
                    new_fut.add_done_callback(
                        lambda fut, dbi=db_info, fi=file_info: self.operate_db(dbi, 'upsert', file_infos=[fi]))

            return BaseResponse(code=200, msg='success')

        @app.delete('/doc/delete')
        async def async_delete_doc(self, request: DeleteDocRequest) -> None:
            if self._draining:
                return BaseResponse(code=503, msg='draining')
            LOG.info(f'Del doc for {request.model_dump_json()}')
            algo_id = request.algo_id
            dataset_id = request.dataset_id
            doc_ids = request.doc_ids
            db_info = request.db_info

            if algo_id not in self._processors:
                return BaseResponse(code=400, msg=f'Invalid algo_id {algo_id}')

            task_id = str(uuid.uuid4())
            self._pending_task_ids.add(task_id)
            self._task_queue.put(('delete', algo_id, task_id,
                                  {'dataset_id': dataset_id, 'doc_ids': doc_ids, 'db_info': db_info}))
            return BaseResponse(code=200, msg='task submit successfully', data={'task_id': task_id})

        @app.post('/doc/cancel')
        async def cancel_task(self, request: CancelDocRequest):  # noqa: C901
            if self._draining:
                LOG.warning('[Cancel task] system is draining')
                raise HTTPException(status_code=503, detail='system is draining')
            task_id = request.task_id
            data = {'task_id': task_id, 'status': 0, 'task_status': 0, 'message': 'canceled'}
            if task_id in self._pending_task_ids:
                with self._lock:
                    self._pending_task_ids.remove(task_id)
                data['status'] = 1
                data['message'] = 'canceled (pending removed)'
                LOG.info(f'[Cancel task] task {task_id} canceled (pending removed), data: {data}')
                return BaseResponse(code=200, msg='success', data=data)
            if task_id not in self._tasks:
                LOG.warning(f'[Cancel task] task {task_id} not found')
                raise HTTPException(status_code=404, detail='task not found')
            entry = self._tasks.get(task_id)
            future = None
            if hasattr(entry, 'done') and hasattr(entry, 'cancel'):
                future = entry
            elif isinstance(entry, tuple):
                for part in entry:
                    if hasattr(part, 'done') and hasattr(part, 'cancel'):
                        future = part
                        break
            else:
                LOG.error(f'[Cancel task] Invalid task entry: {entry}')
                raise HTTPException(status_code=500, detail='Invalid task entry')

            if future and not future.done():
                cancel_token = self._cancel_tokens.get(task_id)
                if cancel_token:
                    cancel_token.set()
                data['status'] = 1
                data['message'] = 'canceled'
                LOG.info(f'[Cancel task] task {task_id} canceled, data: {data}')
                return BaseResponse(code=200, msg='success', data=data)
            else:
                try:
                    if future.cancelled():
                        data['status'] = 1
                        data['message'] = 'canceled'
                        LOG.info(f'[Cancel task] task {task_id} canceled, data: {data}')
                        return BaseResponse(code=200, msg='success', data=data)
                    else:
                        data['status'] = 0
                        data['message'] = 'finished'
                        ex = future.exception()
                        if ex:
                            data['task_status'] = 2
                            data['message'] = str(ex)
                        else:
                            data['task_status'] = 1
                        LOG.info(f'[Cancel task] task {task_id} already finished, data: {data}')
                        return BaseResponse(code=200, msg='task already finished', data=data)
                except Exception as e:
                    LOG.error(f'[Cancel task] error: {e}')
                    raise HTTPException(status_code=500, detail=f'error: {e}')

        def _attach_done(self, task_id, future, callback_path, db_info=None, file_infos=None):
            def _on_done(fut):
                ok = False
                err = ""
                try:
                    if fut.cancelled():
                        err = "canceled"
                    else:
                        ex = fut.exception()
                        if ex:
                            err = str(ex)
                        else:
                            ok = True
                except Exception as e:
                    err = f'exception() failed: {e}'

                try:
                    if ok and ENABLE_DB and db_info is not None and file_infos is not None:
                        self.operate_db(db_info, 'upsert', file_infos=file_infos)
                except Exception as e:
                    LOG.error(f'[DoneCB] DB upsert failed for {task_id}: {e}')
                    ok = False
                    err = f'DB upsert failed: {e}'

                with self._lock:
                    self._tasks.pop(task_id, None)
                    self._cancel_tokens.pop(task_id, None)
                if callback_path:
                    try:
                        self._send_status_message(
                            task_id, callback_path, success=ok,
                            error_code='' if ok else ('Canceled' if err == 'canceled' else 'Exception'),
                            error_msg='' if ok else err
                        )
                    except Exception as e:
                        LOG.error(f'[DoneCB] send_status_message failed for {task_id}: {e}')
            future.add_done_callback(_on_done)

        def _send_status_message(self, task_id: str, callback_path: str, success: bool,
                                 error_code: str = '', error_msg: str = ''):
            if self._feedback_url:
                try:
                    full_url = self._feedback_url + callback_path
                    payload = {'task_id': task_id, 'status': 1 if success else 0, 'error_code': error_code,
                               'error_msg': error_msg}
                    headers = {'Content-Type': 'application/json'}
                    res = None
                    for wait_time in fibonacci_backoff(max_retries=3):
                        try:
                            res = requests.post(full_url, json=payload, headers=headers, timeout=5)
                            if res.status_code == 200:
                                break
                            LOG.warning(
                                f'Task-{task_id}: Unexpected status {res.status_code}, retrying in {wait_time}s…')
                        except Exception as e:
                            LOG.error(f'Task-{task_id}: Request failed: {e}, retrying in {wait_time}s…')
                        time.sleep(wait_time)

                    if res is None:
                        raise RuntimeError('Failed to send feedback—no response received after retries')
                    res.raise_for_status()
                except Exception as e:
                    LOG.error(f'Task-{task_id}: Failed to send feedback to {full_url}: {e}')
            else:
                LOG.error('process_feedback_service is not set')

        def _exec_add_task(self, algo_id, task_id, params):  # noqa: C901
            try:
                file_infos: List[FileInfo] = params.get('file_infos')
                callback_path = params.get('feedback_url')
                db_info: DBInfo = params.get('db_info')
                # new parsing
                new_input_files = []
                new_ids = []
                new_metadatas = []
                # need reparsing
                reparse_group = None
                reparse_doc_ids = []
                reparse_files = []
                reparse_metadatas = []
                # need transfer
                transfer_mode = None
                target_algo_id = None
                target_kb_id = None
                target_doc_ids = []

                for file_info in file_infos:
                    if file_info.reparse_group:
                        reparse_group = file_info.reparse_group
                        reparse_doc_ids.append(file_info.doc_id)
                        reparse_files.append(file_info.file_path)
                        reparse_metadatas.append(file_info.metadata)
                    else:
                        new_input_files.append(file_info.file_path)
                        new_ids.append(file_info.doc_id)
                        new_metadatas.append(file_info.metadata)
                        if file_info.transfer_params:
                            if target_algo_id is not None and \
                                    target_algo_id != file_info.transfer_params.target_algo_id:
                                raise ValueError('target algo_id must be the same')
                            if target_kb_id is not None and \
                                    target_kb_id != file_info.transfer_params.target_kb_id:
                                raise ValueError('target kb_id must be the same')
                            if transfer_mode is not None and \
                                    transfer_mode != file_info.transfer_params.mode:
                                raise ValueError('Mode must be the same')
                            transfer_mode = file_info.transfer_params.mode
                            target_algo_id = file_info.transfer_params.target_algo_id
                            target_kb_id = file_info.transfer_params.target_kb_id
                            target_doc_ids.append(file_info.transfer_params.target_doc_id)

                token = threading.Event()
                with self._lock:
                    self._cancel_tokens[task_id] = token
                if transfer_mode:
                    # NOTE: currently only support transfer to the same algo_id
                    if target_algo_id not in self._processors or algo_id != target_algo_id:
                        raise ValueError('target algo_id must be the same with original algo')
                    transfer_params = {'mode': transfer_mode, 'target_kb_id': target_kb_id,
                                       'target_doc_ids': target_doc_ids}
                    future = self._add_executor.submit(self._processors[algo_id].add_doc, input_files=new_input_files,
                                                       ids=new_ids, metadatas=new_metadatas,
                                                       transfer_params=transfer_params, cancel_event=token)
                elif new_input_files:
                    future = self._add_executor.submit(self._processors[algo_id].add_doc, input_files=new_input_files,
                                                       ids=new_ids, metadatas=new_metadatas, cancel_event=token)
                elif reparse_group:
                    future = self._add_executor.submit(self._processors[algo_id].reparse, group_name=reparse_group,
                                                       doc_ids=reparse_doc_ids, doc_paths=reparse_files,
                                                       metadatas=reparse_metadatas, cancel_event=token)

                else:
                    raise ValueError('No valid input files, reparse group, or transfer doc ids')
                with self._lock:
                    self._tasks[task_id] = (future, callback_path)
                    self._pending_task_ids.remove(task_id)
                self._attach_done(task_id=task_id, future=future, callback_path=callback_path,
                                  db_info=db_info, file_infos=file_infos)
            except Exception as e:
                LOG.error(f'Task-{task_id}: add task error {e}')
                raise e

        def _exec_delete_task(self, algo_id, task_id, params):
            dataset_id = params.get('dataset_id')
            doc_ids = params.get('doc_ids')
            future = self._delete_executor.submit(
                self._processors[algo_id].delete_doc, doc_ids=doc_ids, kb_id=dataset_id
            )
            if ENABLE_DB and params.get('db_info') is not None:
                db_info = params.get('db_info')
                future.add_done_callback(lambda fut: self.operate_db(db_info, 'delete', params=params))
            self._tasks[task_id] = (future, None)
            with self._lock:
                self._pending_task_ids.remove(task_id)

        def _worker(self):  # noqa: C901
            while True:
                try:
                    if self._draining:
                        time.sleep(0.2)
                        continue
                    task_type, algo_id, task_id, params = self._task_queue.get(timeout=1)
                    with self._lock:
                        if task_id not in self._pending_task_ids:
                            LOG.warning(f'[Worker] drop task not in pending: {task_id} ({task_type},'
                                        f' {algo_id}, {params})')
                            continue
                    if task_type == 'add':
                        self._exec_add_task(algo_id=algo_id, task_id=task_id, params=params)
                    elif task_type == 'delete':
                        self._exec_delete_task(algo_id=algo_id, task_id=task_id, params=params)
                    time.sleep(0.2)
                except queue.Empty:
                    time.sleep(0.2)
                except Exception as e:
                    LOG.error(f'[Worker] error: {e}')
                    time.sleep(10)
                    continue

        def __call__(self, func_name: str, *args, **kwargs):
            return getattr(self, func_name)(*args, **kwargs)

    def __init__(self, server: bool = True, port: int = None, url: str = None):
        super().__init__()
        if not url:
            self._impl = DocumentProcessor.Impl(server=server)
            if server:
                self._impl = ServerModule(self._impl, port=port)
        else:
            self._impl = UrlModule(url=ensure_call_endpoint(url))

    def _dispatch(self, method: str, *args, **kwargs):
        impl = self._impl
        if isinstance(impl, ServerModule):
            impl._call(method, *args, **kwargs)
        else:
            getattr(impl, method)(*args, **kwargs)

    def register_algorithm(self, name: str, store: _DocumentStore, reader: ReaderBase, node_groups: Dict[str, Dict],
                           display_name: Optional[str] = None, description: Optional[str] = None,
                           force_refresh: bool = False, **kwargs):
        self._dispatch('register_algorithm', name, store, reader, node_groups,
                       display_name, description, force_refresh, **kwargs)

    def drop_algorithm(self, name: str, clean_db: bool = False) -> None:
        return self._dispatch('drop_algorithm', name, clean_db)
