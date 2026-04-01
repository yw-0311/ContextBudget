import json
import logging
import time
import uuid
from typing import Any, Optional, Tuple, Dict, List

import requests

DEFAULT_TIMEOUT = 30
MAX_RETRIES = 10
INITIAL_RETRY_DELAY = 1

logger = logging.getLogger(__name__)


def _normalize_adapter_url(retrieval_service_url: str) -> str:
    """
    允许传 base url 或完整 /retrieve：
      - http://host:8000
      - http://host:8000/retrieve
    最终都归一化到 /retrieve
    """
    # url = (retrieval_service_url or "").strip().rstrip("/")
    # if not url:
    #     return ""
    # if not url.endswith("/retrieve"):
    #     # url = f"{url}/retrieve"
    #     url = url
    return retrieval_service_url


def call_search_api_via_adapter(
    retrieval_service_url: str,
    query_list: List[str],
    topk: int = 10,
    return_scores: bool = True,
    timeout: int = DEFAULT_TIMEOUT,
) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
    """
    通过检索 adapter 的 /retrieve 接口发请求，带重试逻辑。
    期望 adapter 返回：
      {"result": [ [ {"document": {"contents": "..."},"score": 0.1}, ...], ... ]}

    Returns:
        (response_json, error_message)
    """
    request_id = str(uuid.uuid4())
    log_prefix = f"[Search Request ID: {request_id}] "

    url = _normalize_adapter_url(retrieval_service_url)
    if not url:
        err = "API Call Failed: retrieval_service_url is empty"
        logger.error(f"{log_prefix}{err}")
        return None, err

    # payload = {"queries": query_list, "topk": topk, "return_scores": return_scores}
    TAO_HEADERS = {  
        'Content-Type': 'application/json'  
    }  

    tao_data = {  
        "appId": 48011,  
        "bizCode": "test",  
        "config": {  
            "requestTimeoutMs": 20000  
        },  
        "request": {  
            "queries": query_list,  
            "return_scores": 1,  
            "topk": topk
        }  
    }  

    last_error: Optional[str] = None

    for attempt in range(MAX_RETRIES):
        try:
            logger.info(f"{log_prefix}Attempt {attempt + 1}/{MAX_RETRIES}: POST {url}")
            resp = requests.post(url, headers=TAO_HEADERS, data=json.dumps(tao_data), timeout=20) 

            # 5xx 认为可重试
            if resp.status_code in (500, 502, 503, 504):
                last_error = (
                    f"{log_prefix}Adapter Server Error ({resp.status_code}) "
                    f"on attempt {attempt + 1}/{MAX_RETRIES}"
                )
                logger.warning(last_error)
                if attempt < MAX_RETRIES - 1:
                    delay = INITIAL_RETRY_DELAY * (attempt + 1)
                    logger.info(f"{log_prefix}Retrying after {delay} seconds...")
                    time.sleep(delay)
                continue

            # 其它 HTTP 错误：直接 raise（通常不重试）
            resp.raise_for_status()

            data = resp.json()
            logger.info(f"{log_prefix}Adapter call successful on attempt {attempt + 1}")
            return data, None

        except requests.exceptions.ConnectionError as e:
            last_error = f"{log_prefix}Connection Error: {e}"
            logger.warning(last_error)
        except requests.exceptions.Timeout as e:
            last_error = f"{log_prefix}Timeout Error: {e}"
            logger.warning(last_error)
        except requests.exceptions.RequestException as e:
            # 4xx 等一般不值得重试，直接退出
            last_error = f"{log_prefix}RequestException: {e}"
            logger.error(last_error)
            break
        except json.JSONDecodeError as e:
            raw = resp.text if "resp" in locals() else "N/A"
            last_error = f"{log_prefix}JSONDecodeError: {e}, Response: {raw[:200]}"
            logger.error(last_error)
            break
        except Exception as e:
            last_error = f"{log_prefix}Unexpected Error: {e}"
            logger.error(last_error)
            break

        # 对 Connection/Timeout：按线性退避重试
        if attempt < MAX_RETRIES - 1:
            delay = INITIAL_RETRY_DELAY * (attempt + 1)
            logger.info(f"{log_prefix}Retrying after {delay} seconds...")
            time.sleep(delay)

    logger.error(f"{log_prefix}Adapter call failed. Last error: {last_error}")
    return None, last_error.replace(log_prefix, "API Call Failed: ") if last_error else "API Call Failed after retries"


def _passages2string(retrieval_result: List[Dict[str, Any]]) -> str:
    """
    与你原版一致：把每个 query 的 topk 文档 list 格式化成人类可读文本。
    doc_item 期望包含 doc_item["document"]["contents"]
    """
    format_reference = ""
    # print(f"[DEBUG] len retrieval_result {len(retrieval_result)}")
    for idx, doc_item in enumerate(retrieval_result):
        content = doc_item["document"]["text"]
        text = " ".join(content.split()[:512]) # 和 eva保持一致 page['text'] = " ".join(page['text'].split()[:512])  # 512
        # print(f"[DEBUG]{idx},{text},{len(text)}")
        format_reference += f"Doc {idx + 1}{text}\n\n"
    return format_reference.strip()



def _unwrap_adapter_payload(api_response: Any) -> Any:
    """
    新返回：{"success": True, "data": {"data": old_payload}}
    这里把 old_payload 取出来返回。
    """
    if not isinstance(api_response, dict):
        return api_response
    # 你明确说 data 在 {'success': True, 'data': {'data': data}} 里
    outer_data = api_response.get("data")
    if isinstance(outer_data, dict) and "data" in outer_data:
        return outer_data.get("data")
    return api_response


def _coerce_to_query_batches(raw: Any) -> List[List[Dict[str, Any]]]:
    """
    把各种可能形状，统一成：
      List[ List[doc_item] ]  # 外层=每个 query，内层=topk 文档
    """
    # 1) old_payload 是 dict：尝试常见 key
    if isinstance(raw, dict):
        for k in ("result", "results", "data", "docs"):
            if k in raw:
                raw = raw[k]
                break

    # 2) raw 可能已经是 list
    if isinstance(raw, list):
        if len(raw) == 0:
            return []
        # 如果第一层就是 doc_item dict（而不是 list），说明只有一个 query，补一层
        if all(isinstance(x, dict) for x in raw):
            return [raw]
        # 如果第一层是 list，认为已经是 query batches
        if all(isinstance(x, list) for x in raw):
            # 里面可能不是 dict（比如字符串），这里尽量转成 dict 形式
            batches: List[List[Dict[str, Any]]] = []
            for one_query in raw:
                if not isinstance(one_query, list):
                    one_query = [one_query]
                fixed: List[Dict[str, Any]] = []
                for item in one_query:
                    fixed.append(item if isinstance(item, dict) else {"document": {"contents": str(item)}})
                batches.append(fixed)
            return batches
        # 混合情况：兜底当作单 query
        return [[x if isinstance(x, dict) else {"document": {"contents": str(x)}} for x in raw]]

    # 3) raw 不是 list/dict：兜底当作一个 doc 的 contents
    if raw is None:
        return []
    return [[{"document": {"contents": str(raw)}}]]

def perform_single_search_batch(
    retrieval_service_url: str,
    query_list: List[str],
    topk: int = 10,
    concurrent_semaphore=None,  # 保持签名兼容（SearchTool 里传 None）
    timeout: int = DEFAULT_TIMEOUT,
) -> Tuple[str, Dict[str, Any]]:
    """
    保持 SearchTool 原契约不变，但内部请求改为走 adapter。
    返回：
      result_text: JSON 字符串 {"result": "...拼好的文本..." }（与原版一致）
      metadata: 结构化状态
    """
    logger.info(f"Starting batch search for {len(query_list)} queries via adapter.")

    api_response = None
    error_msg = None

    try:
        if concurrent_semaphore:
            with concurrent_semaphore:
                api_response, error_msg = call_search_api_via_adapter(
                    retrieval_service_url=retrieval_service_url,
                    query_list=query_list,
                    topk=topk,
                    return_scores=True,
                    timeout=timeout,
                )
        else:
            api_response, error_msg = call_search_api_via_adapter(
                retrieval_service_url=retrieval_service_url,
                query_list=query_list,
                topk=topk,
                return_scores=True,
                timeout=timeout,
            )
    except Exception as e:
        error_msg = f"API Request Exception during batch search: {e}"
        logger.exception(error_msg)

    metadata: Dict[str, Any] = {
        "query_count": len(query_list),
        "queries": query_list,
        "api_request_error": error_msg,
        "api_response": None,
        "status": "unknown",
        "total_results": 0,
        "formatted_result": None,
    }

    # 默认失败文本（与原版风格一致）
    result_text = json.dumps({"result": "Search request failed or timed out after retries."}, ensure_ascii=False)

    if error_msg:
        metadata["status"] = "api_error"
        result_text = json.dumps({"result": f"Search error: {error_msg}"}, ensure_ascii=False)
        return result_text, metadata

    if not api_response:
        metadata["status"] = "unknown_api_state"
        result_text = json.dumps({"result": "Unknown API state (no response and no error message)."}, ensure_ascii=False)
        return result_text, metadata

    old_payload = _unwrap_adapter_payload(api_response)

    # 处理成功返回
    metadata["api_response"] = api_response
    raw_results = api_response.get("result", [])

    batches = _coerce_to_query_batches(old_payload)

    if not batches:
        metadata["status"] = "no_results"
        result_text = json.dumps({"result": "No search results found."}, ensure_ascii=False)
        return result_text, metadata

    try:
        pretty_results: List[str] = []
        total_results = 0

        # 外层按 query，内层按 topk docs
        for one_query_docs in batches:
            formatted = _passages2string(one_query_docs)
            pretty_results.append(formatted)
            total_results += len(one_query_docs)

        final_result = "\n---\n".join([s for s in pretty_results if s.strip()])
        result_text = json.dumps({"result": final_result or "No search results found."}, ensure_ascii=False)

        metadata["status"] = "success"
        metadata["total_results"] = total_results
        metadata["formatted_result"] = final_result

        return result_text, metadata

    except Exception as e:
        metadata["status"] = "processing_error"
        err = f"Error processing search results: {e}"
        metadata["api_request_error"] = err
        result_text = json.dumps({"result": err}, ensure_ascii=False)
        return result_text, metadata
