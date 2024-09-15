from dataclasses import dataclass, field
from typing import Any, List

@dataclass
class AgentState:
    search_type: str = None
    raptor_search_type: str = "collapse_tree"
    query: str = None # 사용자 질문
    contexts: List[str] = field(default_factory=list) # 벡터스토어에서 쿼리한 문서
    metainfo: List[str] = field(default_factory=list) # 문서의 메타 정보
    generation: str = None # LLM에서 생성한 답변    