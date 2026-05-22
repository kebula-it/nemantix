import logging
import os

from nemantix.knowledge_base.core.knowledge_base_manager import (
           KnowledgeBaseManager,
           KnowledgeBaseManagerConfig,
)
from nemantix.knowledge_base.core.nemantix_knowledge_base import (
           KnowledgeBaseConfig,
           NemantixKnowledgeBase,
)

os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
os.environ["TRANSFORMERS_VERBOSITY"] = "error"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

logging.getLogger("transformers").setLevel(logging.ERROR)
logging.getLogger("sentence_transformers").setLevel(logging.ERROR)


__all__ = ["NemantixKnowledgeBase", "KnowledgeBaseConfig", "KnowledgeBaseManager", 
           "KnowledgeBaseManagerConfig"]
