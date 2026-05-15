from nemantix.knowledge_base.core.nemantix_knowledge_base import NemantixKnowledgeBase, KnowledgeBaseConfig
from nemantix.knowledge_base.core.knowledge_base_manager import KnowledgeBaseManager, KnowledgeBaseManagerConfig

import os
import logging

os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
os.environ["TRANSFORMERS_VERBOSITY"] = "error"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

logging.getLogger("transformers").setLevel(logging.ERROR)
logging.getLogger("sentence_transformers").setLevel(logging.ERROR)