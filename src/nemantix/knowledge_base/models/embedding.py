import numpy as np

from nemantix.common.logger import get_package_logger
from nemantix.knowledge_base.models.base import TextEmbedding

logger = get_package_logger(__name__)


# TODO: rename to SentenceTransformerEmbedder or similar
class SentenceTransformerWrapper(TextEmbedding):
    """Wrapper for SentenceTransformer models"""

    # TODO: it possible to get model card from huggingface?
    DEFAULT_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
    ALLOWED_MODELS = [
        "sentence-transformers/all-MiniLM-L6-v2",
        "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
        # TODO: add GTE
        "BAAI/bge-small-en-v1.5",  # base, large; BAAI/bge-m3
        "google/embeddinggemma-300m",
        "intfloat/multilingual-e5-large-instruct",
    ]

    def __init__(self, path: str | None = None):
        super().__init__(path or self.DEFAULT_MODEL)
        assert self.model_path in self.ALLOWED_MODELS, (
            f'Model "{self.model_path}" is not available. '
            f'See .available_models() for supported models."'
        )

        from sentence_transformers import SentenceTransformer

        self.model = SentenceTransformer(self.model_path)
        logger.info(f'Loaded model "{self.model_path}".')

    def embed(self, text: str | list[str], **kwargs) -> np.ndarray:
        # TODO: see specialized methods .encode_query(...) and .encode_document(...)
        return self.model.encode(text, **kwargs)

    def available_models(self):
        logger.info("Available models:")
        for model in self.ALLOWED_MODELS:
            logger.info(f"  {model}")


class Word2Vec(TextEmbedding):
    DEFAULT_MODEL = "NeuML/word2vec-quantized"

    def __init__(self, path: str | None = None):
        super().__init__(path or Word2Vec.DEFAULT_MODEL)
        from staticvectors import StaticVectors

        self.model = StaticVectors(self.model_path)
        logger.info(f'Loaded word2vec from "{self.model_path}".')

    def embed(self, text: str, normalize_embeddings=True) -> np.ndarray:
        return self.model.embeddings(text, normalize=bool(normalize_embeddings))
