from typing import List, Dict, Any, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed

from nemantix.common.logger import get_package_logger
from nemantix.core.exceptions import NemantixException
from nemantix.knowledge_base.document_structure.schemas import SegmentEnrichment

logger = get_package_logger(__name__)


class SegmentEnricher:
    """
        Leverages an LLM to generate concise summaries and extract structured metadata
        from individual document segments.
        """

    def __init__(self, llm_proxy: Any, max_workers: int = 5):
        """
        Initializes the SegmentEnricher.

        Args:
            llm_proxy (Any): The LLM client used to invoke structured extraction.
            max_workers (int): Maximum number of concurrent threads for batch processing.
        """

        self.llm = llm_proxy
        self.max_workers = max_workers

    def enrich_batch(self, segments: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Processes a batch of text segments concurrently to extract metadata and summaries.

        It ensures that the original order of the segments is preserved regardless
        of thread completion times, without relying on format-specific coordinate logic.

        Args:
            segments (List[Dict[str, Any]]): A list of flat segment dictionaries.

        Returns:
            List[Dict[str, Any]]: The list of enriched segments, in their original order.
        """
        enriched_segments: List[Tuple[int, Dict[str, Any]]] = []

        def _enrich_single(index: int, segment: Dict[str, Any]) -> Tuple[int, Dict[str, Any]]:
            """Isolated worker function to process a single segment."""
            prompt = f"""
            Analyze the following text segment from a document.
            Hierarchy context: {segment.get('hierarchy_ref')}

            Task:
            1. Write a very brief summary of the content.
            2. Extract useful metadata (e.g., key entities, dates, core topics, concepts) as key-value pairs.
            3. Extract metadata as a single keywords or keyword lists, not sentences.

            Text:
            {segment.get('text')}
            """

            messages = [
                {"role": "developer",
                 "content": "You are a data extraction assistant. Return strict JSON. Keep the summary concise."},
                {"role": "user", "content": prompt.strip()}
            ]

            enriched_segment = segment.copy()

            try:
                # Attempt LLM extraction
                response = self.llm.invoke_structured(
                    prompt=messages,
                    schema=SegmentEnrichment)

                enriched_data = response.result
                metadata_dict = {}

                for item in enriched_data.metadata:
                    key = item.key
                    val = item.value

                    if not val:
                        continue

                    # Flatten single-item lists for cleaner metadata indexing
                    if isinstance(val, list) and len(val) == 1:
                        metadata_dict[key] = val[0]
                    else:
                        metadata_dict[key] = val

                enriched_segment["text_view"] = enriched_data.summary
                enriched_segment["metadata"] = metadata_dict

            except Exception as err:
                # Fallback if LLM fails
                logger.warning("Fallback activated for segment %s. Error: %s",
                               segment.get('node_id', index), err)
                enriched_segment["text_view"] = segment.get("text", "")
                enriched_segment["metadata"] = {}

            return index, enriched_segment

        # Execute concurrent enrichment
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            futures = [executor.submit(_enrich_single, i, seg) for i, seg in enumerate(segments)]

            for future in as_completed(futures):
                try:
                    enriched_segments.append(future.result())
                except Exception as e:
                    logger.error("Thread error during segment enrichment: %s", e)

        if len(enriched_segments) < len(segments):
            failed_count = len(segments) - len(enriched_segments)
            error_msg = (
                f"Data Loss Warning! {failed_count} segments completely failed and were dropped "
                f"during concurrent enrichment. Aborting to prevent incomplete document ingestion."
            )
            logger.error(error_msg)
            raise NemantixException(error_msg)

        # Sort the results based on the original index, effectively restoring physical order
        enriched_segments.sort(key=lambda x: x[0])

        return [res[1] for res in enriched_segments]

    def summarize_parent_node(self, node_title: str, children_summaries: list[str]) -> str:
        """
        Generates a hierarchical summary for a parent node by synthesizing the summaries of its children.

        Args:
            node_title (str): The label or title of the parent node.
            children_summaries (List[str]): A list of string summaries from all immediate child nodes.

        Returns:
            str: A condensed, factual summary representing the parent's entire scope.
        """

        combined_text = "\n".join([f"- {s}" for s in children_summaries if s])

        prompt = f"""
        You are an expert analyst tasked with extracting factual knowledge from documents.
        Your goal is to synthesize the core information for the section titled "{node_title}", based EXCLUSIVELY on the summaries of its sub-elements below:

        {combined_text}

        CRITICAL INSTRUCTIONS:
        1. NO META-DESCRIPTIONS: Never use phrases like "This section explains...", "The document provides...", or "This part describes...". 
        2. STATE FACTS DIRECTLY: Treat the summary as an absolute source of truth. Instead of writing "The text outlines the rules for vacation days", write directly "Employees are entitled to 20 vacation days per year."
        3. BALANCE DETAIL AND BREVITY: Retain the most critical entities, definitions, core rules, or key parameters, but abstract away minor examples, edge cases, and overly granular specifics. Capture the essential meaning without creating an exhaustive list.
        """

        messages = [
            {"role": "developer",
             "content": "You are a precise knowledge extractor. You never use meta-language. You only output direct facts, rules, and core concepts."},
            {"role": "user", "content": prompt.strip()}
        ]

        response = self.llm.invoke(prompt=messages)
        return response.text
