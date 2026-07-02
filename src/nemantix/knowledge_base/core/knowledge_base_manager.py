import json
import os
import pickle
import networkx as nx
import concurrent.futures
import certifi

from pathlib import Path
from typing import List, Dict
from pydantic import BaseModel, Field

from nemantix.knowledge_base.document_plugins.plugin_registry import (
    DocumentPluginRegistry,
)
from nemantix.knowledge_base.document_structure.coordinates import Coordinates
from nemantix.knowledge_base.document_structure.document import Document
from nemantix.knowledge_base.document_structure.hierarchy import DocumentHierarchy
from nemantix.knowledge_base.document_structure.item import Item
from nemantix.knowledge_base.document_structure.location import Location
from nemantix.knowledge_base.persistence.relational_registry import (
    RegistryManager,
    DocumentRecord,
)
from nemantix.knowledge_base.persistence.vector_stores.factory import VectorStoreFactory
from nemantix.knowledge_base.pipeline.enricher import SegmentEnricher
from nemantix.knowledge_base.pipeline.graph_builder import GraphBuilder
from nemantix.knowledge_base.pipeline.hierarchy_planner import HierarchyPlanner
from nemantix.knowledge_base.pipeline.segmenter import DocumentSegmenter
from nemantix.knowledge_base.persistence.relational_registry import KnowledgeIndex
from nemantix.core.exceptions import NemantixException
from nemantix.common.connectors import DBConnector
from nemantix.common.logger import get_package_logger
from nemantix.llm import AbstractLLMProxy

logger = get_package_logger(__name__)


os.environ["REQUESTS_CA_BUNDLE"] = certifi.where()
os.environ["CURL_CA_BUNDLE"] = certifi.where()
os.environ["SSL_CERT_FILE"] = certifi.where()


class KnowledgeBaseManagerConfig(BaseModel):
    base_storage_path: str = Field(default="kb_storage")

    vector_subdir: str = Field(default="vector_db")
    graph_subdir: str = Field(default="graphs")
    debug_subdir: str = Field(default="debug")

    enable_debug: bool = Field(default=False)

    # Planner Parameters
    planner_window_lines: int = Field(default=1000)
    planner_overlap_lines: int = Field(default=200)

    # Enricher Parameters
    enricher_max_workers: int = Field(default=20)

    # Embedding Parameters
    embedder_model: str = Field(default="sentence-transformers/all-MiniLM-L6-v2")

    # Vector Store Parameters
    vector_store_type: str = Field(default="qdrant", description="Type of vector store")

    db_engine: str = Field(default="postgresql")
    db_username: str
    db_password: str
    db_host: str = Field(default="localhost")
    db_port: int = Field(default=5432)
    db_database: str = Field(default="nemantix_db")


class KnowledgeBaseManager:
    def __init__(self, llm: AbstractLLMProxy, config: KnowledgeBaseManagerConfig):
        from nemantix.knowledge_base.models.embedding import SentenceTransformerWrapper

        self.llm = llm

        self.config = config

        self.storage_root = Path(self.config.base_storage_path)
        self.vector_root = self.storage_root / self.config.vector_subdir
        self.graph_root = self.storage_root / self.config.graph_subdir
        self.debug_root = self.storage_root / self.config.debug_subdir

        self.vector_root.mkdir(parents=True, exist_ok=True)
        self.graph_root.mkdir(parents=True, exist_ok=True)
        if self.config.enable_debug:
            self.debug_root.mkdir(parents=True, exist_ok=True)

        self.planner = HierarchyPlanner(
            llm_proxy=self.llm,
            window_lines=self.config.planner_window_lines,
            overlap_lines=self.config.planner_overlap_lines,
        )

        self.enricher = SegmentEnricher(
            llm_proxy=self.llm, max_workers=self.config.enricher_max_workers
        )

        self.embedder = SentenceTransformerWrapper(self.config.embedder_model)

        self.db_connector = DBConnector.from_parameters(
            engine=self.config.db_engine,
            username=self.config.db_username,
            password=self.config.db_password,
            host=self.config.db_host,
            port=self.config.db_port,
            database=self.config.db_database,
        )

        self.registry_manager = RegistryManager(self.db_connector)
        self.registry_manager.initialize_database()

        self.registry = DocumentPluginRegistry.get_available_plugins()

    def _get_vector_store(self, index_name: str):
        """Utility method to dynamically instantiate the vector store."""
        storage_root = Path(self.config.base_storage_path).resolve()
        vector_root = storage_root / self.config.vector_subdir

        return VectorStoreFactory.create(
            store_type=self.config.vector_store_type,
            path=str(vector_root),
            collection_name=index_name,
        )

    def process_folder(
        self,
        folder_path: Path | str,
        index_name: str,
        target_views: List[Dict[str, str]] = None,
        doc_type: str = "unknown",
    ) -> None:
        """
        Scans a folder and performs ingestion for all files with
        supported extensions based on the loaded plugins.
        """
        folder_path = Path(folder_path)
        logger.info(
            f"\n=== Starting folder scan: {folder_path} into '{index_name}' ==="
        )

        supported_extensions = {
            f".{ext}" for ext in self.registry._plugins_by_extension.keys()
        }

        if not supported_extensions:
            logger.warning("No plugins found in the registry. Aborting.")
            return

        logger.info(
            f"  > Loaded supported extensions: {', '.join(supported_extensions)}"
        )

        processed_count = 0
        for filepath in folder_path.iterdir():
            if filepath.is_file() and filepath.suffix.lower() in supported_extensions:
                try:
                    location = Location("path", str(filepath))

                    success = self.index_document(
                        location=location,
                        doc_type=doc_type,
                        index_name=index_name,
                        target_views=target_views,
                    )
                    if success:
                        processed_count += 1
                except Exception as e:
                    logger.info(f"Error processing {filepath.name}: {e}")

        logger.info(
            f"\n=== Scan completed. Processed {processed_count} files into '{index_name}' ==="
        )

    def index_document(
        self,
        location: Location,
        index_name: str,
        target_views: List[Dict[str, str]] = None,
        doc_type: str = "unknown",
    ) -> bool:
        """
        Processes a single document: Planning, Segmentation, Graph construction, and Summarization.
        """
        if target_views is None:
            target_views = [
                {
                    "view_id": "global_view",
                    "name": "Global View",
                    "description": "Default view for all ingested documents.",
                }
            ]

        if location.location_type in ["path", "file"]:
            safe_path = Path(location.value)
            doc_name = safe_path.name
            doc_stem = safe_path.stem
        else:
            # For URLs or S3, we extract the last segment or use a default name
            doc_name = location.value.split("/")[-1] or "remote_document"
            doc_stem = doc_name.split(".")[0]

        logger.info(f"\n--- Indexing: {location.value} into '{index_name}' ---")

        # Obtain plugin via registry
        try:
            plugin = self.registry.get_plugin_for_location(location)
        except ValueError as e:
            logger.warning(f"  > Skipping document: {e}")
            return False

        document = Document.acquire(location, doc_type)

        graph_path = self.graph_root / f"{index_name}.pkl"
        try:
            self.registry_manager.get_or_create_index(
                index_name=index_name,
                graph_path=str(graph_path),
                embedding_model=self.config.embedder_model,
            )
        except ValueError as e:
            logger.error(f"  > [CRITICAL ERROR] {e}")
            raise NemantixException(
                f"Embedding model mismatch. Aborting ingestion to prevent vector space corruption: {e}"
            ) from e

        if self.registry_manager.is_document_in_index(document.doc_id, index_name):
            logger.info(
                f"  > [SKIP] Document '{doc_name}' is already present in index {index_name}. Skipping."
            )

            try:
                self.registry_manager.bind_documents_to_views(
                    [document.doc_id], target_views
                )
                logger.info(
                    f"  > Existing document successfully bound to views: {[v.get('view_id') for v in target_views]}."
                )
            except Exception as e:
                logger.error(
                    f"  > [ERROR] Failed to update views for existing document: {e}"
                )

            return False

        # Load existing Graph for this specific index
        if os.path.exists(graph_path):
            logger.info(
                f"  > [Load] Existing graph found. Loading from {graph_path}..."
            )
            with open(graph_path, "rb") as f:
                knowledge_graph = pickle.load(f)
        else:
            logger.info(
                f"  > [Init] No pre-existing graph found for '{index_name}'. Creating a new graph..."
            )
            knowledge_graph = nx.DiGraph()

        # Hierarchy inference
        logger.info("  > Inferring hierarchy...")
        planner_output = self.planner.plan(document)

        if not planner_output.nodes:
            logger.info(f"  > No nodes found in {doc_name}, skipping to next file.")
            return False

        if document.doc_type == "unknown":
            document.doc_type = planner_output.document_type

        # Save planner debug output
        if self.config.enable_debug:
            debug_planner_path = self.debug_root / f"debug_{doc_stem}_planner.json"
            with open(debug_planner_path, "w", encoding="utf-8") as f:
                f.write(planner_output.model_dump_json(indent=4))

        # Base trees and graphs construction
        doc_graph = GraphBuilder.build_from_hierarchy(document.doc_id, planner_output)
        doc_hierarchy = DocumentHierarchy.from_planner_output(
            doc_id=document.doc_id, planner_output=planner_output
        )

        # Segmentation and Enrichment
        segmenter = DocumentSegmenter(document, plugin)
        flat_segments = segmenter.extract_flat_segments(planner_output)

        logger.info(
            f"  > Extracted {len(flat_segments)} segments. Starting enrichment..."
        )
        enriched_segments = self.enricher.enrich_batch(flat_segments)

        items_collection = []

        # Chunk processing and relationship building
        self._process_chunks(
            enriched_segments, doc_graph, doc_hierarchy, document, items_collection
        )

        # Bottom-Up Summarization
        self._bottom_up_summarization(doc_graph, document, plugin, items_collection)

        # Merge into the global knowledge graph
        knowledge_graph = nx.compose(knowledge_graph, doc_graph)
        logger.info(f"  > {doc_name} processed successfully.")

        if not items_collection:
            logger.info("  > No items extracted.")
            return False

        with open(graph_path, "wb") as f:
            pickle.dump(knowledge_graph, f)

        logger.info("  > Starting embeddings calculation...")
        texts_to_embed = [item.text_view for item in items_collection]
        vectors_array = self.embedder.embed(texts_to_embed, show_progress_bar=True)

        metadata = []
        for item in items_collection:
            meta = item.metadata.copy()
            meta.update(
                {
                    "item_id": item.item_id,
                    "base_node_id": item.item_id,
                    "hierarchy": item.hierarchy_ref,
                    "text": item.content,
                    "doc_id": item.doc_id,
                    "doc_ref": item.doc_ref,
                }
            )
            if item.metadata:
                for key, value in item.metadata.items():
                    if key not in meta:
                        meta[key] = value
            metadata.append(meta)

        vector_store = self._get_vector_store(index_name)
        vector_store.add(vectors=vectors_array, metadata=metadata)
        logger.info(
            f"  > Vectors saved to vector store. Total vectors present: {vector_store.count()}"
        )

        try:
            self.registry_manager.register_document(
                doc_id=document.doc_id,
                index_name=index_name,
                title=doc_stem,
                source_path=location.value,
                doc_format=location.extension.lstrip("."),
                doc_type=document.doc_type,
                has_physical_copy=True,
            )
            self.registry_manager.bind_documents_to_views(
                [document.doc_id], target_views
            )
            logger.info(
                f"  > Document registered in PostgreSQL and bound to views: {[v.get('view_id') for v in target_views]}."
            )
        except Exception as e:
            logger.error(f"  > [ERROR] Failed to register document in DB: {e}")

        if self.config.enable_debug:
            self._save_debug_json(items_collection, doc_stem)

        return True

    def _process_chunks(
        self,
        enriched_segments: List[Dict],
        doc_graph: nx.DiGraph,
        doc_hierarchy,
        document,
        items_collection: list,
    ) -> None:
        """Helper to transform segments into Items and insert them into the graph as leaf nodes."""

        for seg in enriched_segments:
            raw_id = seg.get("node_id", "")
            original_node_id = (
                raw_id.split("_chunk_")[0].split("_intro")[0].split("_outro")[0]
            )

            chunk_node_id = raw_id
            if chunk_node_id == original_node_id:
                chunk_node_id = f"{chunk_node_id}_chunk_0"

            try:
                hierarchy_string = doc_hierarchy.build_hierarchy_ref(
                    original_node_id, include_document=True
                )
            except KeyError:
                hierarchy_string = f"document::Unknown<|>{seg.get('kind', 'unknown')}::{seg.get('label', 'unknown')}"

            coords = seg.get("coordinates", {})

            item = Item(
                item_id=chunk_node_id,
                item_type="text",
                doc_id=document.doc_id,
                doc_ref=str(document.location.value),
                doc_type=document.doc_type,
                content=seg["text"],
                text_view=seg.get("text_view", seg["text"]),
                hierarchy_ref=hierarchy_string,
                coordinates=coords,
                metadata=seg.get("metadata", {}),
            )

            items_collection.append(item)

            doc_graph.add_node(
                item.item_id,
                type="chunk",
                label=item.item_id,
                text=item.content,
                text_view=item.text_view,
                coordinates=coords,
            )
            doc_graph.add_edge(original_node_id, item.item_id, etype="HAS_CHILD")

    def _bottom_up_summarization(
        self, doc_graph: nx.DiGraph, document, plugin, items_collection: list
    ) -> None:
        """Helper to perform bottom-up summarization and update bounding boxes using the plugin."""

        logger.info("  > Starting Bottom-Up Summarization for current document...")
        tree_edges = [
            (u, v)
            for u, v, d in doc_graph.edges(data=True)
            if d.get("etype") == "HAS_CHILD"
        ]
        tree_graph = nx.DiGraph(tree_edges)
        tree_graph.add_nodes_from(doc_graph.nodes())

        reversed_tree = tree_graph.reverse()
        generations = list(nx.topological_generations(reversed_tree))

        for level_idx, generation in enumerate(generations):
            nodes_to_process = [
                n for n in generation if "text_view" not in doc_graph.nodes[n]
            ]

            if not nodes_to_process:
                continue

            logger.info(
                f"    > Processing Level {level_idx}: {len(nodes_to_process)} nodes concurrently..."
            )

            with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
                future_to_node = {
                    executor.submit(
                        self._summarize_single_node, n, doc_graph, document, plugin
                    ): n
                    for n in nodes_to_process
                }

                for future in concurrent.futures.as_completed(future_to_node):
                    node_id = future_to_node[future]
                    try:
                        result = future.result()
                        if result:
                            doc_graph.nodes[node_id]["text_view"] = result["text_view"]
                            if result["coordinates"]:
                                doc_graph.nodes[node_id]["coordinates"] = result[
                                    "coordinates"
                                ]

                            parent_item = Item(
                                item_id=node_id,
                                item_type="text",
                                doc_id=document.doc_id,
                                doc_ref=str(document.location.value),
                                doc_type=document.doc_type,
                                content=result["text_view"],
                                text_view=result["text_view"],
                                hierarchy_ref=result["node_label"],
                                coordinates=result["coordinates"],
                                metadata={"is_macro_node": True},
                            )
                            items_collection.append(parent_item)

                    except Exception as exc:
                        logger.error(f"      Error summarizing node {node_id}: {exc}")

    def _summarize_single_node(
        self, node_id: str, doc_graph: nx.DiGraph, document, plugin
    ) -> dict:
        """
        Isolated worker: calculates new coordinates and calls the LLM for summarization.
        Note: It only reads from the graph, it DOES NOT write to it.
        """
        node_data = doc_graph.nodes[node_id]
        children_ids = [
            v
            for u, v, d in doc_graph.out_edges(node_id, data=True)
            if d.get("etype") == "HAS_CHILD"
        ]

        bounding_coords = None
        parent_coords_dict = node_data.get("coordinates", {})

        if parent_coords_dict:
            bounding_coords = Coordinates(**parent_coords_dict)

        for child_id in children_ids:
            child_coords_raw = doc_graph.nodes[child_id].get("coordinates", {})
            if not child_coords_raw:
                continue

            child_coords = (
                Coordinates(**child_coords_raw)
                if isinstance(child_coords_raw, dict)
                else child_coords_raw
            )

            if bounding_coords is None:
                bounding_coords = child_coords
            else:
                bounding_coords = plugin.get_bounding_coordinates(
                    bounding_coords, child_coords
                )

        coords_dump = bounding_coords.model_dump() if bounding_coords else {}
        if coords_dump and "doc_format" not in coords_dump:
            coords_dump["doc_format"] = document.doc_format

        children_summaries = [
            doc_graph.nodes[cid].get("text_view")
            for cid in children_ids
            if doc_graph.nodes[cid].get("text_view")
        ]

        if not children_summaries:
            return None

        node_label = node_data.get("label", node_id)

        parent_summary = self.enricher.summarize_parent_node(
            node_label, children_summaries
        )

        return {
            "node_label": node_label,
            "text_view": parent_summary,
            "coordinates": coords_dump,
        }

    def _save_debug_json(self, items_collection: list, doc_stem: str) -> None:
        if not self.config.enable_debug:
            return

        logger.info("\n=== Saving generated Items for inspection ===")
        items_to_save = []
        for item in items_collection:
            item_data = (
                item.model_dump()
                if hasattr(item, "model_dump")
                else item.__dict__.copy()
            )

            coords = item_data.get("coordinates")
            if coords is not None:
                if hasattr(coords, "model_dump"):
                    item_data["coordinates"] = coords.model_dump()
                elif hasattr(coords, "__dict__"):
                    item_data["coordinates"] = coords.__dict__.copy()

            items_to_save.append(item_data)

        full_path = self.debug_root / f"debug_{doc_stem}_items.json"
        with open(full_path, "w", encoding="utf-8") as f:
            json.dump(items_to_save, f, indent=4, ensure_ascii=False)

    def delete_index(self, index_name: str) -> bool:
        """
        Radically deletes a physical index from the system.
        """
        logger.info(f"\n[DANGER ZONE] Starting deletion of index: {index_name}")

        with self.db_connector.get_session() as session:
            index_record = (
                session.query(KnowledgeIndex).filter_by(index_name=index_name).first()
            )

            if not index_record:
                logger.info(
                    f"Error: Index {index_name} does not exist in the relational database."
                )
                return False

            graph_path = Path(index_record.graph_path)

        # Vector Store
        try:
            logger.info(f"  > Deleting vector collection: '{index_name}'...")
            vector_store = self._get_vector_store(index_name)

            success = vector_store.delete_collection(collection_name=index_name)
            if success:
                logger.info("    Done.")
            else:
                logger.warning(
                    "    Failed to delete vector collection. It might not exist."
                )
        except Exception as e:
            logger.error(f"    Vector Store Error: {e}")

        # File System
        try:
            logger.info(f"  > Deleting graph file: '{graph_path}'...")
            if graph_path.exists():
                graph_path.unlink()
                print("    Done.")
            else:
                logger.info("    .pkl file not found on disk. Ignored.")
        except Exception as e:
            logger.error(f"    File system error: {e}")

        # PostgreSQL
        try:
            with self.db_connector.get_session() as session:
                logger.info("  > Removing Index record (cascading to associations)...")
                # Execute index deletion
                session.query(KnowledgeIndex).filter_by(index_name=index_name).delete()

                # Force a flush to make the deletion of the
                # relationships in the junction table effective before looking for orphans
                session.flush()

                # Orphan identification and removal
                logger.info("  > Checking for orphan documents...")
                # A document is an orphan if it is NOT associated with ANY index (.spaces)
                orphans = (
                    session.query(DocumentRecord)
                    .filter(~DocumentRecord.indexes.any())
                    .all()
                )

                if orphans:
                    orphan_count = len(orphans)
                    logger.info(
                        f"    found {orphan_count} orphan documents. Deleting..."
                    )
                    for doc in orphans:
                        # Deleting the document will automatically clean up
                        # 'view_documents' as well, thanks to the CASCADE on doc_id
                        session.delete(doc)
                else:
                    logger.info("    No orphan documents found.")

                session.commit()

            logger.info("Index deletion and cleanup completed successfully.")
            return True

        except Exception as e:
            error_msg = f"Critical DB Error during cleanup. Index '{index_name}' was partially deleted (Vectors/Graph removed, but DB records remain). Manual intervention required. Details: {e}"
            logger.error(f"    {error_msg}")
            raise NemantixException(error_msg) from e

    # TODO: define and implement other management functions for the knowledge base
    def delete_document(self, doc_id: str) -> bool:
        """
        Placeholder for future implementation to delete a single document across all layers.
        """
        raise NotImplementedError("To be implemented in the future!")
