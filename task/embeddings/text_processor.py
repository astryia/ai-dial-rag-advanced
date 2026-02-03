from enum import StrEnum
from typing import List, Dict, Any
import logging

import psycopg2
import psycopg2.pool

from task.embeddings.embeddings_client import DialEmbeddingsClient
from task.utils.text import chunk_text

logger = logging.getLogger(__name__)


class SearchMode(StrEnum):
    EUCLIDIAN_DISTANCE = "euclidean"
    COSINE_DISTANCE = "cosine"


class TextProcessor:
    def __init__(self, embeddings_client: DialEmbeddingsClient, db_config: Dict[str, Any]):
        self.embeddings_client = embeddings_client
        self.db_config = db_config
        self._connection_pool = None

    def _get_connection(self):
        if self._connection_pool is None:
            self._connection_pool = psycopg2.pool.SimpleConnectionPool(
                1, 10,
                host=self.db_config['host'],
                port=self.db_config['port'],
                database=self.db_config['database'],
                user=self.db_config['user'],
                password=self.db_config['password']
            )
        return self._connection_pool.getconn()

    def _return_connection(self, conn):
        if self._connection_pool:
            self._connection_pool.putconn(conn)

    def process_text_file(self, file_name: str, chunk_size: int, overlap: int, dimensions: int, should_truncate: bool) -> None:
        conn = None
        cursor = None
        try:
            conn = self._get_connection()
            cursor = conn.cursor()
            
            if should_truncate:
                cursor.execute("TRUNCATE TABLE vectors;")
                conn.commit()

            with open(file_name, 'r', encoding='utf-8') as file:
                content = file.read()
                chunks = chunk_text(content, chunk_size, overlap)
                
                for chunk in chunks:
                    embedding_dict = self.embeddings_client.get_embeddings(chunk, dimensions)
                    embedding = embedding_dict[0]
                    embedding_str = str(embedding)
                    cursor.execute(
                        "INSERT INTO vectors (document_name, text, embedding) VALUES (%s, %s, %s::vector);",
                        (file_name, chunk, embedding_str)
                    )
                
                conn.commit()
        except (psycopg2.Error, IOError, ValueError) as e:
            if conn:
                conn.rollback()
            logger.error(f"Error processing text file {file_name}: {e}", exc_info=True)
            raise
        finally:
            if cursor:
                cursor.close()
            if conn:
                self._return_connection(conn)

    def search(self, search_mode: SearchMode, user_request: str, top_k: int, min_score: float, dimensions: int) -> List[str]:
        logger.info(f"Vector search started - search_mode: {search_mode}, top_k: {top_k}, min_score: {min_score}, dimensions: {dimensions}")
        logger.info(f"User request: {user_request[:100]}{'...' if len(user_request) > 100 else ''}")
        
        conn = None
        cursor = None
        try:
            if search_mode == SearchMode.EUCLIDIAN_DISTANCE:
                distance_operator = "<->"
            elif search_mode == SearchMode.COSINE_DISTANCE:
                distance_operator = "<=>"
            else:
                raise ValueError(f"Invalid search mode: {search_mode}")

            embedding_dict = self.embeddings_client.get_embeddings(user_request, dimensions)
            embedding = embedding_dict[0]
            embedding_str = str(embedding)
            logger.debug(f"Generated embedding vector with {len(embedding)} dimensions")

            conn = self._get_connection()
            cursor = conn.cursor()
            query = f"""
                SELECT text 
                FROM vectors 
                WHERE embedding {distance_operator} %s::vector < %s 
                ORDER BY embedding {distance_operator} %s::vector 
                LIMIT %s
            """
            logger.debug(f"Executing query with distance operator: {distance_operator}")
            cursor.execute(query, (embedding_str, min_score, embedding_str, top_k))
            results = cursor.fetchall()
            result_texts = [row[0] for row in results]
            
            logger.info(f"Vector search completed - found {len(result_texts)} results")
            for i, text in enumerate(result_texts, 1):
                preview = text[:100].replace('\n', ' ') + ('...' if len(text) > 100 else '')
                logger.info(f"Result {i}/{len(result_texts)}: {preview}")
            
            return result_texts
        except (psycopg2.Error, ValueError) as e:
            logger.error(f"Error during search: {e}", exc_info=True)
            raise
        finally:
            if cursor:
                cursor.close()
            if conn:
                self._return_connection(conn)

    def close(self):
        if self._connection_pool:
            self._connection_pool.closeall()
            self._connection_pool = None
