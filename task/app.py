import logging

from task._constants import API_KEY
from task.chat.chat_completion_client import DialChatCompletionClient
from task.embeddings.embeddings_client import DialEmbeddingsClient
from task.embeddings.text_processor import TextProcessor, SearchMode
from task.models.message import Message
from task.models.role import Role

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)


SYSTEM_PROMPT = """
You are a helpful assistant that can answer questions about the microwave manual.
You are given a context and a user question.
You should use the context to answer the user question.
You should not answer questions that are not related to the context.
You should not answer questions that are out of the context.
You should not answer questions that are not related to the microwave manual.
"""

USER_PROMPT = """
RAG Context:
{context}

User Question:
{user_question}
"""


embeddings_client = DialEmbeddingsClient(deployment_name='text-embedding-3-small-1', api_key=API_KEY)
text_processor = TextProcessor(embeddings_client=embeddings_client, db_config={'host': '172.22.171.235','port': 5433,'database': 'vectordb','user': 'postgres','password': 'postgres'})
chat_completion_client = DialChatCompletionClient(deployment_name='gpt-5-mini-2025-08-07', api_key=API_KEY)

def run_console_chat():
    while True:
        try:
            user_question = input("User Question: ")
            if not user_question.strip():
                continue
            
            context_list = text_processor.search(
                search_mode=SearchMode.COSINE_DISTANCE,
                user_request=user_question,
                top_k=5,
                min_score=0.5,
                dimensions=1536
            )
            
            context = "\n\n".join(context_list) if context_list else "No relevant context found."
            
            messages = [
                Message(role=Role.SYSTEM, content=SYSTEM_PROMPT),
                Message(role=Role.USER, content=USER_PROMPT.format(
                    context=context,
                    user_question=user_question
                ))
            ]
            
            message = chat_completion_client.get_completion(messages=messages)
            print(message.content)
        except KeyboardInterrupt:
            print("\nExiting...")
            break
        except Exception as e:
            print(f"Error: {e}")


if __name__ == "__main__":
    try:
        print("Loading embeddings from microwave_manual.txt...")
        text_processor.process_text_file(
            file_name='task/embeddings/microwave_manual.txt',
            chunk_size=300,
            overlap=40,
            dimensions=1536,
            should_truncate=True
        )
        print("Embeddings loaded successfully!\n")
        run_console_chat()
    finally:
        text_processor.close()