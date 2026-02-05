from app.config.settings import config
from langchain_openai import OpenAIEmbeddings
from tenacity import retry, stop_after_attempt, wait_random_exponential
from loguru import logger

Embedding_Model = config.OPENAI_EMBEDDING_MODEL
Embedding_Dim = config.OPENAI_EMBEDDING_DIM

embeddings_model = OpenAIEmbeddings(
    model=Embedding_Model,
    api_key=config.OPENAI_API_KEY
)

@retry(wait=wait_random_exponential(min=1, max=2), stop=stop_after_attempt(2))
def get_embedding(data) -> list:

    if isinstance(data, str):
        result = embeddings_model.embed_query(data)
        logger.info(f'Embedding dimensions: {len(result)}')
        return [result]

    if isinstance(data, list):
        result = embeddings_model.embed_documents(data)
        logger.info(f'Embedding dimensions: {len(result[0])}')
        return result

    else:
        raise ValueError("Only string or list is supported currently")


# def main():
#     docs = ['Machine learning is a subset of AI.', 'hi hi how are you']
#     embeddings = get_embedding(docs)
#     print(len(embeddings))

# if __name__ == "__main__":
#     main()
