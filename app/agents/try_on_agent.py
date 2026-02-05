from app.config.settings import config
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser, JsonOutputParser
from tenacity import retry, stop_after_attempt, wait_random_exponential
from loguru import logger

# Models
primary_llm = ChatOpenAI(
    model=config.OPENAI_MODEL,
    api_key=config.OPENAI_API_KEY,
    temperature=0
)

secondary_llm = ChatOpenAI(
    model=config.OPENAI_MODEL,
    api_key=config.OPENAI_API_KEY,
    temperature=0
)

def _build_chain(system_prompt: str, json_mode: bool = False, llm=None):
    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", "{input}")
    ])

    parser = JsonOutputParser() if json_mode else StrOutputParser()
    return prompt | llm | parser


@retry(wait=wait_random_exponential(min=1, max=2), stop=stop_after_attempt(2))
def call_llm_robust(system_prompt: str, user_prompt: str, json_mode: bool = False):
    try:
        try:
            chain = _build_chain(system_prompt, json_mode, primary_llm)
            return chain.invoke({"input": user_prompt})
        except Exception:
            chain = _build_chain(system_prompt, json_mode, secondary_llm)
            return chain.invoke({"input": user_prompt})

    except Exception as e:
        logger.info(f"Error calling llm {e}")
        raise e
