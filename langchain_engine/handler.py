import asyncio
from typing import List

from fastapi_poe import PoeHandler
from fastapi_poe.types import ProtocolMessage, QueryRequest

from langchain.callbacks import AsyncIteratorCallbackHandler
from langchain.callbacks.manager import AsyncCallbackManager
from langchain.chains import ConversationChain, ConversationalRetrievalChain
from langchain.chains.llm import LLMChain
from langchain.chains.question_answering import load_qa_chain
from langchain.chat_models import ChatOpenAI
from langchain.llms import OpenAI
from langchain.memory import ConversationBufferMemory
from langchain.prompts import (
    ChatPromptTemplate,
    HumanMessagePromptTemplate,
    MessagesPlaceholder,
    SystemMessagePromptTemplate
)
from langchain.schema import AIMessage, BaseMessage, ChatMessage, HumanMessage, SystemMessage
from langchain.document_loaders import TextLoader
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain.chains.conversational_retrieval.prompts import CONDENSE_QUESTION_PROMPT, QA_PROMPT


def convert_messages(poe_messages: List[ProtocolMessage]) -> List[BaseMessage]:
    messages = []
    for msg in poe_messages:
        if msg.role == "user":
            messages.append(HumanMessage(content=msg.content))
        elif msg.role == "assistant":
            messages.append(AIMessage(content=msg.content))
        elif msg.role == "system":
            messages.append(SystemMessage(content=msg.content))
        else:
            messages.append(ChatMessage(content=msg.content, role=msg.role))
    return messages


CHAT_PROMPT_TEMPLATE = ChatPromptTemplate.from_messages([
    SystemMessagePromptTemplate.from_template(
        "The following is a friendly and helpful AI assistant. If the AI doesn't know the answer, it will say so clearly."
    ),
    MessagesPlaceholder(variable_name="history"),
    HumanMessagePromptTemplate.from_template("{input}")
])


class LLMOrchestrator(PoeHandler):
    """Base handler that streams response from a ChatOpenAI model."""

    async def get_response(self, query: QueryRequest):
        callback_handler = AsyncIteratorCallbackHandler()
        model = ChatOpenAI(callbacks=[callback_handler], streaming=True)
        messages = convert_messages(query.query)
        run = asyncio.create_task(model.agenerate([messages]))

        async for token in callback_handler.aiter():
            yield self.text_event(token)

        await run


class ConversationalContextEngine(PoeHandler):
    """Conversational memory-enabled handler using LangChain's ConversationChain."""

    def __init__(self):
        self.sessions = {}

    async def get_response(self, query: QueryRequest):
        callback_handler = AsyncIteratorCallbackHandler()
        model = ChatOpenAI(callbacks=[callback_handler], streaming=True)

        memory = self.sessions.get(query.conversation_id)
        if memory is None:
            memory = ConversationBufferMemory(return_messages=True)
            self.sessions[query.conversation_id] = memory

        chain = ConversationChain(llm=model, prompt=CHAT_PROMPT_TEMPLATE, memory=memory)
        input_text = query.query[-1].content
        run = asyncio.create_task(chain.arun(input=input_text))

        async for token in callback_handler.aiter():
            yield self.text_event(token)

        await run


class ContextAwareRetrievalEngine(PoeHandler):
    """Retrieval-augmented handler for answering questions from documents."""

    def __init__(self):
        self.chat_logs = {}

        loader = TextLoader("assistant_contex.txt")
        docs = loader.load()
        splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
        chunks = splitter.split_documents(docs)

        self.vectorstore = Chroma.from_documents(chunks, OpenAIEmbeddings())

    async def get_response(self, query: QueryRequest):
        callback_handler = AsyncIteratorCallbackHandler()
        callback_manager = AsyncCallbackManager([callback_handler])

        llm_condense = OpenAI(temperature=0)
        llm_stream = OpenAI(callback_manager=callback_manager, streaming=True)

        history = self.chat_logs.get(query.conversation_id, [])
        self.chat_logs[query.conversation_id] = history

        input_text = query.query[-1].content

        question_chain = LLMChain(llm=llm_condense, prompt=CONDENSE_QUESTION_PROMPT)
        doc_chain = load_qa_chain(llm_stream, chain_type="stuff", prompt=QA_PROMPT, callback_manager=callback_manager)

        chain = ConversationalRetrievalChain(
            combine_docs_chain=doc_chain,
            question_generator=question_chain,
            retriever=self.vectorstore.as_retriever(),
        )

        run = asyncio.create_task(chain.acall({"question": input_text, "chat_history": history}))

        async for token in callback_handler.aiter():
            yield self.text_event(token)

        result = await run
        history.append((input_text, result["answer"]))
