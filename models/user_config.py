# from models.dummy_model import DummyModel
# from models.rag_llama_baseline import RAGModel
# from models.rag_llama_baseline_custom import CustomRAGModel
# from models.rag_llama_lcw_dev import LcwDevModel
from models.rag_llama_baseline_batch import BatchRAGModel
from models.rag_llama_batch_with_calculator import BatchRAGModelWithCalculator
from models.rag_llama_baseline_batch_with_reasoning import BatchReasoningRAGModel
from models.rag_llama_baseline_batch_custom import BatchRAGModelCustom
from functools import partial

# UserModel = BatchRAGModel
UserModel = BatchRAGModelWithCalculator
# UserModel = BatchReasoningRAGModel
# UserModel = BatchRAGModelCustom

##############################################
# UserModel = partial(CustomRAGModel, model="llama3-8b-groq")
# UserModel = DummyModel
# UserModel = RAGModel
# UserModel = LcwDevModel
# Uncomment the lines below to use the Vanilla LLAMA baseline
# from models.vanilla_llama_baseline import ChatModel
# UserModel = ChatModel


# Uncomment the lines below to use the RAG LLAMA baseline
# from models.rag_llama_baseline import RAGModel
# UserModel = RAGModel
