"""RAGAS eval setup and metric wrappers
   Configures 3 metrics Faithfulness,
   Precision, Recall"""

from typing import List, Dict, Any, Optional
from langchain_groq import ChatGroq
from ragas.dataset_schema import SingleTurnSample
from ragas.llms import LangchainLLMWrapper
from ragas.metrics._faithfulness import Faithfulness
from ragas.metrics._context_precision import LLMContextPrecisionWithReference
from ragas.metrics._context_recall import LLMContextRecall

from src.common.logging import get_logger
from src.config.settings import Settings, get_settings

logger=get_logger("Umbrella.eval.ragas_setup")

class RagasEvaluator:

    def __init__(
        self,
        settings: Optional[Settings] = None,
        llm: Optional[LangchainLLMWrapper] = None,
    ):
        self.settings = settings or get_settings()

        if llm is not None:
            self.llm=llm
        else:
            api_key = self.settings.groq_api_key
            if not api_key:
                raise ValueError("")
            
            chat = ChatGroq(
                model=self.settings.eval_model,
                api_key=api_key,
                temperature=0.0,
                max_retries=2,
            )
            """Wrapping langchain llm 'Chat' so that ragas can use the llm
               for generating evaluation repsonse beacause ragas cannot directly
               access llm from langchain's interface but through its own"""
            self.llm = LangchainLLMWrapper(chat)
        
        self.faithfulness_metric= Faithfulness(llm=self.llm)
        self.context_precision_metric= LLMContextPrecisionWithReference(llm=self.llm)
        self.context_recall_metric= LLMContextRecall(llm=self.llm)
      
    def score_faithfulness(self, question:str, response:str,
        retrieved_contexts: List[str],)-> Optional[float]:
        """
        Score faithfulness of an answer against retrieved contexts.
        Returns None if response was a refusal or if contexts are empty.
        """

        if not retrieved_contexts or response.startswith("REFUSED:"):
            logger.debug("faithfulness score skipped due to refusal or empty context")
            return None
        
        sample = SingleTurnSample(
            user_input=question,
            response=response,
            retrieved_contexts=retrieved_contexts,
        )
        try:
            score= self.faithfulness_metric.single_turn_score(sample)
            return float(score)
        except Exception as e:
            logger.warning(f"Error evaluating faithnulness: {e}")
            return None
    
    def score_context_precision(self, question:str, retrieved_contexts: List[str],
        reference:str,)-> Optional[float]:
        if not retrieved_contexts or not reference:
            return None
        
        sample= SingleTurnSample(
            user_input=question,
            retrieved_contexts=retrieved_contexts,
            reference=reference,
        )
        try:
            score =self.context_precision_metric.single_turn_score(sample)
            return float(score)
        except Exception as e:
            logger.warning(f"Error evaluating precision: {e}")
            return None
    
    def score_context_recall(self, question:str, retrieved_contexts: List[str],
        reference:str,) -> Optional[float]:
        if not retrieved_contexts or not reference:
            return None
        
        sample= SingleTurnSample(
            user_input=question,
            retrieved_contexts=retrieved_contexts,
            reference=reference,
        )
        try:
            score= self.context_recall_metric.single_turn_score(sample)
            return float(score)
        except Exception as e:
            logger.warning(f"Error evaluating Recall: {e}")
            return None
    
    def evaluate_sample(
        self, question:str, response:str, retrieved_contexts: List[str],
        reference: Optional[str]= None, 
    )-> Dict[str, Optional[float]]:
        
        faithfulness=self.score_faithfulness(question, response, retrieved_contexts)
        context_precision=self.score_context_precision(question, retrieved_contexts, reference
        if reference
        else None
        )
        context_recall=self.score_context_recall(question, retrieved_contexts, reference
        if reference
        else None
        )

        return {
            "faithfulness": faithfulness,
            "context_precision": context_precision,
            "context_recall": context_recall,
        }

 
