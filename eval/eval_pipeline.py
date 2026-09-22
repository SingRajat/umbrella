""" it's an offline benchmarking runner for RAG. it takes the evaluation dataset and run it agaisnt the RAG pipeline
and measure accuracy, faithfullnes, latency and retrieved chunks"""

from src.storage.chroma import _vector_store_instance
from numpy.random import seed
import hashlib, json, time, uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from src.api.schemas import QueryResponse, RefusalResponse
from src.chains.query_chain import run_query_pipeline
from src.common.logging import get_logger
from src.config.settings import get_settings
from src.query.retriever import retrieve_chunks
from src.storage.chroma import get_vector_store 

from eval.ragas_setup import RagasEvaluator

logger= get_logger("umbrella.eval.pipeline")

RESULTS_DIR= Path("eval/results") ##location at which results will be stored

def compute_config_hash(settings) ->str:
    config_dict ={
        "chunk_size": settings.chunk_size,
        "chunk_overlap": settings.chunk_overlap,
        "top_k":settings.top_k,
        "similarity_threshold": settings.similarity_threshold,
        "groq_model": settings.groq_model,
        "temperature": settings.temperature,
    }
    encoded = json.dumps(config_dict, sort_keys=True).encode("utf-8") 
    """converts the dictionary into a json string and convert json strings into bytes
       because hash functions only operate on bytes
    """
    return hashlib.md5(encoded).hexdigest()[:8] ## calculates MD5 hash object and converts to 32 hexadecimal character

def compute_percentile(values: List[float], p: float) -> float:
    ## Calculate the p-th percentile of a list of floats
    if not values:
        return 0.0
    sorted_vals= sorted(values)
    k =(len(sorted_vals)- 1) * (p/100.0)
    floor_idx= int(k)
    ceil_idx= floor_idx+1
    if ceil_idx < len(sorted_vals):
        return round(sorted_vals[floor_idx] + (k - floor_idx) * (sorted_vals[ceil_idx] - sorted_vals[floor_idx]), 2)
    return round(sorted_vals[floor_idx], 2)


class EvaluationPipeline:
    def __init__(self, results_dir: Path = RESULTS_DIR, vector_store=None,
                 evaluator: Optional[RagasEvaluator]= None):
        self.results_dir = Path(results_dir)
        self.results_dir.mkdir(parents=True, exist_ok=True)
        self.settings = get_settings()
        self.vector_store = vector_store or get_vector_store()
        self.evaluator=evaluator

    ## runs one test question through your RAG pipeline and return all evalaution info 
    def run_single_sample(
        self,
        question: str,
        ground_truth: Optional[str] = None,
        doc_id: Optional[str] = None,
    ) -> Dict[str, Any]:

        start_time = time.perf_counter() ## to calculate latency

        ## run retrieval pipeline
        retrieval_start= time.perf_counter()
        retrieved = retrieve_chunks(
            query_text=question,
            doc_id=doc_id,
            vector_store=self.vector_store,
        )

        retrieval_latency_ms= round((time.perf_counter() - retrieval_start)* 1000, 2)

        ## call the main rag pipeline
        generation_start=time.perf_counter()
        query_response = run_query_pipeline(
            query=question,
            doc_id=doc_id,
            vector_store=self.vector_store,
        )

        generation_latency_ms= round((time.perf_counter() - generation_start)* 1000, 2)

        ## Calculate latency
        latency_ms = round(
            (time.perf_counter() - start_time) * 1000, 2
        )

        ## context format needed for evaluation
        contexts = [chunk.text for chunk in retrieved]

        if isinstance(query_response, QueryResponse):
            answer = query_response.answer
            citations = [c.model_dump() for c in query_response.citations]
            """model_dump is used to convert pydantic model object 
            into python dictionary as evaluation results are stored in json format""" 

        elif isinstance(query_response, RefusalResponse):
            answer = f"REFUSED: {query_response.reason}"
            citations = []

        else:
             answer = getattr(query_response, "answer", str(query_response))

        token_usage= getattr(query_response, "token_usage", None) or {
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0,
        }

        faithfulness_score = None
        precision_score= None
        recall_score= None

        if self.evaluator is not None:
            faithfulness_score = self.evaluator.score_faithfulness(
                question=question,
                response=answer,
                retrieved_contexts=contexts,
            )
            if ground_truth:
                
                precision_score= self.evaluator.score_context_precision(
                        question=question,
                        retrieved_contexts=contexts,
                        reference=ground_truth,
                )
                recall_score= self.evaluator.score_context_recall(
                        question=question,
                        retrieved_contexts=contexts,
                        reference=ground_truth,
                )

        return {
            "question": question,
            "ground_truth": ground_truth or "",
            "answer": answer,
            "contexts": contexts,
            "status": query_response.status,
            "citations": citations,
            "token_usage": token_usage,
            "faithfulness": faithfulness_score,
            "context_precision": precision_score,
            "context_recall": recall_score,
            "retrieved_chunk_count": len(retrieved),
            "latency": latency_ms,
            "latency_ms": latency_ms,
            "retrieval_latency_ms": retrieval_latency_ms,
            "generation_latency_ms": generation_latency_ms,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }

    def run_dataset(
        self,
        samples: List[Dict[str, Any]],
        run_name: str = "eval_run",
    ) -> Dict[str, Any]:

        ## generates unique identifier for each run
        run_id = str(uuid.uuid4())

        ## records the RAG settings used in certain run
        config_hash = compute_config_hash(self.settings)

        start_time = time.perf_counter()

        logger.info(
            f"starting evaluation run '{run_name}' "
            f"[{run_id}] with {len(samples)} samples"
        )

        ## store the results of each test question
        sample_results: List[Dict[str, Any]] = []

        for idx, sample in enumerate(samples, start=1):
            ## fetches values from the sample while tracking progress number
            question = sample.get("question", "")
            ground_truth = sample.get("ground_truth")
            doc_id = sample.get("doc_id")

            logger.info(f"evaluating sample {idx}/{len(samples)}")

            result = self.run_single_sample(
                question=question,
                ground_truth=ground_truth,
                doc_id=doc_id,
            )
            ## Store the results
            sample_results.append(result)

        answered_count = sum(1 for r in sample_results if r.get("status") == "answered")
        refused_count = sum(1 for r in sample_results if r.get("status") == "refused")
        total_duration_s = round(time.perf_counter() - start_time, 2)
        avg_latency = (
            round(sum(r.get("latency", 0.0) for r in sample_results) / len(sample_results), 2)
            if sample_results
            else 0.0
        )

        total_latencies= [r.get("latency_ms", 0.0) for r in sample_results]
        retrieval_latencies = [r.get("retrieval_latency_ms", 0.0) for r in sample_results]
        generation_latencies = [r.get("generation_latency_ms", 0.0) for r in sample_results]

        p50_latency = compute_percentile(total_latencies, 50)
        p95_latency = compute_percentile(total_latencies, 95)
        p99_latency = compute_percentile(total_latencies, 99)

        avg_retrieval_latency = (
            round(sum(retrieval_latencies) / len(retrieval_latencies), 2)
            if retrieval_latencies
            else 0.0
        )

        p50_retrieval_latency = compute_percentile(retrieval_latencies, 50)
        p95_retrieval_latency = compute_percentile(retrieval_latencies, 95)
        p99_retrieval_latency = compute_percentile(retrieval_latencies, 99)

        avg_generation_latency = (
            round(sum(generation_latencies) / len(generation_latencies), 2)
            if generation_latencies
            else 0.0
        )

        p50_generation_latency = compute_percentile(generation_latencies, 50)
        p95_generation_latency = compute_percentile(generation_latencies, 95)
        p99_generation_latency = compute_percentile(generation_latencies, 99)

        total_prompt_tokens =sum((r.get("token_usage") or {}).get("prompt_tokens", 0) for r in sample_results)
        total_completion_tokens =sum((r.get("token_usage") or {}).get("completion_tokens", 0) for r in sample_results)
        total_tokens =sum((r.get("token_usage") or {}).get("total_tokens, 0") for r in sample_results)
        avg_tokens_per_query=(
            round(total_tokens/ len(sample_results), 2)
            if sample_results
            else 0.0
        )

        valid_faithfulness = [
            r["faithfulness"] for r in sample_results if r.get("faithfulness") is not None
        ]
        avg_faithfulness = (
            round(sum(valid_faithfulness) / len(valid_faithfulness), 4)
            if valid_faithfulness
            else None
        )

        valid_precisions= [
            r["context_precision"] for r in sample_results if r.get("context_precision")
            is not None
        ]
        avg_context_precision= (
            round(sum(valid_precisions)/ len(valid_precisions), 4)
            if valid_precisions
            else None
        )

        valid_recalls= [
            r["context_recall"] for r in sample_results if r.get("context_recall")
            is not None
        ]

        avg_recall=(
            round(sum(valid_recalls)/len(valid_recalls), 4)
            if valid_recalls
            else None
        )

        timestamp_str = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        filename = f"{run_name}_{config_hash}_{timestamp_str}.json"
        output_file = self.results_dir / filename
        output_data = {
            "summary": {
                "run_id": run_id,
                "run_name": run_name,
                "config_hash": config_hash,
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "total_samples": len(samples),
                "answered_count": answered_count,
                "refused_count": refused_count,
                 "total_prompt_tokens": total_prompt_tokens,
                "total_completion_tokens": total_completion_tokens,
                "total_tokens": total_tokens,
                "avg_tokens_per_query": avg_tokens_per_query,
                "avg_latency_ms": avg_latency,
                "avg_latency_ms": avg_latency,
                "p50_latency_ms": p50_latency,
                "p95_latency_ms": p95_latency,
                "p99_latency_ms": p99_latency,
                "avg_retrieval_latency_ms": avg_retrieval_latency,
                "p50_retrieval_latency_ms": p50_retrieval_latency,
                "p95_retrieval_latency_ms": p95_retrieval_latency,
                "p99_retrieval_latency_ms": p99_retrieval_latency,
                "avg_generation_latency_ms": avg_generation_latency,
                "p50_generation_latency_ms": p50_generation_latency,
                "p95_generation_latency_ms": p95_generation_latency,
                "p99_generation_latency_ms": p99_generation_latency,
                "total_duration_s": total_duration_s,
                "avg_faithfulness": avg_faithfulness,
                "avg_context_precision": avg_context_precision,
                "avg_context_recall": avg_recall,
                "settings": {
                    "chunk_size": self.settings.chunk_size,
                    "chunk_overlap": self.settings.chunk_overlap,
                    "top_k": self.settings.top_k,
                    "similarity_threshold": self.settings.similarity_threshold,
                    "groq_model": self.settings.groq_model,
                    "temperature": self.settings.temperature,
                },
            },
            "results": sample_results,
        }
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(output_data, f, indent=2)
        logger.info(f"Evaluation run saved to {output_file}")
        return output_data

    @staticmethod
    def load_dataset(file_path: Union[str, Path]) -> List[Dict[str, Any]]:
        ## Load Golden evalaution dataset

        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"Evaluation dataset not found: {path}")
        if path.suffix == ".jsonl":
            samples = []
            with open(path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line:
                        samples.append(json.loads(line))
            return samples
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, list):
            return data
        elif isinstance(data, dict) and "samples" in data:
            return data["samples"]
        else:
            raise ValueError(
                f"Unsupported dataset format in {path}. Expected list or dict with 'samples' key."
            )

if __name__ == "__main__":
    pipeline = EvaluationPipeline()
    config_hash = compute_config_hash(pipeline.settings)
    print(f"EvaluationPipeline initialized successfully. Active config hash: {config_hash}")
