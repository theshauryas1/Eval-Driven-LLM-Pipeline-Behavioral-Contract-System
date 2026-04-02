import os
import time
from typing import List, Dict, Any
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage
from .schema import TestCase

def execute_tests(tests: List[TestCase], models: List[str] = ["llama-3.3-70b-versatile"]) -> Dict[str, Any]:
    """Execute standard TestCases against the specified models via Groq API."""
    
    groq_api_key = os.environ.get("GROQ_API_KEY")
    if not groq_api_key:
        raise ValueError("GROQ_API_KEY environment variable is not set. Required for Execution Engine.")

    results = {}

    for model_name in models:
        print(f"\n🚀 Running Execution Engine for model: {model_name}")
        chat = ChatGroq(temperature=0, model_name=model_name, groq_api_key=groq_api_key)
        
        model_results = []
        for case in tests:
            print(f"   [Input] {case.test_name}: {case.input[:50]}...")
            
            start_time = time.time()
            try:
                response = chat.invoke([HumanMessage(content=case.input)])
                latency = time.time() - start_time
                output_text = response.content
            except Exception as e:
                latency = time.time() - start_time
                output_text = f"ERROR: {str(e)}"
                
            print(f"   [Output] latency={latency:.2f}s | response length={len(output_text)}")
            
            model_results.append({
                "test_name": case.test_name,
                "input": case.input,
                "output": output_text,
                "latency_s": latency,
                "expected": case.expected.model_dump(),
                "constraints": [c.model_dump() for c in case.constraints]
            })
            
        results[model_name] = model_results
        
    return results
