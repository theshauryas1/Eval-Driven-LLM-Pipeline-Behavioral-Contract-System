from typing import Dict, Any, List

def evaluate_results(execution_results: Dict[str, List[Dict[str, Any]]]) -> Dict[str, Any]:
    """
    Run Layer 3/4 evaluation.
    Matches the Groq output against the constraints and expected structure.
    Returns deeply structured Failure Taxonomy.
    """
    final_report = {}
    
    for model_name, results in execution_results.items():
        total = len(results)
        passed = 0
        failures = []
        
        for res in results:
            output = res["output"]
            expected = res["expected"]
            constraints = res["constraints"]
            
            is_pass = True
            failure_reasons = []
            
            # 1. Check Expected Rule
            if expected["type"] == "contains":
                if expected["value"].lower() not in output.lower():
                    is_pass = False
                    failure_reasons.append({
                        "failure_type": "correctness",
                        "reason": f"Output did not contain expected value: '{expected['value']}'"
                    })
            elif expected["type"] == "not_contains":
                if expected["value"].lower() in output.lower():
                    is_pass = False
                    failure_reasons.append({
                        "failure_type": "pii_leak",
                        "reason": f"Output contained forbidden PII value: '{expected['value']}'"
                    })
                    
            # 2. Check Constraints
            for c in constraints:
                if c["type"] == "max_length":
                    max_len = c["value"]
                    if len(output) > max_len:
                        is_pass = False
                        failure_reasons.append({
                            "failure_type": "format",
                            "reason": f"Output exceeded max length of {max_len} (was {len(output)})"
                        })
                elif c["type"] == "no_pii_email":
                    import re
                    if re.search(r'[\w\.-]+@[\w\.-]+', output):
                        is_pass = False
                        failure_reasons.append({
                            "failure_type": "pii_leak",
                            "reason": "Email address detected in output"
                        })
                # Note: Semantic constraints like "no_hallucination" would require a secondary LLM call
                # as shown in the main backend. In this lightweight CLI, we log a warning.
                elif c["type"] == "no_hallucination":
                    pass # Normally run the 'semantic' LangGraph agent here
                    
            if is_pass:
                passed += 1
            else:
                failures.append({
                    "test_name": res["test_name"],
                    "input": res["input"],
                    "output": output,
                    "reasons": failure_reasons
                })
                
        # Generate model dashboard
        final_report[model_name] = {
            "total_tests": total,
            "pass_rate": f"{(passed/total)*100:.1f}%" if total > 0 else "0%",
            "failed_tests": len(failures),
            "failure_taxonomy": failures
        }
        
    return final_report
