import os
import copy
from pathlib import Path
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage
from .schema import parse_test_suite
from .execution import execute_tests
from .evaluators import evaluate_results

def repair_loop(tests_dir: Path):
    """
    Layer 5: Auto-Repair Loop
    Finds failing tests and asks an LLM to rewrite the input prompt to fix the failures.
    """
    groq_api_key = os.environ.get("GROQ_API_KEY")
    if not groq_api_key:
        print("GROQ_API_KEY required for Auto-Repair.")
        return
        
    chat = ChatGroq(temperature=0.7, model_name="llama-3.3-70b-versatile", groq_api_key=groq_api_key)
    
    files = [tests_dir] if tests_dir.is_file() else list(tests_dir.glob('*.json'))
    
    for f in files:
        tests = parse_test_suite(f)
        
        # 1. Run baseline
        baseline = execute_tests(tests, models=["llama-3.3-70b-versatile"])
        report = evaluate_results(baseline)
        
        failures = report["llama-3.3-70b-versatile"].get("failure_taxonomy", [])
        if not failures:
            print(f"✅ All tests in {f.name} passed! No repair needed.")
            continue
            
        print(f"⚠️ Found {len(failures)} failures in {f.name}. Initiating auto-repair...")
        
        # 2. Repair each failure
        for failure in failures:
            original_input = failure["input"]
            reasons = [r["reason"] for r in failure["reasons"]]
            
            prompt = f"""
            I have an AI system prompt that is failing its behavioral contracts.
            Prompt: "{original_input}"
            
            Failures observed:
            {chr(10).join(reasons)}
            
            Rewrite this prompt to ensure the AI strictly follows these rules without changing the user's ultimate goal.
            Return ONLY the rewritten prompt string. No conversational filler.
            """
            
            try:
                new_prompt_msg = chat.invoke([HumanMessage(content=prompt)])
                new_prompt = new_prompt_msg.content.strip().strip('"\'')
                print(f"   [Repair Strategy Generated]")
                print(f"     Old: {original_input[:60]}...")
                print(f"     New: {new_prompt[:60]}...")
                
                # 3. Verify the fix
                test_case = next(t for t in tests if t.test_name == failure["test_name"])
                repaired_case = copy.deepcopy(test_case)
                repaired_case.input = new_prompt
                
                verification_run = execute_tests([repaired_case], models=["llama-3.3-70b-versatile"])
                verification_report = evaluate_results(verification_run)
                
                if len(verification_report["llama-3.3-70b-versatile"]["failure_taxonomy"]) == 0:
                    print(f"   🌟 REPAIR SUCCESSFUL! The new prompt passes all contracts.")
                    # Ideally we would save this back to the JSON file
                else:
                    new_fail_reasons = verification_report["llama-3.3-70b-versatile"]["failure_taxonomy"][0]["reasons"]
                    print(f"   ❌ REPAIR FAILED. Failed again with: {[r['reason'] for r in new_fail_reasons]}")
                    
            except Exception as e:
                print(f"   Error during repair: {e}")
