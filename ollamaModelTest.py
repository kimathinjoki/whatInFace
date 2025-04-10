import pandas as pd
import os
from pathlib import Path
import re
import time
from datetime import datetime
import random
import json
from tqdm import tqdm
import requests
import subprocess

# Define the models to test - make sure these are pulled first with: ollama pull <model>
MODELS = [
    "llama3:8b",
    "codellama:7b",
    "gemma3:1b", 
    "gemma3:4b",
    "phi3:mini",
    "mistral:7b",      
    "qwen:1.5-7b",     
    "qwen:2-7b",        
    "deepseek-r1:7b",  
]

# Function to verify models are available
def check_models_available(models):
    """Check if the required models are available in Ollama"""
    try:
        # Run the ollama list command
        result = subprocess.run(['ollama', 'list'], capture_output=True, text=True)
        available_models = result.stdout.strip()
        
        print("Available models:")
        print(available_models)
        print("\nChecking for required models...")
        
        missing_models = []
        for model in models:
            if model not in available_models:
                missing_models.append(model)
        
        if missing_models:
            print(f"WARNING: The following models need to be pulled first: {', '.join(missing_models)}")
            print("You can pull them with: ollama pull <model>")
            
            should_continue = input("Would you like to continue anyway? (yes/no): ")
            if should_continue.lower() != 'yes':
                print("Exiting. Please pull the missing models and run again.")
                return False
        else:
            print("All required models are available.")
        
        return True
    except Exception as e:
        print(f"Error checking models: {e}")
        print("Please ensure Ollama is installed and running.")
        return False

# Path to the input file
input_file_path = './source_doc/clean_filtered_summary.csv'

def create_prompt(row):
    prompt = f"""You are a loan officer at a cash loan company evaluating loan application for applicants. Based on the following information, decide whether to approve or reject this loan application. Consider the likelihood of repayment based on the applicant's profile.

            Applicant Information:
            - Age: {row['age']} years old
            - Gender: {'Male' if row['gender'] == 1 else 'Female'}
            - Career: {row['career']}
            - Location: {row['city_u']} ({row['best_u']})

            Loan Information:
            - Loan Amount: {row['loan_amount_k'] * 1000} RMB
            - Loan Duration: {row['loanday']} days
            - Multi-period Loan: {'Yes' if row['multiperiod'] == 1 else 'No'}
            - Interest Rate: {row['overall_rate'] * 100:.2f}%

            Credit Information:
            - Credit Score the credit score evaluated by Alipay, based on five dimensions of information
            such as identity, performance, history, relationships, and behavior: {row['zmscore']}
            - Average Monthly Phone Bill: {row['avgbill']} RMB
            - Number of Loan Applications in Past Month from other loan providers (Please note that this variable only indicates the number of
            applications, not representing approval, or loan amount the person needs to repay. The average is 20 times.): {row['action1mon']}

            Contact Information Provided:
            - Family Member Contact: {'Yes' if row['family'] == 1 else 'No'}
            - Workplace Contact: {'Yes' if row['firm'] == 1 else 'No'}
            - Friend Contact: {'Yes' if row['friend'] == 1 else 'No'}

            Make your decision based on these factors and your knowledge of lending practices. Your response should be structured as follows:

            First, explain your reasoning process considering the various risk factors and positive indicators.
            Second, give your final decision: "APPROVE" or "REJECT".
            """
    return prompt

def query_ollama(prompt, model_name):
    """Send a prompt to Ollama and get the response"""
    start_time = time.time()
    url = "http://localhost:11434/api/generate"
    
    data = {
        "model": model_name,
        "prompt": prompt,
        "stream": False
    }
    
    try:
        response = requests.post(url, json=data)
        response.raise_for_status()
        result = response.json()
        end_time = time.time()
        
        # Calculate metrics
        processing_time = end_time - start_time
        response_text = result.get("response", "")
        
        return {
            "text": response_text,
            "processing_time": processing_time,
            "tokens_per_second": result.get("eval_count", 0) / processing_time if processing_time > 0 else 0
        }
    except Exception as e:
        print(f"Error querying Ollama: {e}")
        return {
            "text": f"ERROR: {str(e)}",
            "processing_time": 0,
            "tokens_per_second": 0
        }

def benchmark_models(file_path, models, sample_size=20):
    """Benchmark multiple Ollama models using a sample of data"""
    # Load the input file
    input_df = pd.read_csv(file_path)
    
    # Take a random sample for benchmarking
    if len(input_df) > sample_size:
        sample_df = input_df.sample(sample_size, random_state=42)
    else:
        sample_df = input_df.copy()
    
    # Create the results directory
    results_dir = "./ollama_benchmark_results"
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)
        print(f"Created directory: {results_dir}")
    
    # Dictionary to store all results
    benchmark_results = {}
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Test each model
    for model_name in models:
        print(f"\nBenchmarking model: {model_name}")
        model_results = []
        
        # Create a file to save results incrementally for this model
        model_file = os.path.join(results_dir, f"{model_name.replace(':', '_')}_{timestamp}.csv")
        
        # Create in-progress indicator file
        progress_file = os.path.join(results_dir, f"{model_name.replace(':', '_')}_{timestamp}_in_progress.txt")
        with open(progress_file, 'w') as f:
            f.write(f"Benchmark in progress for model: {model_name}\nStarted: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
        sample_count = 0
        for index, row in tqdm(sample_df.iterrows(), total=len(sample_df)):
            try:
                # Create prompt for this loan application
                prompt = create_prompt(row)
                
                # Query Ollama
                result = query_ollama(prompt, model_name)
                
                # Extract decision
                response_text = result["text"]
                if "APPROVE" in response_text.upper():
                    decision = 1  # Approve
                elif "REJECT" in response_text.upper():
                    decision = 0  # Reject
                else:
                    # If no clear decision, try to find it in the text
                    if "approve" in response_text.lower():
                        decision = 1
                    elif "reject" in response_text.lower():
                        decision = 0
                    else:
                        # Default to reject if no decision found
                        decision = 0
                
                # Store result
                result_data = {
                    "index": index,
                    "decision": decision,
                    "processing_time": result["processing_time"],
                    "tokens_per_second": result["tokens_per_second"],
                    "reasoning": response_text
                }
                model_results.append(result_data)
                
                # Save results incrementally
                sample_count += 1
                if sample_count % 5 == 0 or sample_count == len(sample_df):
                    # Create DataFrame from current results
                    current_df = pd.DataFrame(model_results)
                    current_df.to_csv(model_file, index=False)
                    
                    # Update progress file
                    with open(progress_file, 'w') as f:
                        f.write(f"Benchmark in progress for model: {model_name}\n")
                        f.write(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                        f.write(f"Progress: {sample_count}/{len(sample_df)} samples processed\n")
                        f.write(f"Current avg processing time: {sum(r['processing_time'] for r in model_results) / len(model_results):.2f} seconds\n")
                        f.write(f"Current avg tokens per second: {sum(r['tokens_per_second'] for r in model_results) / len(model_results):.2f}")
                
                # Small pause between requests
                time.sleep(0.5)
                
            except Exception as e:
                print(f"Error processing row {index} with model {model_name}: {e}")
        
        # Calculate aggregate stats
        if model_results:
            avg_processing_time = sum(r["processing_time"] for r in model_results) / len(model_results)
            avg_tokens_per_second = sum(r["tokens_per_second"] for r in model_results) / len(model_results)
            
            benchmark_results[model_name] = {
                "results": model_results,
                "avg_processing_time": avg_processing_time,
                "avg_tokens_per_second": avg_tokens_per_second
            }
            
            print(f"Model: {model_name}")
            print(f"Average processing time: {avg_processing_time:.2f} seconds")
            print(f"Average tokens per second: {avg_tokens_per_second:.2f}")
        
        # Remove the in-progress file
        if os.path.exists(progress_file):
            os.remove(progress_file)
    
    # Save overall results
    results_file = os.path.join(results_dir, f"benchmark_summary_{timestamp}.json")
    
    with open(results_file, 'w') as f:
        # Extract the summary data (without full response texts to keep file size manageable)
        summary = {}
        for model, data in benchmark_results.items():
            summary[model] = {
                "avg_processing_time": data["avg_processing_time"],
                "avg_tokens_per_second": data["avg_tokens_per_second"],
                "sample_size": len(data["results"])
            }
        json.dump(summary, f, indent=2)
        
    print(f"\nBenchmark summary saved to {results_file}")
    
    return benchmark_results

def run_full_test(file_path, model_name, num_samples=100):
    """Run a more complete test on a single model using more samples"""
    print(f"\nRunning full test on model: {model_name} with {num_samples} samples")
    
    # Load the input file
    input_df = pd.read_csv(file_path)
    
    # Take a larger random sample
    if len(input_df) > num_samples:
        test_df = input_df.sample(num_samples, random_state=42)
    else:
        test_df = input_df.copy()
    
    # Create the results directory
    results_dir = "./ollama_full_test_results"
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)
        print(f"Created directory: {results_dir}")
    
    # Create a copy of the dataframe to store results
    result_df = test_df.copy()
    result_df['ai_decision'] = None
    result_df['ai_reasoning'] = None
    result_df['processing_time'] = None
    result_df['tokens_per_second'] = None
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = os.path.join(results_dir, f"{model_name.replace(':', '_')}_{timestamp}.csv")
    
    # Start timer for overall process
    overall_start = time.time()
    
    # Process each row
    for index, row in tqdm(result_df.iterrows(), total=len(result_df)):
        try:
            # Create prompt for this loan application
            prompt = create_prompt(row)
            
            # Query Ollama
            result = query_ollama(prompt, model_name)
            
            # Extract decision
            response_text = result["text"]
            if "APPROVE" in response_text.upper():
                decision = 1  # Approve
            elif "REJECT" in response_text.upper():
                decision = 0  # Reject
            else:
                # If no clear decision, try to find it in the text
                if "approve" in response_text.lower():
                    decision = 1
                elif "reject" in response_text.lower():
                    decision = 0
                else:
                    # Default to reject if no decision found
                    decision = 0
            
            # Store results
            result_df.loc[index, 'ai_decision'] = decision
            result_df.loc[index, 'ai_reasoning'] = response_text
            result_df.loc[index, 'processing_time'] = result["processing_time"]
            result_df.loc[index, 'tokens_per_second'] = result["tokens_per_second"]
            
            # Save incrementally
            if index % 10 == 0:
                result_df.to_csv(output_file, index=False)
            
            # Small pause between requests
            time.sleep(0.5)
            
        except Exception as e:
            print(f"Error processing row {index}: {e}")
            result_df.loc[index, 'ai_reasoning'] = f"ERROR: {str(e)}"
    
    # Calculate overall time
    overall_time = time.time() - overall_start
    
    # Save final results
    result_df.to_csv(output_file, index=False)
    
    # Calculate and save summary statistics
    summary = {
        "model": model_name,
        "samples": len(result_df),
        "overall_time": overall_time,
        "avg_processing_time": result_df['processing_time'].mean(),
        "avg_tokens_per_second": result_df['tokens_per_second'].mean(),
        "approve_rate": (result_df['ai_decision'] == 1).mean() * 100,
        "reject_rate": (result_df['ai_decision'] == 0).mean() * 100
    }
    
    summary_file = os.path.join(results_dir, f"{model_name.replace(':', '_')}_summary_{timestamp}.json")
    with open(summary_file, 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"Test completed in {overall_time:.2f} seconds")
    print(f"Average processing time per sample: {summary['avg_processing_time']:.2f} seconds")
    print(f"Average tokens per second: {summary['avg_tokens_per_second']:.2f}")
    print(f"Results saved to {output_file}")
    print(f"Summary saved to {summary_file}")
    
    return result_df, summary

# Example usage
if __name__ == "__main__":
    # Check if the input file exists
    if not os.path.exists(input_file_path):
        print(f"Error: Input file not found at {input_file_path}")
        print("Please update the 'input_file_path' variable in the script to point to your CSV file")
        exit(1)

    # Check if models are available
    if not check_models_available(MODELS):
        exit(1)
    
    # Ask user which test to run
    print("\nWhat would you like to do?")
    print("1. Run a quick benchmark of all models (5 samples each)")
    print("2. Run a full test on a specific model (50 samples)")
    print("3. Run both tests")
    
    choice = input("Enter your choice (1, 2, or 3): ")
    
    if choice == '1' or choice == '3':
        # Ask for sample size
        try:
            sample_size = int(input("Enter number of samples for benchmark test (default: 5): ") or 5)
        except ValueError:
            sample_size = 5
            print("Using default sample size of 5")
        
        # Run benchmark
        print(f"\nRunning benchmark of all models with {sample_size} samples...")
        results = benchmark_models(input_file_path, MODELS, sample_size=sample_size)
    
    if choice == '2' or choice == '3':
        # Ask which model to use for full test
        print("\nWhich model would you like to use for the full test?")
        for i, model in enumerate(MODELS, 1):
            print(f"{i}. {model}")
        
        model_choice = input(f"Enter your choice (1-{len(MODELS)}): ")
        try:
            model_index = int(model_choice) - 1
            selected_model = MODELS[model_index]
        except (ValueError, IndexError):
            print("Invalid choice. Using llama3:8b as default.")
            selected_model = "llama3:8b"
        
        # Ask for sample size
        try:
            num_samples = int(input("Enter number of samples for full test (default: 50): ") or 50)
        except ValueError:
            num_samples = 50
            print("Using default sample size of 50")
        
        # Run full test
        print(f"\nRunning full test on {selected_model} with {num_samples} samples...")
        full_results, summary = run_full_test(input_file_path, selected_model, num_samples=num_samples)