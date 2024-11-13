import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import pandas as pd
from fuzzywuzzy import fuzz
import time

# Load the model and tokenizer
model_name = "fdqerq22ds/MathScale-Mistral"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

# Set the padding token to EOS token if not set
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# Move model to GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

# Function to solve math problems with multiple passes (for RR.All)
def solve_math_problem(problem_text, num_attempts=64):
    prompt = f"Solve the following math problem: {problem_text}\nAnswer:"
    inputs = tokenizer(prompt, return_tensors="pt", padding=True).to(device)

    outputs = []
    for _ in range(num_attempts):
        with torch.no_grad():
            output_ids = model.generate(
                inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
                max_new_tokens=150,
                do_sample=True,
                temperature=0.7,
                pad_token_id=tokenizer.eos_token_id
            )
        output_text = tokenizer.decode(output_ids[0], skip_special_tokens=True).strip()
        outputs.append(output_text)
    return outputs

# Function to compute Exact Match
def exact_match(predicted_answer, reference_answer):
    return int(predicted_answer.strip() == reference_answer.strip())

# Load dataset
dataset = pd.read_json('/content/100scoredataset.json')

# Function to calculate RR.All
def evaluate_rr_all_metric(dataset):
    total_questions = len(dataset)
    rr_all = 0  # Counter for RR.All

    for idx, row in dataset.iterrows():
        print(f"Solving problem ID {idx + 1}/{total_questions}...")

        problem_text = row['algebraic_word_problem']
        reference_answer = row['reference_answer']

        # Get predictions for multiple attempts
        predicted_answers = solve_math_problem(problem_text, num_attempts=64)

        # Calculate RR.All: check if the correct answer appears in any of the generated answers
        if any(exact_match(answer, reference_answer) for answer in predicted_answers):
            rr_all += 1

    # Calculate RR.All rate
    rr_all_rate = rr_all / total_questions * 100
    print(f"RR.All Accuracy: {rr_all_rate:.2f}%")

# Run the RR.All metric evaluation
evaluate_rr_all_metric(dataset)
