#
# Load patient samples
#
import json
import random

def load_patient_samples(filepath, num_samples=5):
    with open(filepath, "r") as f:
        patients = json.load(f)
    return random.sample(patients, num_samples)

#
# Load Oracle Model
#
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

def load_oracle_model():
    # Path to LoRA adapter and tokenizer
    adapter_path = "./biomistral_oracle_lora"

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(adapter_path)
    tokenizer.pad_token = tokenizer.eos_token

    # Load base model (BioMistral-7B)
    base_model = AutoModelForCausalLM.from_pretrained("BioMistral/BioMistral-7B")

    # Load LoRA adapter
    model = PeftModel.from_pretrained(base_model, adapter_path)
    model = model.to("cuda")

    # Set to eval mode for inference
    model.eval()  

    return model, tokenizer

#
# Ask questions to the oracle
#
import torch

def ask_question_to_oracle(oracle_model, oracle_tokenizer, patient_context, question):
    prompt = f"""### Instruction:
                You are a patient. Answer the doctor's question based on the patient context given. 
                Use lay language only. Please be concise and answer in one or two short sentences.

                ### Input:
                {patient_context}\n
                Question: {question}

                ### Response:"""

    # Tokenize and generate
    inputs = oracle_tokenizer(prompt, return_tensors="pt").to(oracle_model.device)
    with torch.no_grad():
        outputs = oracle_model.generate(
            **inputs,
            pad_token_id=oracle_tokenizer.eos_token_id, 
            max_new_tokens=50
        )

    # Decode
    response = oracle_tokenizer.decode(outputs[0], skip_special_tokens=True)
    return response.split("### Response:")[-1].strip()


#
# Ask the target model
#
import openai

def ask_target_model(conversation_history, model_name=None, base_url=None, api_key=None, force_final_answer=False):

    # Construct conversation as a readable dialogue string
    dialogue = "\n".join(
        [f"{turn['role'].capitalize()}: {turn['content']}" for turn in conversation_history]
    )

    # Build message list for OpenAI Chat API
    messages = [
        {
            "role": "system",
            "content": (
                "You are a doctor diagnosing a patient through a multi-turn Q&A session.\n"
                "Ask one question at a time to gather enough information.\n"
                "When ready, respond with your final diagnosis using this format:\n"
                "<Final Answer>: [\"SNOMED_CODE_1\", \"SNOMED_CODE_2\"]\n"
                "Use concise, medically relevant questions."
            )
        },
        {
            "role": "user",
            "content": (
                f"Conversation so far:\n{dialogue}\n\n"
            )
        }
    ]

    if force_final_answer:
        messages.append({
            "role": "user",
            "content": "Please respond with your final answer in this format:\n"
                "<Final Answer>: [\"SNOMED_CODE_1\", \"SNOMED_CODE_2\"]"
        })
    else:
        messages.append({
            "role": "user",
            "content": 
                "What is your next question for the patient?\n"
                "Or if you are ready, respond with your final answer in this format:\n"
                "<Final Answer>: [\"SNOMED_CODE_1\", \"SNOMED_CODE_2\"]"
        })

    client = openai.OpenAI(api_key=api_key, base_url=base_url)

    response = client.chat.completions.create(
        model=model_name,
        messages=messages,
        temperature=0.7
    )

    return response.choices[0].message.content.strip()

import re
import json

def extract_snomed_codes(response_text):
    match = re.search(r"<Final Answer>:\s*(\[[^\]]*\])", response_text)
    if match:
        try:
            return set(json.loads(match.group(1)))
        except json.JSONDecodeError:
            return set()
    return set()

#
# Single Interview
#
def run_single_interview(patient, oracle_model, oracle_tokenizer, model_name="gpt-3.5-turbo", base_url=None, api_key=None, max_turns=8):
    history = []
    snomed_gold = set(patient["snomed_gold"])
    asked_questions = set()
    guessed_snomeds = set()

    for turn in range(max_turns):
        # Ask target LLM for the next question
        llm_response = ask_target_model(history, model_name=model_name, base_url=base_url, api_key=api_key)
        if "Final Answer" in llm_response:
            break

        question = llm_response  # Assume it's a plain question

        print(f"Turn {turn + 1}: LLM Question: {question}")

        asked_questions.add(question)

        # Get oracle answer
        answer = ask_question_to_oracle(oracle_model, oracle_tokenizer, patient["oracle_context"], question)

        print(f"Turn {turn + 1}: Oracle Answer: {answer}")

        # Append to chat history
        history.append({"role": "user", "content": question})
        history.append({"role": "assistant", "content": answer})

    if "Final Answer" in history[-1]["content"]:
        final_response = history[-1]["content"]
    else:
        history.append({"role": "user", "content": "Please give your final answer as a list of SNOMED codes."})
        final_response = ask_target_model(history, model_name=model_name, base_url=base_url, api_key=api_key, force_final_answer=True)

    print(f"LLM Final Response: {final_response}")

    guessed_snomeds = extract_snomed_codes(final_response)

    return {
        "patient_id": patient["patient_id"],
        "gold": list(snomed_gold),
        "predicted": list(guessed_snomeds),
        "intersection": list(snomed_gold & guessed_snomeds),
        "precision": len(snomed_gold & guessed_snomeds) / max(len(guessed_snomeds), 1),
        "recall": len(snomed_gold & guessed_snomeds) / max(len(snomed_gold), 1)
    }

#
# Run Benchmark
#
def run_benchmark(
    oracle_model,
    oracle_tokenizer,
    model_name="gpt-3.5-turbo",
    base_url=None,
    api_key=None,
    context_file="synthea_oracle_context.json",
    num_samples=5,
    max_turns=8
):
    patients = load_patient_samples(context_file, num_samples)
    results = []

    for i, patient in enumerate(patients):
        print(f"\n🧪 Running case {i + 1}/{num_samples} for patient {patient['patient_id']}...")
        print(f"Patient Context: {patient['oracle_context']}")
        print(f"Gold SNOMED Codes: {patient['snomed_gold']}")
        result = run_single_interview(
            patient=patient,
            oracle_model=oracle_model,
            oracle_tokenizer=oracle_tokenizer,
            model_name=model_name,
            base_url=base_url,
            api_key=api_key,
            max_turns=max_turns
        )
        results.append(result)

    # Compute average scores
    avg_precision = sum(r["precision"] for r in results) / len(results)
    avg_recall = sum(r["recall"] for r in results) / len(results)

    print("\n📊 Benchmark Summary")
    print(f"Model: {model_name}")
    print(f"Samples: {num_samples}")
    print(f"Average Precision: {avg_precision:.2f}")
    print(f"Average Recall: {avg_recall:.2f}")

    return results

# Suppress warnings from flash attention
import warnings
warnings.filterwarnings("ignore")

# Load the fine-tuned oracle model
oracle_model, oracle_tokenizer = load_oracle_model()

import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run SNOMED benchmark with a target LLM")

    parser.add_argument("--model_name", type=str, required=True, help="Name of the target LLM (e.g., gpt-3.5-turbo, grok-3-mini)")
    parser.add_argument("--base_url", type=str, default=None, help="Base URL for non-OpenAI models (e.g., https://api.x.ai/v1)")
    parser.add_argument("--api_key", type=str, required=True, help="API key for the target model provider")
    parser.add_argument("--num_samples", type=int, default=3, help="Number of samples to run from the dataset")
    parser.add_argument("--max_turns", type=int, default=20, help="Maximum number of conversation turns per interview")

    args = parser.parse_args()

    results = run_benchmark(
        oracle_model=oracle_model,
        oracle_tokenizer=oracle_tokenizer,
        context_file="./data/synthea_oracle_context.json",
        model_name=args.model_name,
        base_url=args.base_url,
        api_key=args.api_key,
        num_samples=args.num_samples,
        max_turns=args.max_turns
    )
