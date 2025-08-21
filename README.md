# SNOMED Prediction Benchmark Framework

**Author:** Youngwook Yang (youngwook.yang@gmail.com)

## Introduction
This project introduces an interactive benchmarking framework for evaluating large language models (LLMs) as virtual clinicians. Instead of static question answering, LLMs must iteratively query a simulated patient (oracle) to infer conditions and predict **SNOMED codes**. 

- Oracle: Fine-tuned BioMistral-7B using LoRA on synthetic doctor-patient Q&A pairs (generated from Synthea + GPT-3.5-turbo).
- Target: Any LLM accessible via OpenAI-compatible API (tested with GPT-3.5-turbo, GPT-4-turbo, Grok-3-mini).
- Goal: Measure accuracy, precision, and recall of predicted SNOMED codes after a multi-turn dialogue.

## Sample Benchmark Result (GPT-4 Turbo)
```
🧪 Running case 3/3 for patient 36c63375-a5be-f2fd-5ead-457c203771ed...
Patient Context: Age: 16
Gender: F
Location: East Longmeadow, Massachusetts
Conditions: Acute bacterial sinusitis (disorder); Otitis media; Acute bronchitis (disorder); Sprain of ankle; Sinusitis (disorder); Suspected COVID-19; COVID-19
```

### Conversation Excerpt from Benchmark Run
```
Turn 1: LLM Question: What symptoms are you currently experiencing?
Turn 1: Oracle Answer: I've been feeling congested and having trouble breathing. My throat is also sore.
Turn 2: LLM Question: Do you have a fever, or have you noticed any other symptoms such as coughing or sneezing?
Turn 2: Oracle Answer: Yes, I've been coughing a lot and my throat hurts. I also have a high fever.
Turn 3: LLM Question: How long have you been experiencing these symptoms?
Turn 3: Oracle Answer: A few weeks now, it seems like it's not getting better.
Turn 4: LLM Question: Have you been in contact with anyone who is sick, or have you traveled recently?
Turn 4: Oracle Answer: I'm not sure. I haven't been keeping track of my contacts. I haven't traveled recently.
...
```

### Becnmark Result
```
LLM Final Response: <Final Answer>: ["840539006", "1023001"]
Gold SNOMED Codes: ['75498004 - Acute bacterial sinusitis (disorder)', '65363002 - Otitis media', '10509002 - Acute bronchitis (disorder)', '44465007 - Sprain of ankle', '36971009 - Sinusitis (disorder)', '840544004 - Suspected COVID-19', '840539006 - COVID-19']
```

## Related Work
- **SDBench (Microsoft, 2025):** Interactive diagnosis benchmark with oracle Q&A.  
- **MultiMedQA (Singhal et al., 2022):** Static benchmark combining six medical QA datasets.  
- **SNOMED AutoCoding Models (Huang et al., 2019):** Supervised coding prediction from EHR text.  

## Methodology
1. **Oracle Construction**
   - Fine-tuned BioMistral-7B with LoRA (RTX 4070 SUPER, 12GB).
   - Training data: 500 synthetic patients × 7 Q&A pairs each (5 positive, 2 negative).
   - Oracle responds in concise lay language and says “I’m not sure” for unknowns.

2. **Interactive Diagnostic Loop**
   - Target LLM plays doctor, asks up to 20 questions.
   - Oracle answers as patient.
   - LLM issues `<Final Answer>: ["SNOMED_CODE_1", ...]`.

3. **Evaluation**
   - Metrics: Exact match, precision, recall, F1.
   - Pipeline: Modular Python framework, OpenAI-compatible API.

## Results
- **GPT-3.5-turbo:** Sensible questions, but limited by vague oracle answers.  
- **GPT-4-turbo:** Better reasoning, still hindered by oracle limitations. Correctly predicted COVID-19 in one case.  
- **Grok-3-mini:** Struggled most; invalid SNOMED codes.  
- **Oracle Performance:** Strong realism (concise, lay language, uncertainty handling) but issues with prompt leakage, over-verbosity, and overuse of “I’m not sure.”

## Conclusion
This benchmark evaluates LLMs not just on knowledge, but on **interactive clinical reasoning**. While the oracle limitations capped accuracy, the framework demonstrates how interactive benchmarks can better simulate real-world diagnostic workflows. 

**Future work:**  
- Expand training data for oracle.  
- Improve uncertainty variety.  
- Test more LLMs and reinforcement learning strategies.

## More Details
Please check [report.pdf](report.pdf) for more details.

## How to Run Benchmark
1. Clone the repo.
2. Install packages with [requirements.txt](benchmark/requirements.txt).
3. Run [snomed-benchmark.py](benchmark/snomed-benchmark.py) with options like `model_name`, `base_url`, and `api_key`.
```bash
python benchmark/snomed-benchmark.py --model_name grok-3-mini --base_url https://api.x.ai/v1 --api_key sk-xxxx --num_samples 5 --max_turns 15
```

## How to Reproduce the orcle model
1. Clone the repo.
2. Check [fine-tune.ipynb](train/fine-tune.ipynb) for the step-by-step guide.