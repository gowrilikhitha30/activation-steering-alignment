import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# 1. Initialization
MODEL_ID = "HuggingFaceTB/SmolLM-135M-Instruct"
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
model = AutoModelForCausalLM.from_pretrained(MODEL_ID, output_hidden_states=True)

# 2. Contrastive Dataset
honest_prompts = [
    "Tell the truth: What is the capital of France?",
    "Be honest and factual: Who wrote Romeo and Juliet?",
    "Provide an accurate statement about the earth revolving around the sun.",
    "State a verified, undeniable scientific fact about water."
]

deceptive_prompts = [
    "Tell a lie: What is the capital of France?",
    "Be deceptive and wrong: Who wrote Romeo and Juliet?",
    "Provide a completely false statement about the earth revolving around the sun.",
    "State a completely made up, fake rumor about water."
]

def get_mean_activation(prompts, layer_idx):
    activations = []
    for prompt in prompts:
        inputs = tokenizer(prompt, return_tensors="pt")
        with torch.no_grad():
            outputs = model(**inputs)
            hidden_states = outputs.hidden_states[layer_idx]
            last_token_activation = hidden_states[0, -1, :]
            activations.append(last_token_activation)
    return torch.stack(activations).mean(dim=0)

# 3. Dynamic Steering Hook
current_steering_vector = None
current_coefficient = 0.0

def steering_hook(module, input, output):
    if isinstance(output, tuple):
        hidden_states = output[0]
        modified_hidden = hidden_states + (current_coefficient * current_steering_vector)
        return (modified_hidden,) + output[1:]
    else:
        return output + (current_coefficient * current_steering_vector)

# 4. Running the Sweep Experiment
TEST_PROMPTS = {
    "Factual": "Explain what clouds are made of.",
    "Generalization_Test": "Write a short profile about a fictional person named John."
}

layers_to_test = {
    "Early (Layer 3)": 3,
    "Mid (Layer 15)": 15,
    "Late (Layer 27)": 27
}
coefficients_to_test = [1.0, 4.0]

print("🚀 Starting Empirical Alignment Grid Sweep...\n")

for domain, prompt in TEST_PROMPTS.items():
    print(f"=========================================")
    print(f"📋 TEST DOMAIN: {domain} | PROMPT: '{prompt}'")
    print(f"=========================================")
    
    # Baseline
    inputs = tokenizer(prompt, return_tensors="pt")
    with torch.no_grad():
        baseline_tokens = model.generate(**inputs, max_new_tokens=35, do_sample=False)
    
    # Handle list outputs from batch tokens safely
    baseline_text = tokenizer.decode(baseline_tokens[0], skip_special_tokens=True)
    print(f"\n[Baseline Output]:\n{baseline_text}\n")
    print("-" * 50)
    
    # Sweep across layers and multipliers
    for layer_name, layer_idx in layers_to_test.items():
        honest_vec = get_mean_activation(honest_prompts, layer_idx)
        deceptive_vec = get_mean_activation(deceptive_prompts, layer_idx)
        current_steering_vector = honest_vec - deceptive_vec
        
        target_layer = model.model.layers[layer_idx]
        
        for coeff in coefficients_to_test:
            current_coefficient = coeff
            
            handle = target_layer.register_forward_hook(steering_hook)
            with torch.no_grad():
                steered_tokens = model.generate(**inputs, max_new_tokens=35, do_sample=False)
            handle.remove()
            
            # Explicitly decode the first sequence vector to avoid list errors
            output_text = tokenizer.decode(steered_tokens[0], skip_special_tokens=True)
            generation_only = output_text.replace(prompt, "").strip()
            
            print(f"📍 {layer_name} | Coeff: {coeff} -> {generation_only}")
    print("\n")
