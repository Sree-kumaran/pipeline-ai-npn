from functools import lru_cache
from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

from app.model_service import predict


PROJECT_DIR = Path(__file__).resolve().parent.parent
LORA_DIR = PROJECT_DIR / "models" / "priorauthlm"
BASE_MODEL_ID = "Qwen/Qwen2.5-1.5B-Instruct"


@lru_cache
def load_llm():
    """
    Loads the Qwen base model and attaches the local LoRA adapter once.
    The base model downloads only on the first run if not already cached.
    """
    use_gpu = torch.cuda.is_available()
    dtype = torch.float16 if use_gpu else torch.float32

    tokenizer = AutoTokenizer.from_pretrained(
        LORA_DIR,
        local_files_only=True,
    )

    # Use the fine-tuned adapter's supplied chat template.
    template_path = LORA_DIR / "chat_template.jinja"
    tokenizer.chat_template = template_path.read_text(encoding="utf-8")

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    base_model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL_ID,
        torch_dtype=dtype,
        device_map="auto" if use_gpu else None,
    )

    model = PeftModel.from_pretrained(base_model, LORA_DIR)
    model.eval()

    device = next(model.parameters()).device
    return model, tokenizer, device


def generate_explanation(
    features: dict[str, float],
    clinical_summary: str,
) -> dict:
    """
    Runs XGBoost first, then asks PriorAuthLM to explain the result.
    """
    xgboost_result = predict(features)
    model, tokenizer, device = load_llm()

    prompt = f"""
Clinical summary:
{clinical_summary}

Model results:
- Priority class: {xgboost_result["priority_class"]}
- Priority probabilities: {xgboost_result["priority_probabilities"]}
- Criticality score: {xgboost_result["criticality_score"]}

Provide a concise prior-authorization explanation based only on the
clinical summary and model results. Do not invent patient facts.
Do not state that this output is a final clinical decision.
""".strip()

    messages = [
        {
            "role": "system",
            "content": (
                "You are PriorAuthLM, an assistant that produces clear, "
                "evidence-based prior-authorization explanations."
            ),
        },
        {
            "role": "user",
            "content": prompt,
        },
    ]

    model_inputs = tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_tensors="pt",
        return_dict=True,
    ).to(device)

    with torch.inference_mode():
        generated_tokens = model.generate(
            **model_inputs,
            max_new_tokens=300,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )

    new_tokens = generated_tokens[0][model_inputs["input_ids"].shape[-1]:]
    explanation = tokenizer.decode(
        new_tokens,
        skip_special_tokens=True,
    ).strip()

    return {
        "xgboost_prediction": xgboost_result,
        "llm_explanation": explanation,
    }