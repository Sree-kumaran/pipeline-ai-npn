import json
import sys
from pathlib import Path

import joblib
from safetensors import safe_open
from tokenizers import Tokenizer


# Project folders
PROJECT_DIR = Path(__file__).resolve().parent.parent
XGBOOST_DIR = PROJECT_DIR / "models" / "xgboost"
LORA_DIR = PROJECT_DIR / "models" / "priorauthlm"


def fail(message):
    print(f"\n❌ ERROR: {message}")
    sys.exit(1)


def check_xgboost_models():
    print("\n========== XGBOOST MODEL CHECK ==========")

    priority_path = XGBOOST_DIR / "xgb_priority_classifier.joblib"
    criticality_path = XGBOOST_DIR / "xgb_criticality_regressor.joblib"

    if not priority_path.exists():
        fail(f"Missing file: {priority_path}")

    if not criticality_path.exists():
        fail(f"Missing file: {criticality_path}")

    # Priority classifier
    try:
        priority_model = joblib.load(priority_path)
        print("\n✅ Priority classifier loaded successfully")
        print("Model type:", type(priority_model).__name__)
        print("Feature count:", getattr(priority_model, "n_features_in_", "Not stored"))
        print("Feature names:", getattr(priority_model, "feature_names_in_", "Not stored"))
        print("Classes:", getattr(priority_model, "classes_", "Not stored"))
    except Exception as error:
        fail(f"Could not load priority classifier: {error}")

    # Criticality regressor
    try:
        criticality_model = joblib.load(criticality_path)
        print("\n✅ Criticality regressor loaded successfully")
        print("Model type:", type(criticality_model).__name__)
        print("Feature count:", getattr(criticality_model, "n_features_in_", "Not stored"))
        print("Feature names:", getattr(criticality_model, "feature_names_in_", "Not stored"))
    except Exception as error:
        fail(f"Could not load criticality regressor: {error}")


def check_lora_adapter():
    print("\n========== QWEN LORA ADAPTER CHECK ==========")

    required_files = [
        "adapter_config.json",
        "adapter_model.safetensors",
        "chat_template.jinja",
        "tokenizer_config.json",
        "tokenizer.json",
    ]

    for filename in required_files:
        file_path = LORA_DIR / filename
        if not file_path.exists():
            fail(f"Missing LoRA adapter file: {file_path}")
        print(f"✅ Found: {filename}")

    # Read adapter configuration
    try:
        with open(LORA_DIR / "adapter_config.json", "r", encoding="utf-8") as file:
            adapter_config = json.load(file)

        base_model = adapter_config.get("base_model_name_or_path")
        peft_type = adapter_config.get("peft_type")
        task_type = adapter_config.get("task_type")
        target_modules = adapter_config.get("target_modules")

        print("\n✅ adapter_config.json is valid")
        print("Base model required:", base_model)
        print("PEFT type:", peft_type)
        print("Task type:", task_type)
        print("LoRA target modules:", target_modules)

        if not base_model:
            fail("base_model_name_or_path is missing in adapter_config.json")

        if peft_type != "LORA":
            fail(f"Expected a LORA adapter, but found: {peft_type}")

    except Exception as error:
        fail(f"Could not read adapter_config.json: {error}")

    # Verify adapter weights can be read
    try:
        weights_path = LORA_DIR / "adapter_model.safetensors"

        with safe_open(str(weights_path), framework="np") as adapter_weights:
            tensor_names = list(adapter_weights.keys())

        if len(tensor_names) == 0:
            fail("adapter_model.safetensors contains no tensors")

        print(f"\n✅ LoRA adapter weights are readable")
        print("Number of adapter tensors:", len(tensor_names))

    except Exception as error:
        fail(f"Could not read adapter_model.safetensors: {error}")

    # Verify packaged tokenizer
    try:
        tokenizer = Tokenizer.from_file(str(LORA_DIR / "tokenizer.json"))
        vocabulary_size = tokenizer.get_vocab_size(with_added_tokens=True)

        if vocabulary_size <= 0:
            fail("Tokenizer vocabulary size is invalid")

        print("\n✅ tokenizer.json is valid")
        print("Tokenizer vocabulary size:", vocabulary_size)

    except Exception as error:
        fail(f"Could not read tokenizer.json: {error}")

    # Check chat template is not empty
    try:
        chat_template = (LORA_DIR / "chat_template.jinja").read_text(
            encoding="utf-8"
        ).strip()

        if not chat_template:
            fail("chat_template.jinja is empty")

        print("✅ chat_template.jinja is present and non-empty")

    except Exception as error:
        fail(f"Could not read chat_template.jinja: {error}")


def main():
    print("PHASE 0 — MODEL READINESS VERIFICATION")

    check_xgboost_models()
    check_lora_adapter()

    print("\n==========================================")
    print("✅ PHASE 0 PASSED — MODELS ARE READY")
    print("You can continue to Phase 1: input validation.")
    print("==========================================")


if __name__ == "__main__":
    main()