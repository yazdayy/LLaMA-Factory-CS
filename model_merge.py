import argparse
from peft import PeftModel
from transformers import AutoTokenizer
from transformers import AutoModelForCausalLM

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_model_path", type=str, default=None,
                        help="Path to base model.")
    
    parser.add_argument("--adapter_path", type=str, default=None,
                        help="Path to adapter.")
    
    parser.add_argument("--save_path", type=str, default=None,
                        help="Path to save merged model.")
    
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.base_model_path)    
    base_model = AutoModelForCausalLM.from_pretrained(args.base_model_path, device_map="cpu")
    peft_model = PeftModel.from_pretrained(base_model, args.adapter_path, device_map="cpu")

    merged_model = peft_model.merge_and_unload()
    print(type(merged_model))

    total = sum(p.numel() for p in merged_model.parameters())
    print(f"Total parameters: {total / 1e9:.2f}B")

    # Save in optimized format
    merged_model.half()
    merged_model.save_pretrained(
        args.save_path,
        safe_serialization=True,
        max_shard_size="5GB"
    )
    tokenizer.save_pretrained(args.save_path)