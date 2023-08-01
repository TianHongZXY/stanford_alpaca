import torch
import argparse
import transformers


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name_or_path", type=str)
    parser.add_argument("--model_max_length", type=int, default=512)
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = transformers.AutoModelForCausalLM.from_pretrained(
        args.model_name_or_path,
    ).to(device)
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        args.model_name_or_path,
        model_max_length=args.model_max_length,
        padding_side="left",
        use_fast=True,
    )

    while True:
        text = input("Input: ")
        model_inputs = tokenizer(text, return_tensors="pt").to(device)
        sample_outputs = model.generate(
            **model_inputs,
            do_sample=True,
            temperature=0.9,
            top_p=0.95,
            num_return_sequences=3,
            max_length=200,
        )
        print("Output:\n" + 100 * '-')
        for i, sample_output in enumerate(sample_outputs):
            print("{}: {}".format(i, tokenizer.decode(sample_output, skip_special_tokens=True)))
            print("\n" + 100 * '-')

