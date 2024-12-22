import datasets

class DPODataCollator:
    def __init__(self, tokenizer, max_length=2048):
        self.tokenizer = tokenizer
        self.max_length = max_length

    def format_message(self, message):
        """Format a single message into ChatML format string."""
        role = message["role"]
        content = message["content"]
        return f"<|im_start|>{role}\n{content}<|im_end|>\n"

    def format_conversation(self, messages):
        """Format a list of messages into a complete conversation string."""
        return "".join(self.format_message(msg) for msg in messages).strip()

    def process_example(self, example):
        """Process a single example to extract prompt and responses."""
        # Extract the prompt text directly from the example
        if isinstance(example["prompt"], str):
            prompt_text = example["prompt"]
        else:
            # If prompt is a message or list of messages
            prompt_messages = example["prompt"] if isinstance(example["prompt"], list) else [example["prompt"]]
            prompt_text = self.format_conversation(prompt_messages)

        # Process chosen responses
        chosen_messages = example["chosen"]
        # If the chosen response includes the prompt messages, extract only the assistant's response
        if chosen_messages[0]["role"] == "user" and chosen_messages[0]["content"] == prompt_text:
            chosen_messages = chosen_messages[1:]  # Skip the prompt message
        chosen_text = self.format_conversation(chosen_messages)

        # Process rejected responses
        rejected_messages = example["rejected"]
        # If the rejected response includes the prompt messages, extract only the assistant's response
        if rejected_messages[0]["role"] == "user" and rejected_messages[0]["content"] == prompt_text:
            rejected_messages = rejected_messages[1:]  # Skip the prompt message
        rejected_text = self.format_conversation(rejected_messages)

        prompt_text = self.format_conversation([{"role": "user", "content": prompt_text}])
        # Build complete conversations by combining prompt with responses
        # full_chosen_text = f"{prompt_text}\n{chosen_text}"
        # full_rejected_text = f"{prompt_text}\n{rejected_text}"

        return {
            "prompt": prompt_text,
            "chosen": chosen_text,
            "rejected": rejected_text
        }

    def __call__(self, examples):
        # Process each example
        processed = [self.process_example(ex) for ex in examples]
        
        # Extract prompts and responses
        prompts = [p["prompt"] for p in processed]
        chosen = [p["chosen"] for p in processed]
        rejected = [p["rejected"] for p in processed]

        # Tokenize prompts
        prompt_tokens = self.tokenizer(
            prompts,
            max_length=self.max_length,
            padding=True,
            truncation=True,
            return_tensors="pt"
        )

        # Tokenize chosen and rejected responses
        chosen_tokens = self.tokenizer(
            chosen,
            max_length=self.max_length,
            padding=True,
            truncation=True,
            return_tensors="pt"
        )

        rejected_tokens = self.tokenizer(
            rejected,
            max_length=self.max_length,
            padding=True,
            truncation=True,
            return_tensors="pt"
        )

        return {
            'prompt_input_ids': prompt_tokens['input_ids'],
            'prompt_attention_mask': prompt_tokens['attention_mask'],
            'chosen_input_ids': chosen_tokens['input_ids'],
            'chosen_attention_mask': chosen_tokens['attention_mask'],
            'rejected_input_ids': rejected_tokens['input_ids'],
            'rejected_attention_mask': rejected_tokens['attention_mask'],
        }

if __name__ == '__main__':
    dataset_path = '/home/yueyulin/data/Infinity-Preference/'
    import glob
    #Train files starts with train ends with parquet
    train_files = glob.glob(dataset_path + 'train*.parquet')
    print(train_files)
    train_ds = datasets.load_dataset('parquet', data_files=train_files)['train']
    print(train_ds)
    print(train_ds[0])
    model_path = "/home/yueyulin/models/Qwen2.5-7B-Instruct/"
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    collator = DPODataCollator(tokenizer)
    import torch
    from torch.utils.data import DataLoader
    loader = DataLoader(train_ds, batch_size=2, collate_fn=collator)
    for batch in loader:
        print(batch['prompt_input_ids'].tolist())
        print(batch['chosen_input_ids'].tolist())
        print(batch['rejected_input_ids'].tolist())
        print(batch['prompt_attention_mask'].tolist())
        print(batch['chosen_attention_mask'].tolist())
        print(batch['rejected_attention_mask'].tolist())
        break