
from typing import Literal
import json
import re
import torch
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import LoraConfig, get_peft_model
from trl import SFTTrainer, SFTConfig


class OpinionExtractor:

    # SET THE FOLLOWING CLASS VARIABLE to "FT" if you implemented a fine-tuning approach
    method: Literal["NOFT", "FT"] = "FT"

    # DO NOT MODIFY THE SIGNATURE OF THIS METHOD, add code to implement it
    def __init__(self, cfg) -> None:
        self.cfg = cfg

        # 1. Clear GPU memory to avoid Out-Of-Memory (OOM) across multiple runs
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        # 2. Define constants and target model (Strictly constrained by the assignment)
        self.model_id = "Qwen/Qwen3-4B"
        self.aspects = ["Price", "Food", "Service"]
        self.valid_labels = {"Positive", "Negative", "Mixed", "No Opinion"}

        # 3. Load Tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_id)
        # Qwen models usually need a padding token assigned for batch processing
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id

        # 4. Load Base Model
        # device_map=None is REQUIRED so accelerate can properly wrap the model in DDP
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_id,
            torch_dtype=torch.bfloat16,
            device_map=None,
            attn_implementation="sdpa"  #little optimisation but not in the assignment
        )

    def _format_prompt(self, review: str, target: dict = None) -> str:
        """
        Helper method to format the input as a strict instruction for Causal LM.
        """
        messages = [
            {
                "role": "system", 
                "content": (
                    "You are a strict JSON-only sentiment analysis AI. "
                    "Extract the overall opinions for: 'Price', 'Food', 'Service'.\n"
                    "Allowed values: 'Positive', 'Negative', 'Mixed', 'No Opinion'.\n"
                    "Output ONLY a valid JSON dictionary, without markdown formatting or conversational text."
                )
            },
            {"role": "user", "content": f"Review: {review}"}
        ]
        
        if target is not None:
            messages.append({"role": "assistant", "content": json.dumps(target)})
            return self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
        else:
            return self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

        


    # DO NOT MODIFY THE SIGNATURE OF THIS METHOD, add code to implement it
    def train(self, train_data: list[dict], val_data: list[dict]) -> None:
        """
        Trains the model, if OpinionExtractor.method=="FT"
        """
        # 1. Format the tabular data into HuggingFace Datasets for TRL
        def format_dataset(data: list[dict]) -> Dataset:
            formatted = []
            for row in data:
                target = {
                    "Price": row.get("Price", "No Opinion"),
                    "Food": row.get("Food", "No Opinion"),
                    "Service": row.get("Service", "No Opinion")
                }
                formatted.append({"text": self._format_prompt(row["Review"], target)})
            return Dataset.from_list(formatted)

        train_ds = format_dataset(train_data)
        val_ds = format_dataset(val_data)

        # 2. Configure PEFT / LoRA
        # Targeting attention and MLP projections maximizes adaptation capacity
        peft_config = LoraConfig(
            r=16,
            lora_alpha=32,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
            lora_dropout=0.05,
            bias="none",
            task_type="CAUSAL_LM"
        )

        # 3. Calculate dynamic gradient accumulation and scaling
        num_devices = max(1, torch.cuda.device_count())
        grad_accum_steps = max(1, 16 // num_devices)
        
        # Scale the learning rate dynamically based on the number of available devices.
        # We use a square-root scaling rule, standard for Adam/AdamW optimizers.
        base_lr = 5e-5 
        scaled_lr = base_lr * (num_devices ** 0.5)

        # 4. Set up the SFT Trainer
        training_args = SFTConfig(
            output_dir="./lora_absa_weights",
            per_device_train_batch_size=1,              
            gradient_accumulation_steps=grad_accum_steps,
            learning_rate=scaled_lr,
            num_train_epochs=2,
            bf16=True,                                  
            logging_steps=10,
            eval_strategy="epoch",
            save_strategy="no",
            report_to="none",
            dataset_text_field="text",
            max_length=512,
            packing=True,                               
            dataloader_num_workers=4,                   
            optim="adamw_torch_fused"     
        )

        trainer = SFTTrainer(
            model=self.model,
            train_dataset=train_ds,
            eval_dataset=val_ds,
            peft_config=peft_config,
            processing_class=self.tokenizer,
            args=training_args,
        )

        # 5. Launch Training
        trainer.train()
        
        # 6. Switch model to evaluation mode for the upcoming predict() calls
        self.model.eval()

    # DO NOT MODIFY THE SIGNATURE OF THIS METHOD, add code to implement it
    def predict(self, texts: list[str]) -> list[dict]:
        """
        :param texts: list of reviews from which to extract the opinion values
        :return: a list of dicts, one per input review, containing the opinion values for the 3 aspects.
        """
        predictions = []
        prompts = [self._format_prompt(text) for text in texts]
        
        device = next(self.model.parameters()).device
        
        self.tokenizer.padding_side = "left" 
        
        inputs = self.tokenizer(
            prompts, 
            return_tensors="pt", 
            padding=True, 
            truncation=True, 
            max_length=512
        ).to(device)

        outputs = self.model.generate(
            **inputs,
            max_new_tokens=80, 
            temperature=0.0,   
            do_sample=False,
            pad_token_id=self.tokenizer.pad_token_id,
            eos_token_id=self.tokenizer.eos_token_id
        )

        input_length = inputs.input_ids.shape[1]
        generated_tokens = outputs[:, input_length:]
        decoded_responses = self.tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)

        for response in decoded_responses:
            pred = {aspect: "No Opinion" for aspect in self.aspects}
            
            match = re.search(r'\{.*?\}', response, re.DOTALL)
            if match:
                parsed_json = json.loads(match.group(0))
                for aspect in self.aspects:
                    if aspect in parsed_json and parsed_json[aspect] in self.valid_labels:
                        pred[aspect] = parsed_json[aspect]
        
            predictions.append(pred)

        return predictions