"""Бейзлайн-решение: gemma-4-E2B-it

На входе:  /workspace/input.pickle
На выходе: /workspace/output.json
Веса:      /workspace/weights (предварительно скачиваются download_weights.py)
"""
import json
import os
import pickle
import torch
from transformers import AutoProcessor, AutoModelForCausalLM


MODEL_DIR = "./weights"
MAX_NEW_TOKENS = 2048
MAX_MODEL_LEN = 4096


SYSTEM_PROMPT = """
You are a professional English-to-Russian translator.

Rules:
- preserve meaning exactly
- preserve terminology consistently across the whole text
- preserve pronoun references and coreference
- preserve paragraph structure, lists, and punctuation
- keep direct speech as direct speech; do not turn quotations into reported speech
- translate quoted UI labels, button names, and menu items consistently everywhere
- keep code, URLs, file paths, commands, placeholders, and identifiers unchanged unless they are ordinary UI words
- keep product names, acronyms, and code fragments in the most natural form for Russian technical text
- output translation only, with no comments or explanations

English text:
{src}

Russian translation:
"""


TRANSLATION_PROMPT = """
English text:
{src}

Russian translation:
"""


REVISION_PROMPT = """
Edit the Russian translation draft.

Fix only:
- mistranslations
- inconsistent terminology
- incorrect pronoun/coreference resolution
- broken direct speech or quotation style
- awkward wording that changes meaning

Important:
- do not add information
- do not omit information
- do not rewrite correct parts unnecessarily
- preserve paragraph structure
- output only the corrected Russian translation
"""


DRAFT_PROMPT = """
English text:
{src}

Draft Russian translation:
{draft}

Corrected Russian translation:
"""


def main() -> None:
    with open("input.pickle", "rb") as f:
        rows = pickle.load(f)

    processor = AutoProcessor.from_pretrained(MODEL_DIR)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_DIR,
        dtype="auto",
        device_map="auto"
    )

    model.eval()
    
    # batch-size 1 for simplicity
    results = []
    for row in rows:
        # 1 step
        translation_prompt = TRANSLATION_PROMPT.format(src=row["src"])
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": translation_prompt},
            ]
        text = processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False
        )

        inputs = processor(text=text, return_tensors="pt").to(model.device)
        input_len = inputs["input_ids"].shape[-1]

        with torch.inference_mode():
            # Generate output
            outputs = model.generate(**inputs, max_new_tokens=MAX_NEW_TOKENS, do_sample=False)

        response = processor.decode(outputs[0][input_len:], skip_special_tokens=False)

        try:
            draft_translation = processor.parse_response(response)['content']
        except Exception:
            draft_translation = processor.decode(outputs[0][input_len:], skip_special_tokens=True)

        draft_translation = draft_translation.strip()

        # 2 step
        draft_prompt = DRAFT_PROMPT.format(src=row["src"], draft=draft_translation)

        messages = [
            {"role": "system", "content": REVISION_PROMPT},
            {"role": "user", "content": draft_prompt},
            ]
        rev_text = processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False
        )
        
        rev_inputs = processor(text=rev_text, return_tensors="pt").to(model.device)
        rev_input_len = rev_inputs["input_ids"].shape[-1]
        
        with torch.inference_mode():
            rev_outputs = model.generate(**rev_inputs, max_new_tokens=MAX_NEW_TOKENS, do_sample=False)
            
        rev_response = processor.decode(rev_outputs[0][rev_input_len:], skip_special_tokens=False)
        
        try:
            final_translation = processor.parse_response(rev_response)['content']
        except Exception:
            final_translation = processor.decode(rev_outputs[0][rev_input_len:], skip_special_tokens=True)

        # Parse output
        results.append({
            'rid': row['rid'],
            'translation': final_translation,
        })
        inputs.to('cpu')

    with open("output.json", "w") as f:
        json.dump(results, f, ensure_ascii=False)


if __name__ == "__main__":
    main()
