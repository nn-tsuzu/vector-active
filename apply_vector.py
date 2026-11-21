import torch
from threading import Thread
from transformers import AutoModelForCausalLM, AutoTokenizer, TextIteratorStreamer

def apply_pre_hook(vector: torch.Tensor, strength: float):
    """
    Pre-Hookは入力に介入するため、その後の正規化で減衰する
    そのため、通常のHookより高いstrength（15.0~）が必要
    """
    def hook(module, args):
        if isinstance(args, tuple) and len(args) > 0:
            hidden_states = args[0]
            
            if isinstance(hidden_states, torch.Tensor) and hidden_states.dim() == 3:
                intervention = vector * strength
                intervention = intervention.to(hidden_states.dtype)
                new_hidden = hidden_states + intervention
                return (new_hidden,) + args[1:]

        return args
    return hook


def apply_vector(name, vector_path, strength):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = AutoModelForCausalLM.from_pretrained(name, dtype="bfloat16", trust_remote_code=True).eval().to(device)
    vectors = torch.load(vector_path)

    handles = []
    for layer_idx, vec in vectors.items():
        layer_module = model.model.language_model.layers[layer_idx]
        h1 = layer_module.register_forward_pre_hook(apply_pre_hook(vec, strength))
        handles.append(h1)
    return model, handles


def model_generetion(name, vector_path, strength):
    model, handles = apply_vector(name, vector_path, strength)
    tokenizer = AutoTokenizer.from_pretrained(name)
    
    messages = [
        {
            "role": "system",
            "content": "You are a cynical AI that called Kikumaru."
        }
    ]

    try:
        while True:
            user_input = input("User: ")

            if not user_input:
                continue
            if user_input.lower() in ["exit", "quit", "q"]:
                break
            
            messages.append({"role": "user", "content": user_input})
            inputs = tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True, return_tensors="pt")
            streamer = TextIteratorStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)

            generation_kwargs = dict(
                inputs,
                streamer=streamer,
                max_new_tokens=256,
                temperature=0.7,
                top_p=0.9,
                do_sample=True
            )
            
            thread = Thread(target=model.generate, kwargs=generation_kwargs)
            thread.start()

            print("assistant: ", end="", flush=True)
            full_response = ""
            
            for new_text in streamer:
                print(new_text, end="", flush=True)
                full_response += new_text
            
            print()
            messages.append({"role": "assistant", "content": full_response})
    
    finally:
        for h in handles:
            h.remove()

model_generetion("nntsuzu/kikka-2b-base", "./vector_file/character_vector_manual.pt", strength=15.0)