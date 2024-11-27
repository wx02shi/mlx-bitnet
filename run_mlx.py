import argparse
import time
import mlx.core as mx

from torch_bitnet import BitnetForCausalLM as TorchBitnetForCausalLM
from mlx_bitnet import load_causal_model

parser = argparse.ArgumentParser()
parser.add_argument('--seed', default=0, type=int)
parser.add_argument('--model_name', default="1bitLLM/bitnet_b1_58-large")
parser.add_argument('--prompt', default='Elon musk is', type=str)
parser.add_argument('--maxlen', default=20, type=int)

def generate_text(model, tokenizer, prompt, max_length=50):
    temp = 1.0

    inputs = tokenizer(prompt, return_tensors="pt")
    input_ids = mx.array(inputs.input_ids.numpy())
    attention_mask = mx.array(inputs.attention_mask.numpy())

    start_time = time.time()
    tokens = []
    for token in model.generate(input_ids, attention_mask, temp):
        tokens.append(token)

        if len(tokens) == 1:
            mx.eval(token)
        
        if len(tokens) >= max_length:
            break

    mx.eval(tokens)
    end_time = time.time()

    s = tokenizer.decode([t.item() for t in tokens], skip_special_tokens=True)
    print(s)

    generation_time = end_time - start_time
    num_tokens = len(tokens)
    tokens_per_second = num_tokens / generation_time

    print(f"Generated {num_tokens} tokens in {generation_time:.2f} seconds")
    print(f"Tokens per second: {tokens_per_second:.2f}")

    return s


def main(args):
    torch_model = TorchBitnetForCausalLM.from_pretrained(args.model_name)

    model, tokenizer = load_causal_model(args.model_name)
    model.lm_head.load_weights([
        ("weight", mx.array(torch_model.lm_head.weight.detach().numpy()))
    ])  # TODO: Remind me why do I gotta load these weights again?

    generated_text = generate_text(model, tokenizer, args.prompt)
    print(generated_text)


if __name__ == '__main__':
    args = parser.parse_args()
    mx.random.seed(args.seed)
    main(args)