from vllm import LLM, SamplingParams

prompts = [
    'Hello, my name is',
    'The president of the United States is',
    'The capital of France is',
    'The future of AI is',
]
sampling_params = SamplingParams(temperature=0.8, top_p=0.95)

llm = LLM(model='decapoda-research/llama-7b-hf', gpu_memory_utilization=0.9)

# Print the outputs.
for i in range(10):
    outputs = llm.generate(prompts, sampling_params)
    for output in outputs:
        prompt = output.prompt
        generated_text = output.outputs[0].text
        print(f'Prompt: {prompt!r}, Generated text: {generated_text!r}')
