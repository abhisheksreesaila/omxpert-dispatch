from huggingface_hub import snapshot_download
snapshot_download('openai/gpt-oss-20b', allow_patterns=['*.json','*.safetensors','tokenizer*','*.jinja','*.txt'], ignore_patterns=['original/*','metal/*'])
print('GPTOSS DONE', flush=True)
snapshot_download('allenai/OLMoE-1B-7B-0125-Instruct')
print('OLMOE DONE', flush=True)
