from huggingface_hub import hf_hub_download
hf_hub_download('unsloth/Qwen3.6-35B-A3B-GGUF','Qwen3.6-35B-A3B-UD-Q4_K_M.gguf', local_dir=str(__import__('pathlib').Path.home()/'models'))
print('QWEN DONE', flush=True)
