from transformers import AutoTokenizer, AutoModel
model_address = "/root/autodl-tmp/models/chatglm3-6b"
tokenizer = AutoTokenizer.from_pretrained(model_address, trust_remote_code=True)
model = AutoModel.from_pretrained(model_address, trust_remote_code=True, device='cuda')
model = model.eval()
response, history = model.chat(tokenizer, "你好", history=[])
print(response)