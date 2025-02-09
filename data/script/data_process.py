from transformers import AutoTokenizer,AutoModelForCausalLM
from datasets import load_dataset


##load data
data_path = './countdown_tasks/'
dataset = load_dataset(data_path,split='train')

dataset = dataset.shuffle(seed=42).select(range(50000))

## load token
model_path = '../models/Qwen/Qwen2.5-3B-Instruct/'
tokenizer = AutoTokenizer.from_pretrained(model_path)
#model = AutoModelForCausalLM.from_pretrained(model_path,device_map="cuda",torch_dtype='bfloat16')

## chat_template
def generate_r1_prompt(numbers, target):
    r1_prefix = [{
        "role": "system",
        "content": "You are a helpful assistant. You first thinks about the reasoning process in the mind and then provides the user with the answer."
      },
      { 
        "role": "user",
        "content": f"Using the numbers {numbers}, create an equation that equals {target}. You can use basic arithmetic operations (+, -, *, /) and each number can only be used once. Show your work in <think> </think> tags. And return the final equation and answer in <answer> </answer> tags, for example <answer> (1 + 2) / 3 = 1 </answer>."
      },
      {
        "role": "assistant",
        "content": "Let me solve this step by step.\n<think>"
      }]
    return {"prompt": tokenizer.apply_chat_template(r1_prefix, tokenize=False, continue_final_message=True), "target": target}

## process data

dataset = dataset.map(lambda x: generate_r1_prompt(x["nums"], x["target"]))
train_test_split = dataset.train_test_split(test_size=0.1)
print('train',train_test_split['train'][0])
print('test',train_test_split['test'][0])
train_test_split.save_to_disk('./chat_data')
