from datasets import load_dataset
#,load_from_disk,save_to_disk

data_path = './datas/Bespoke-Stratos-17k'

dataset = load_dataset(data_path,split='train')

##map_func

def trans_data(example):
    system_prompt = example['system']
    messages = example['conversations']
    res = list()
    system_dict = dict()
    system_dict['role'] = 'system'
    system_dict['content'] = system_prompt
    res.append(system_dict)
    for mess in messages:
        tmp_dict = dict()
        tmp_dict['role'] = 'user' if 'user' == mess.get('from',None) else 'assistant'
        tmp_dict['content'] = mess.get('value',None)
        res.append(tmp_dict)
    return {'messages':res}

dataset = dataset.map(trans_data,remove_columns=['system','conversations'])
print(dataset)
print(dataset[0])

dataset = dataset.shuffle(42)

train_test_split = dataset.train_test_split(test_size=0.05)

print(train_test_split)

new_data_path = data_path+'-process'
train_test_split.save_to_disk(new_data_path)
#print(train_test_split['train'][0])
