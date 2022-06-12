from transformers import BertTokenizer
from transformers import AutoTokenizer, T5ForConditionalGeneration
from torch.utils.data import Dataset, DataLoader

class ReasoningCachimboDataset(Dataset):

    def __init__(self, dataframe, input_length, output_length, 
                 path_model="flax-community/spanish-t5-small"):
      
        self.df = dataframe
        print(path_model)
        self.tokenizer = AutoTokenizer.from_pretrained(path_model)
        self.input_length = input_length
        self.output_length = output_length

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):

        text = self.df.loc[index, "text"]
        question = self.df.loc[index, "question"]
        
        str_source = "Razonar: " + text + " pregunta: " + question + "</s>"
        source = self.tokenizer.encode_plus(str_source, 
                                                  max_length= self.input_length,
                                                  pad_to_max_length=True, 
                                                  return_tensors='pt', 
                                                  truncation=True)
        
        if "reason" in self.df:
          reason = self.df.loc[index, "reason"]
          str_target = reason.strip() + "</s>"
          target = self.tokenizer.encode_plus(str_target, 
                                              max_length= self.output_length, 
                                              pad_to_max_length=True, 
                                              return_tensors='pt', 
                                              truncation=True)
          
          return {"input_ids": source["input_ids"],
                "attention_mask": source["attention_mask"],
                "labels": target["input_ids"],
                "decoder_attention_mask": target["attention_mask"]}
        else:
          return {"input_ids": source["input_ids"],
                "attention_mask": source["attention_mask"]}


class CachimboDataset(Dataset):

    def __init__(self, dataframe, maxlen, 
                 bert_model="dccuchile/bert-base-spanish-wwm-cased"):
      
        self.df = convert_qadataframe(dataframe)
        self.answer = self.df.loc[:, "answer"]
        self.tokenizer = BertTokenizer.from_pretrained(bert_model)
        self.maxlen = maxlen

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):

        text = self.df.loc[index, "text"]
        question = self.df.loc[index, "question"]
        possible_answer = self.df.loc[index, "possible_answer"]
        answer_tensor = torch.tensor(self.answer.loc[index]).type(torch.FloatTensor)
        
        content = text + " " + question + " " + possible_answer
        input_text = self.tokenizer.encode_plus(content, max_length= self.maxlen, 
                                            pad_to_max_length=True, 
                                            return_tensors='pt', 
                                            truncation=True)
        for key in input_text:
          input_text[key] = input_text[key].reshape(-1)
        input_text["labels"] = answer_tensor
        return input_text


def process_row(row):
	data = []
	for index, option in enumerate(["A","B","C","D","E"]):
		if str(row[option]) == "--":
			continue
		text = row["text"]
		question = row["question"]
		possible_answer = row[option]
		answer = (1 if option == row["answer"] else 0)
		reason = row["reason"] #(row["reason"] if option == row["answer"] else "-")
		data.append([text, question, possible_answer, answer, reason])
	return data

def convert_qadataframe(dataframe):
	data = []
	for index, row in enumerate(dataframe.iterrows()):  
		data += process_row(row[1])
	return pd.DataFrame(data, columns=["text", "question", "possible_answer", "answer", "reason"])


def get_loader(dataset, batch_size, is_train=True):
	params = {'batch_size': batch_size,
				'shuffle': is_train,
				'num_workers': 0
			}

	return DataLoader(dataset, **params)



