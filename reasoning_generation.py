import pandas as pd
import torch
import transformers
from constants import MODELS
from dataset import ReasoningCachimboDataset, get_loader
from transformers import T5ForConditionalGeneration, AutoTokenizer
import os
from transformers import Adafactor
import math


def evaluate(model, loader, tokenizer, device):

	model.eval()
	total_loss = 0
	n = 0
	for index, data in enumerate(loader, 0):
		batch_size, _, len = data["input_ids"].shape
		ids = data['input_ids'].to(device, dtype = torch.long).view(batch_size, len)
		mask = data['attention_mask'].to(device, dtype = torch.long).view(batch_size, len)
    
		batch_size, _, len = data['labels'].shape
		lm_labels = data['labels'].to(device, dtype = torch.long).view(batch_size, len)
		lm_labels[lm_labels[:,:] == tokenizer.pad_token_id] = -100
		decoder_mask = data['decoder_attention_mask'].to(device, dtype = torch.long).view(batch_size, len)
    
		outputs = model(input_ids = ids,
					attention_mask = mask,
					decoder_attention_mask = decoder_mask,
					labels = lm_labels)
                    
		loss = outputs[0]
		total_loss += loss.item()
		n += 1

	return total_loss / n

def train_epoch(model, epoch, loader, tokenizer, optimizer, print_every, device, accumulation_steps):

	total_loss = 0.0
	n = 0

	model.train()

	for index, data in enumerate(loader, 0):
		batch_size, _, len = data["input_ids"].shape
		ids = data['input_ids'].to(device, dtype = torch.long).view(batch_size, len)
		mask = data['attention_mask'].to(device, dtype = torch.long).view(batch_size, len)

		batch_size, _, len = data['labels'].shape
		lm_labels = data['labels'].to(device, dtype = torch.long).view(batch_size, len)
		lm_labels[lm_labels[:,:] == tokenizer.pad_token_id] = -100
		decoder_mask = data['decoder_attention_mask'].to(device, dtype = torch.long).view(batch_size, len)
    
		outputs = model(input_ids = ids,
					attention_mask = mask,
					decoder_attention_mask = decoder_mask,
					labels = lm_labels)

		loss = outputs[0]
		loss = loss / accumulation_steps
		total_loss += loss.item()
		n += 1
    
		if (index+1) % print_every == 0:
			print(f'Epoch: {epoch+1} | Step: {index+1} | Loss: {total_loss/n} | Peplexity: {round(math.exp(total_loss/n), 3)}')
			n = 0
			total_loss = 0
    
		loss.backward(retain_graph=False)
		if (index+1) % accumulation_steps == 0:
			optimizer.step()
			optimizer.zero_grad()
    
		optimizer.step()
		optimizer.zero_grad()


def freeze_embeddings(model):
	if fixed_embeddings:
		fixed_name = "shared.weight"
		for name, param in model.named_parameters():
			if fixed_name == name:
				param.requires_grad = False
				print("Freezing ", fixed_name)



def rg_train(args):

	train_df = pd.read_csv(args.train_data, sep="\t", header='infer')
	dev_df = pd.read_csv(args.dev_data, sep="\t", header='infer')

	transformers.set_seed(args.seed)
	device = torch.device('cuda' if torch.cuda.is_available() and args.gpu else 'cpu')

	src_length = args.src_max_length
	tgt_length = args.tgt_max_length
	epochs = args.epochs
	lr = args.learning_rate
	batch_size = args.batch_size
	grad_accum = args.accum_steps
	print_every = args.print_every
	path_model = MODELS[args.pretrained_model]

	train_dataset = ReasoningCachimboDataset(train_df, src_length, tgt_length, path_model)
	dev_dataset = ReasoningCachimboDataset(dev_df, src_length, tgt_length, path_model)

	train_loader = get_loader(train_dataset, batch_size)
	dev_loader = get_loader(dev_dataset, batch_size, is_train = False)


	model = T5ForConditionalGeneration.from_pretrained(path_model)
	model = model.to(device)

	if args.fixed_embeddings:
		freeze_embeddings(model)

	optimizer = Adafactor(model.parameters(), lr=lr, relative_step=False)

	tokenizer = AutoTokenizer.from_pretrained(path_model)

	print(f'Creating {args.save_dir} folder')
	os.makedirs(args.save_dir, exist_ok = True)

	best_metric = float('inf')
	best_epoch = -1

	for epoch in range(epochs):

		train_epoch(model, epoch, train_loader, tokenizer, optimizer, print_every, device, grad_accum)
		loss = evaluate(model, dev_loader, tokenizer, device)
		validation_metric = round(math.exp(loss), 3)
  
		print(f'Validation at epoch {epoch+1} - Perplexity: {validation_metric:.3f}')
  
		if validation_metric < best_metric:
			best_metric = validation_metric
			best_epoch = epoch
			print("Saving checkpoint ... Best checkpoint:", str(best_epoch + 1), "(", str(best_metric), ")")
			model.save_pretrained(args.save_dir)
		tokenizer.save_pretrained(args.save_dir)

