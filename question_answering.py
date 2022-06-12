import pandas as pd
import torch
import transformers
from constants import MODELS
from dataset import CachimboDataset
from transformers import BertForSequenceClassification
import os
from transformers import Trainer, TrainingArguments

def qa_train(args):

	train_df = pd.read_csv(args.train_data, sep="\t", header='infer')
	dev_df = pd.read_csv(args.dev_data, sep="\t", header='infer')

	transformers.set_seed(args.seed)
	device = torch.device('cuda' if torch.cuda.is_available() and args.gpu else 'cpu')

	max_length = args.src_max_length
	epochs = args.epochs
	lr = args.learning_rate
	batch_size = args.batch_size
	grad_accum = args.accum_steps
	warmup_ratio = args.warmup_ratio
	weight_decay = args.weight_decay

	train_dataset = CachimboDataset(train_df, max_length, bert_model= MODELS[args.pretrained_model])
	dev_dataset = CachimboDataset(dev_df, max_length, bert_model= MODELS[args.pretrained_model])


	model = BertForSequenceClassification.from_pretrained(MODELS[args.pretrained_model], num_labels=1)
	model = model.to(device)

	output_dir = args.save_dir
	log_dir = output_dir + "_log"

	print(f'Creating {output_dir} and {log_dir} folders ')
	os.makedirs(log_dir, exist_ok = True)
	os.makedirs(output_dir, exist_ok = True)

	training_args = TrainingArguments(
							output_dir=output_dir,
							overwrite_output_dir=True,
							evaluation_strategy="epoch",
							save_strategy="epoch",
							logging_dir=log_dir,
							save_total_limit=10,
							load_best_model_at_end=True,
							logging_steps = args.print_every,
							do_train=True,
							do_eval=True,
							seed=args.seed,
							gradient_accumulation_steps = grad_accum,
							per_device_eval_batch_size=batch_size,
							per_device_train_batch_size=batch_size,
							num_train_epochs=epochs,
							learning_rate=lr,
							warmup_ratio=warmup_ratio,
							weight_decay=weight_decay
							)

	print("Building Trainer ...")
	trainer = Trainer(
				model=model,
				args=training_args,
				train_dataset=train_dataset,
				eval_dataset=dev_dataset,
				)

	print("Trainer built.")

	print("Start training ...")
	trainer.train()
