from arguments import get_args
from question_answering import qa_train
from reasoning_generation import rg_train

def run(args):
	if args.task == "qa":
		qa_train(args)
	else:
		rg_train(args)		


if __name__ == "__main__":
	args = get_args()
	run(args)

