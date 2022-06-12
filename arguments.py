""" File to hold arguments """
import argparse

# data arguments

parser = argparse.ArgumentParser(description="Main Arguments")

parser.add_argument(
  '-train-data', '--train_data', type=str, required=False, help='Path to train dataset')

parser.add_argument(
  '-dev-data', '--dev_source', type=str,  required=False, help='Path to dev dataset')

parser.add_argument(
  '-test-data', '--test_data', type=str, default="", required=False, help='Path to test dataset')

# training parameters
parser.add_argument(
  '-epochs', '--epochs', type=int, required=False, default=3, help='Number of training epochs')
parser.add_argument(
  '-print-every', '--print_every', type=int, default=10, required=False, help='Print the metrics every training steps')

parser.add_argument(
  '-batch-size', '--batch_size', type=int, required=False, default=8, help='Batch size')
parser.add_argument(
  '-src-max-length', '--src_max_length', type=int, required=False, default=180, help='Max length in encoder')
parser.add_argument(
  '-tgt-max-length', '--tgt_max_length', type=int, required=False, default=80, help='Max length in decoder')


# hyper-parameters
parser.add_argument(
  '-optimizer','--optimizer', type=str, required=False, default="AdamW", choices= ['Adafactor', 'AdamW'], help='Optimizer that will be used')
parser.add_argument(
  '-lr','--learning_rate', type=float, required=False, default=0.0001, help='Learning rate')
parser.add_argument(
  '-adam-epsilon','--adam_epsilon', type=float, default=1.0e-8, required=False, help='Adam epsilon')
parser.add_argument(
  '-warmup-ratio','--warmup_ratio', type=float, required=False, default=0.03, help='Warmup ratio')
parser.add_argument(
  '-weight-decay','--weight_decay', type=float, required=False, default=0.25, help='Weight decay')


parser.add_argument(
  '-accum-steps','--accum_steps', type=int, required=False, default=1, help='Gradient Accumulation')

parser.add_argument(
  '-beam-size','--beam_size', type=int, required=False, default=5, help='Beam search size ')

parser.add_argument(
  '-seed', '--seed', type=int, required=False, help='Seed')
parser.add_argument(
  '-gpu','--gpu', action='store_true', required=False, help='Use GPU or CPU')

parser.add_argument(
  '-fixed-embed','--fixed_embeddings', action='store_true', required=False, help='Fixed embeddings or not')

parser.add_argument(
  '-save-dir','--save_dir', type=str, required=True, default="/content/", help='Output directory')

parser.add_argument(
  '-model','--model', type=str, required=False, default="pierreguillou/gpt2-small-portuguese", help='Path for a model file')

parser.add_argument(
  '-pretrained-model', '--pretrained-model', default='bert', type=str, choices=['bert','t5'], required=False, help='Pretrained model to be used')

parser.add_argument(
  '-task', '--task', default='qa', type=str, choices=['qa','reasoning'], required=False, help='Question-Answering or Reasoning Generation')

def get_args():
  args = parser.parse_args()
  return args
