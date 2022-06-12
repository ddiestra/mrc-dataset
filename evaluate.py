import pandas as pd
from bert_score import score, scorer

def compute_accuracy(df_ref, df_out):
    return (df_ref[["answer"]] == df_out[["answer"]]).value_counts()[True] / df_ref.shape[0]

def compute_cat1(df_ref, df_out):
    correct = (df_ref[["answer"]] == df_out[["answer"]]).value_counts()[True]
    total = df_ref.shape[0]
    unanswered = df_out[df_out["answer"] == "--"].shape[0]

    return (correct + unanswered * (correct / total)) / total


def compute_bertscore(df_ref, df_out, lng="es"):

    bert_scorer = scorer.BERTScorer(lang=lng)

    hypothesis = df_out["reason"].tolist()
    
    references = []
    for i, item in df_ref.iterrows():
        references.append([item["reason"].strip()])

    scores = []
    for hyp, refs in zip(hypothesis, references):
        if refs[0] != "--":
            P, R, F1 = bert_scorer.score([hyp.strip()], [refs])
            scores.append(F1)
    bert = float(sum(scores) / len(scores))
    return bert


def load_file(path):
    df = pd.read_csv(path, sep="\t", encoding="utf-8")
    return df

df_reference = load_file("reference_both.csv")
df_out = load_file("outputs/sada/results_SADA.csv")

acc = compute_accuracy(df_reference, df_out)
cat1 = compute_cat1(df_reference, df_out)

print(acc, "\t", cat1)

df_out = load_file("outputs/versae_fernandez/test_no_answer_predictions.csv")
acc = compute_accuracy(df_reference, df_out)
cat1 = compute_cat1(df_reference, df_out)
bs = compute_bertscore(df_reference, df_out)


print(acc, "\t", cat1, "\t", bs)
