from argparse import ArgumentParser

import numpy as np
import pandas as pd
from pyannote.core import Annotation, Segment
from pyannote.metrics.errors.identification import IdentificationErrorAnalysis
from sklearn.metrics import f1_score

import wandb


def get_class_maps(num_class):
    if num_class == 5:
        class_name_map = {
            "Introduction_or_Opening": "Opening",
            "Lecture_or_Presentation": "Lecture",
            "Break_or_Transition": "Break",
            "Conclusion_or_Summary": "Conclusion",
            "Opening": "Opening",
            "Lecture": "Lecture",
            "Break": "Break",
            "Conclusion": "Conclusion",
            "Others": "Others",
        }
        class_dict = {
            "Opening": 0,
            "Lecture": 1,
            "Break": 2,
            "Conclusion": 3,
            "Others": 4,
        }
        short_class_names = ["Opening", "Lecture", "Break", "Conclusion", "Others"]
    else:
        class_name_map = {
            "Introduction_or_Opening": "Lecture",
            "Lecture_or_Presentation": "Lecture",
            "Break_or_Transition": "Others",
            "Conclusion_or_Summary": "Lecture",
            "Opening": "Lecture",
            "Lecture": "Lecture",
            "Break": "Others",
            "Conclusion": "Lecture",
            "Others": "Others",
        }
        class_dict = {
            "Opening": 1,
            "Lecture": 1,
            "Break": 0,
            "Conclusion": 1,
            "Others": 0,
        }
        short_class_names = [
            "Others",
            "Lecture",
        ]
    return class_name_map, class_dict, short_class_names


def parse_args():
    parser = ArgumentParser()
    parser.add_argument("--num_class", type=int, default=5)
    return parser.parse_args()


def main():
    args = parse_args()
    num_class = args.num_class
    class_name_map, class_dict, short_class_names = get_class_maps(num_class)

    wandb.init(project="Class stage classification", name=f"Whisper+ChatGPT {num_class}-class")
    test_files = [
        "(list_8)〔高二物化〕線上直播課程(00_01_29_40)",
        "(list_5)〔高一英文〕線上直播課程(00_00_27_45)",
        "Label1(酷課雲公民停課不停學)",
        "X2Download.app - [學盟文教]國九總複習師資-弘理社會(李偉歷史) (128 kbps)",
        "(list_4)〔高一國文〕線上直播課程(00_00_51_59)",
        "List_4【112統測直播】英文",
        "List_2【111統測重點複習課程– 共同英文】",
        "List_5【112統測直播】設計群專二 基礎圖學",
        "停課不停學〔高一物理〕線上直播課程 (1)",
        "List_9【111統測重點複習課程– 數學C】",
    ]
    print("Calculating metrics for test files", test_files)
    label_dir = "data/label"
    hyp_dir = "data/pred"
    ref_list = []
    hyp_list = []

    # parse hyp time to m
    for test_file in test_files:
        ref_df = pd.read_csv(f"{label_dir}/{test_file}.txt", sep="\s+", header=None)
        ref_df[2] = ref_df[2].map(lambda x: class_name_map[x])
        ref_list.append(ref_df)
        hyp_df = pd.read_csv(f"{hyp_dir}/{test_file}.txt", sep="\s+", header=None)
        hyp_df.iloc[:, 0] = pd.to_timedelta(hyp_df.iloc[:, 0]).dt.total_seconds()
        hyp_df.iloc[:, 1] = pd.to_timedelta(hyp_df.iloc[:, 1]).dt.total_seconds()
        hyp_df[2] = hyp_df[2].map(lambda x: class_name_map[x])
        hyp_list.append(hyp_df)

    true_labels = []
    pred_labels = []
    for reference_df, hypothesis_df in zip(ref_list, hyp_list):
        hypothesis_df.iloc[0, 0] = 0
        hypothesis_df.iloc[-1, 1] = reference_df.iloc[-1, 1]
        for i in range(1, hypothesis_df.shape[0]):
            if hypothesis_df.iloc[i, 0] != hypothesis_df.iloc[i - 1, 1]:
                hypothesis_df.iloc[i - 1, 1] = hypothesis_df.iloc[i, 0]
        start = 0
        end = reference_df.iloc[-1, 1]
        ref_duration = 0
        hyp_duration = 0
        reference = Annotation()
        for i, row in reference_df.iterrows():
            reference[Segment(row[0], row[1])] = row[2]
            ref_duration += row[1] - row[0]
        hypothesis = Annotation()
        for i, row in hypothesis_df.iterrows():
            hypothesis[Segment(row[0], row[1])] = row[2]
            hyp_duration += row[1] - row[0]
        error = IdentificationErrorAnalysis().difference(
            reference, hypothesis, uem=Segment(start, end)
        )

        for segment, _ in error.itertracks():
            label = error.get_labels(segment)
            status, true_label, pred_label = next(iter(label))
            duration = segment.duration
            true_labels.extend([true_label] * int(duration))
            pred_labels.extend([pred_label] * int(duration))

    true_labels = [class_dict[label] for label in true_labels]
    pred_labels = [class_dict[label] for label in pred_labels]
    y_true = np.array(true_labels)
    y_pred = np.array(pred_labels)
    macro_f1 = f1_score(y_true, y_pred, average="macro")
    micro_f1 = f1_score(y_true, y_pred, average="micro")
    acc = (y_true == y_pred).sum().item() / len(y_true)
    result = {"test_macro_f1": macro_f1, "test_micro_f1": micro_f1, "test_acc": acc}
    wandb.log(result)
    wandb.log(
        {
            "conf_mat": wandb.plot.confusion_matrix(
                probs=None, y_true=y_true, preds=y_pred, class_names=short_class_names
            )
        }
    )


if __name__ == "__main__":
    main()
