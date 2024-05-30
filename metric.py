import numpy as np
import pandas as pd
from pyannote.core import Annotation, Segment
from pyannote.metrics.errors.identification import IdentificationErrorAnalysis
from sklearn.metrics import f1_score


def accuracy(references, hypotheses):
    total_duration = 0
    correct_duration = 0
    for reference_df, hypothesis_df in zip(references, hypotheses):
        start = 0
        end = reference_df.iloc[-1, 1]
        reference = Annotation()
        for i, row in reference_df.iterrows():
            reference[Segment(row[0], row[1])] = row[2]
        hypothesis = Annotation()
        for i, row in hypothesis_df.iterrows():
            hypothesis[Segment(row[0], row[1])] = row[2]
        error = IdentificationErrorAnalysis().difference(
            reference, hypothesis, uem=Segment(start, end)
        )
        for segment, _ in error.itertracks():
            label = error.get_labels(segment)
            status, true_label, pred_label = next(iter(label))
            if status == "correct":
                correct_duration += segment.duration
            total_duration += segment.duration

    acc = correct_duration / total_duration

    return acc


def f1_scores(references, hypotheses):
    true_labels = []
    pred_labels = []
    for reference_df, hypothesis_df in zip(references, hypotheses):
        start = 0
        end = reference_df.iloc[-1, 1]
        reference = Annotation()
        for i, row in reference_df.iterrows():
            reference[Segment(row[0], row[1])] = row[2]
        hypothesis = Annotation()
        for i, row in hypothesis_df.iterrows():
            hypothesis[Segment(row[0], row[1])] = row[2]
        error = IdentificationErrorAnalysis().difference(
            reference, hypothesis, uem=Segment(start, end)
        )

        for segment, _ in error.itertracks():
            label = error.get_labels(segment)
            status, true_label, pred_label = next(iter(label))
            duration = segment.duration
            true_labels.extend([true_label] * int(duration))
            pred_labels.extend([pred_label] * int(duration))

    all_classes = set(true_labels+pred_labels)
    y_true = np.array(true_labels)
    y_pred = np.array(pred_labels)
    macro_f1 = f1_score(y_true, y_pred, average="macro")
    micro_f1 = f1_score(y_true, y_pred, average="micro")
    result = {"macro_f1": macro_f1, "micro_f1": micro_f1}

    return result


class_name_map = {
    "Introduction_or_Opening": "Opening",
    "Lecture_or_Presentation": "Lecture",
    "Break_or_Transition": "Break",
    "Conclusion_or_Summary": "Conclusion",
    "Others": "Others",
}

if __name__ == "__main__":
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
    hyp_dir = "data/test"
    ref_list = []
    hyp_list = []
    for test_file in test_files:
        ref_df = pd.read_csv(f"{label_dir}/{test_file}.txt", sep="\s+", header=None)
        ref_df[2] = ref_df[2].map(lambda x: class_name_map[x])
        ref_list.append(ref_df)
        hyp_df = pd.read_csv(f"{hyp_dir}/{test_file}.txt", sep="\s+", header=None)
        hyp_df[2] = hyp_df[2].map(lambda x: class_name_map[x])
        hyp_list.append(hyp_df)

    print(accuracy(ref_list, hyp_list))
    print(f1_scores(ref_list, hyp_list))
