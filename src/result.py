"""import json
from result import Result

# Save
result_obj = Result(accuracy=0.8, precision=0.75, recall=0.85, f1=0.8, confusion_matrix=[[100, 10], [5, 200]], classification_report="...")

with open("result.json", "w") as json_file:
    json.dump(result_obj.__dict__(), json_file, indent=4)

# Load
with open("result.json", "r") as json_file:
    loaded_result_dict = json.load(json_file)

loaded_result_obj = Result(
    accuracy=loaded_result_dict["accuracy"],
    precision=loaded_result_dict["precision"],
    recall=loaded_result_dict["recall"],
    f1=loaded_result_dict["f1"],
    confusion_matrix=loaded_result_dict["confusion_matrix"],
    classification_report=loaded_result_dict["classification_report"]
)"""

class Result:
    accuracy: float
    precision: float
    recall: float
    f1: float
    confusion_matrix: []
    classification_report: str

    def __init__(self, accuracy: float, precision: float, recall: float, f1: float, confusion_matrix: [], classification_report: str):
        self.accuracy = accuracy
        self.precision = precision
        self.recall = recall
        self.f1 = f1
        self.confusion_matrix = confusion_matrix
        self.classification_report = classification_report
    
    def __dict__(self):
        return {
            "accuracy": self.accuracy,
            "precision": self.precision,
            "recall": self.recall,
            "f1": self.f1,
            "confusion_matrix": self.confusion_matrix,
            "classification_report": self.classification_report
        }    