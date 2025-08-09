import json
# Go one directory back
import sys
sys.path.append("../")


def stats_to_json(metrics, model_name, tr):
        
        # Loads json file
        with open('stats.json', 'r') as f: data = json.load(f)
        
        # If the model isn't in the json file
        if model_name not in data.keys(): 
            replace_model_in_data(data, model_name, metrics, tr)
            print(f"Stats for {model_name} added for the first time")


        # If the accuarcy is better than rewrite it
        if metrics["asc"] > data[model_name]["metrics"]["accuracy"]:

            replace_model_in_data(data, model_name, metrics, tr)
            print(f"Best accuracy for {model_name}")

        elif metrics["asc"] == data[model_name]["metrics"]["accuracy"]:
            if metrics["f1"] > data[model_name]["metrics"]["f1"]:
                replace_model_in_data(data, model_name, metrics, tr)
                print(f"Best accuracy for {model_name} - > f1")
            
        

        # Writing to sample.json
        with open('stats.json', 'w') as f:
            json.dump(data, f, indent=4, separators=(',', ': '))



# Replaca zapis modela
def replace_model_in_data(data, model_name, metrics, tr):
    if metrics["index"] != 6:
        data[model_name] = {
                        "name": model_name,
                        "index": metrics["index"],
                        "even_data": metrics["even_data"],
                        "metrics": {
                            "accuracy": metrics["asc"],
                            "recall": metrics["rs"],
                            "precision": metrics["ps"],
                            "f1": metrics["f1"]
                        },
                        "mini_test": {
                            "real_values_test": str(metrics["real_values_test"]),
                            "prediction_test": str(metrics["prediction_test"])
                        },
                        "test_ratio": tr,
                        "train_time": metrics["train_time"],
                        "predict_time": metrics["predict_time"],
                        "file_name": metrics["dir"]       
        }

    # For nn
    elif metrics["index"] == 6:
        data[model_name] = {
                        "name": model_name,
                        "index": metrics["index"],
                        "even_data": metrics["even_data"],
                        "metrics": {
                            "accuracy": metrics["asc"],
                            "recall": metrics["rs"],
                            "precision": metrics["ps"],
                            "f1": metrics["f1"]
                        },
                        "mini_test": {
                            "real_values_test": metrics["real_values_test"],
                            "prediction_test": metrics["prediction_test"]
                        },
                        "nn_conf":{
                            "validation_split": metrics["validation_split"],
                            "epochs": metrics["epochs"],
                            "batch_size": metrics["batch_size"]
                        },
                        "test_ratio": tr,
                        "train_time": metrics["train_time"],
                        "predict_time": metrics["predict_time"],
                        "file_name": metrics["dir"]       
        }


# V dictionary dj vse za NN