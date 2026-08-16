import json

def load_golden_dataset(path:str) -> list[dict]:
    with open(path,'r') as f:
        dataset=json.load(f)
    return dataset

if __name__ == "__main__":
    data = load_golden_dataset("golden_dataset/os.json")
    print(len(data))
    print(data[0])