import json

if __name__ == "__main__":
    with open("./configs/config.json") as fp:
        data = json.load(fp)

    print(data)
