import json

with open("./configs/config.json") as fp:
    data = json.load(fp)

print(data)
