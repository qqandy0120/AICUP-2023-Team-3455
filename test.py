import json
from pathlib import Path
path = "arg.json"


d = {"dsjkfl":1}
with open(path, 'w') as f:
    json.dump(d, f)
