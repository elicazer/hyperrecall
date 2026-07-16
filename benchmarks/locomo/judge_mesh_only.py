"""Judge MeshMind predictions only (keeps the vector-RAG baseline fixed).
Prints the meshmind summary as JSON. Used for ablation sweeps."""
import json, os, sys
from pathlib import Path
ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
env_file = Path.home()/".config"/"openclaw"/"gemini.env"
if env_file.exists():
    for line in env_file.read_text().splitlines():
        if line.startswith("export "):
            k,_,v=line[len("export "):].partition("="); os.environ.setdefault(k.strip(),v.strip())
from google import genai
import phase3_judge as J
client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])
_, summary = J.judge_system(client, "meshmind")
print("SUMMARY_JSON:" + json.dumps(summary))
