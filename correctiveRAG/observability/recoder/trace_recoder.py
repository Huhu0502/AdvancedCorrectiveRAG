# trace落盘与查询
import datetime
import json
from pathlib import Path


class TurnRecorder:
    def __init__(self):
        now = datetime.datetime.now()
        data_str = now.strftime("%Y-%m-%d")
        base = Path(__file__).resolve().parent.parent  # observability/
        self.trace_dir = base / 'output' / data_str

    def record(self, trace):
        self.trace_dir.mkdir(parents=True, exist_ok=True)
        path = self.trace_dir / f"{trace.session_id}.json"
        with open(path, "a", encoding="utf-8") as f:
            f.write(json.dumps(trace.to_dict(), ensure_ascii=False) + "\n")

    def get_turns(self,  session_id, date: str = None):
        if date is not None:
            path = f'../output/{date}/{session_id}.json'
        else:
            path = f'../output/{datetime.datetime.now().strftime("%Y-%m-%d")}/{session_id}.json'
        with open(path, "r") as f:
            pass
