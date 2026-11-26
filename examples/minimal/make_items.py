"""Generate the public example set (examples/minimal/items.jsonl).

All documents, questions and reference answers here were written by hand for this repository. They describe a
fictional community centre and share no text with any experiment dataset. Licensed Apache-2.0 with the code.
"""
import hashlib
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent


def cid(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:12]


DOCS = {
    "garden": ["社区花园地块分配：每户每年最多申请 1 块地块，申请须在每年 3 月 10 日前向社区中心提交。",
               "社区花园地块使用：地块连续 6 周无人打理的，社区中心可收回并重新分配。"],
    "telescope": ["望远镜预约：会员须提前 48 小时在前台登记预约，每次使用不超过 2 小时。",
                  "望远镜安全：首次使用的会员须先参加 1 次 30 分钟的操作培训，由天文小组组长签字确认。",
                  "望远镜保管：使用结束后须将镜头盖复位并交还钥匙。"],
    "camping": ["露营装备租借：帐篷每天租金 20 元，押金 200 元，归还时检查无损坏后退还押金。",
                "露营装备逾期：逾期归还的，每逾期 1 天加收 1 天租金。"],
}
EVID = {d: [{"cid": cid(t), "source": f"{d}.md", "doc": d, "section": i + 1, "text": t} for i, t in enumerate(ts)]
        for d, ts in DOCS.items()}
C = {d: [e["cid"] for e in es] for d, es in EVID.items()}


def item(i, behavior, question, docs, gold, facts, fact_cids, ref, typ="A"):
    ev = [e for d in docs for e in EVID[d]]
    return {"id": f"ex_{i:02d}", "type": typ, "subtype": "", "behavior": behavior, "question": question,
            "evidence": ev, "gold_cids": gold, "reference_answer": ref, "facts": facts, "fact_cids": fact_cids,
            "forbid": [], "format": None, "source_docs": sorted({e["doc"] for e in ev if e["cid"] in gold}) or docs[:1],
            "no_conflict": False, "conflict_kind": None, "split": "example", "generation_method": "hand_written"}


ITEMS = [
    item(1, "answer", "社区花园地块最晚什么时候申请？", ["garden", "camping"], [C["garden"][0]], ["3月10日"], [C["garden"][0]],
         f"须在每年 3 月 10 日前向社区中心提交申请[{C['garden'][0]}]。"),
    item(2, "answer", "第一次用望远镜需要做什么？要提前多久预约？", ["telescope"], [C["telescope"][0], C["telescope"][1]],
         ["48小时", "30分钟", "组长签字"], [C["telescope"][0], C["telescope"][1], C["telescope"][1]],
         f"须提前 48 小时在前台登记预约[{C['telescope'][0]}]；首次使用须先参加 30 分钟的操作培训，并由天文小组组长签字确认[{C['telescope'][1]}]。",
         typ="B"),
    item(3, "answer", "租帐篷 3 天，晚还 1 天，一共要付多少租金？押金多少？", ["camping", "garden"], [C["camping"][0], C["camping"][1]],
         ["20元", "200元", "加收"], [C["camping"][0], C["camping"][0], C["camping"][1]],
         f"帐篷每天租金 20 元，押金 200 元[{C['camping'][0]}]；逾期 1 天加收 1 天租金[{C['camping'][1]}]。", typ="B"),
    item(4, "refuse", "望远镜可以带回家用吗？", ["telescope"], [], [], [], "根据提供的资料无法回答该问题。"),
    item(5, "answer", "地块多久没人打理会被收回？每户最多几块？", ["garden"], [C["garden"][0], C["garden"][1]],
         ["1块", "6周"], [C["garden"][0], C["garden"][1]],
         f"每户每年最多申请 1 块地块[{C['garden'][0]}]；连续 6 周无人打理的，社区中心可收回[{C['garden'][1]}]。", typ="B"),
    item(6, "answer", "望远镜每次能用多久？", ["telescope", "camping"], [C["telescope"][0]], ["2小时"], [C["telescope"][0]],
         f"每次使用不超过 2 小时[{C['telescope'][0]}]。"),
]

if __name__ == "__main__":
    with open(HERE / "items.jsonl", "w", encoding="utf-8", newline="\n") as f:
        for r in ITEMS:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(json.dumps(C, indent=1))
