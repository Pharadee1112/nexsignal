"""
NexSignal prototype (synthetic data only, not CPF data).

Flow:  event -> which suppliers are hit -> read supplier replies -> impact on stock -> options
Ours:  (1) reading free-text supplier replies in any language, (2) turning them into stock impact.

Run:
  python nexsignal.py          # demo + evaluate the rule-based reader (no AI)
  python nexsignal.py llm      # also evaluate the LLM reader (needs ANTHROPIC_API_KEY)
"""
import json
import math
import os   
import re
import sys

suppliers = {s["id"]: s for s in json.load(open("data/suppliers.json", encoding="utf-8"))}
events = json.load(open("data/events.json", encoding="utf-8"))
replies = json.load(open("data/replies.json", encoding="utf-8"))


# ---------- step 2: which suppliers does an event hit ----------
def distance_km(lat1, lon1, lat2, lon2):
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dp, dl = p2 - p1, math.radians(lon2 - lon1)
    a = math.sin(dp / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    return 6371 * 2 * math.asin(math.sqrt(a))


def affected(event):
    hit = []
    for s in suppliers.values():
        by_port = s["port"] in event["ports"]
        by_area = event["lat"] is not None and \
            distance_km(event["lat"], event["lon"], s["lat"], s["lon"]) <= event["radius_km"]
        if by_port or by_area:
            hit.append(s)
    return hit


# ---------- step 3a: rule-based reader (baseline, no AI) ----------
NORMAL = ["no impact", "not affected", "normal", "all good", "ปกติ", "ไม่กระทบ", "ไม่มีผลกระทบ",
          "sem impacto", "sin impacto", "bình thường", "không bị ảnh hưởng"]
UNKNOWN = ["not sure", "still assessing", "hard to say", "reply later", "ยังไม่แน่ใจ", "ติดต่อ",
           "não sabemos", "no sabemos", "chưa xác định"]


def read_rules(text):
    t = text.lower()
    if any(k in t for k in UNKNOWN):
        return {"status": "Unknown", "delay_days": None, "deliverable_pct": None}
    if any(k in t for k in NORMAL):
        return {"status": "Normal", "delay_days": 0, "deliverable_pct": 100}
    d = re.search(r"(\d+)\s*(days?|วัน|dias|días|ngày)", t)
    p = re.search(r"(\d+)\s*%", t)
    return {"status": "Disrupted",
            "delay_days": int(d.group(1)) if d else 0,
            "deliverable_pct": int(p.group(1)) if p else 100}


# ---------- step 3b: LLM reader (the AI part) ----------
PROMPT = """You read a supplier's reply about a disruption and return ONLY a JSON object:
{{"status": "Normal" | "Disrupted" | "Unknown",
  "delay_days": integer or null,
  "deliverable_pct": integer 0-100 or null}}
Rules:
- Normal: no impact. Use delay_days 0 and deliverable_pct 100.
- Unknown: the supplier does not yet know the impact. Use null for both numbers.
- Disrupted: any delay or reduced quantity. Convert words to numbers ("a week" = 7, "two weeks" = 14, "half" = 50).
  If the reply says the quantity is complete or full, deliverable_pct is 100; if it says timing is unchanged, delay_days is 0.
  If a number is not stated and cannot be inferred, use null. Never guess.
- Ignore numbers about the past or about when they will reply.
Reply: {text}"""


def read_llm(text):
    import anthropic
    client = anthropic.Anthropic()
    msg = client.messages.create(
        model=os.environ.get("NEXSIGNAL_MODEL", "claude-haiku-5-5"),
        max_tokens=200,
        messages=[{"role": "user", "content": PROMPT.format(text=text)}],
    )
    raw = msg.content[0].text
    return json.loads(raw[raw.find("{"): raw.rfind("}") + 1])


# ---------- step 4: impact on stock ----------
def impact(s, r):
    if r["status"] == "Unknown" or r["delay_days"] is None or r["deliverable_pct"] is None:
        return {"level": "CHECK", "buffer_days": None, "short_tons": None}
    buffer_days = s["stock_days"] - (s["eta_days"] + r["delay_days"])
    short_pct = 100 - r["deliverable_pct"]
    if buffer_days < 0:
        level = "CRITICAL"
    elif buffer_days <= 2 or short_pct >= 30:
        level = "HIGH"
    elif buffer_days <= 4 or short_pct > 0:
        level = "MEDIUM"
    else:
        level = "LOW"
    return {"level": level, "buffer_days": buffer_days,
            "short_tons": round(s["order_tons"] * short_pct / 100)}


# ---------- step 5: options from unaffected suppliers of the same material ----------
def options(s, hit_ids):
    alt = [o for o in suppliers.values()
           if o["material"] == s["material"] and o["id"] != s["id"] and o["id"] not in hit_ids]
    return sorted(alt, key=lambda o: o["eta_days"])


# ---------- evaluation ----------
def evaluate(reader, name):
    fields = ["status", "delay_days", "deliverable_pct"]
    right = {f: 0 for f in fields}
    all_right = level_right = dangerous = 0
    misses = []
    for r in replies:
        got = reader(r["text"])
        ok = [got.get(f) == r["truth"][f] for f in fields]
        for f, o in zip(fields, ok):
            right[f] += o
        all_right += all(ok)
        s = suppliers[r["supplier"]]
        got_level, true_level = impact(s, got)["level"], impact(s, r["truth"])["level"]
        level_right += got_level == true_level
        # dangerous = real risk is HIGH/CRITICAL/CHECK but the system says it is fine (LOW/MEDIUM)
        dangerous += true_level in ("HIGH", "CRITICAL", "CHECK") and got_level in ("LOW", "MEDIUM")
        if not all(ok):
            misses.append((r["id"], r["text"][:60], got))
    n = len(replies)
    print(f"\n{name}  ({n} replies, 5 languages)")
    print(f"  status correct          {right['status']:>3}/{n}  {right['status'] / n:.0%}")
    print(f"  delay_days correct      {right['delay_days']:>3}/{n}  {right['delay_days'] / n:.0%}")
    print(f"  deliverable_pct correct {right['deliverable_pct']:>3}/{n}  {right['deliverable_pct'] / n:.0%}")
    print(f"  all 3 fields correct    {all_right:>3}/{n}  {all_right / n:.0%}")
    print(f"  risk level correct      {level_right:>3}/{n}  {level_right / n:.0%}")
    print(f"  DANGEROUS misses        {dangerous:>3}/{n}  (real risk high, system says fine)")
    return misses


def demo(event_id="E1", reply_id=1):
    e = next(x for x in events if x["id"] == event_id)
    hit = affected(e)
    print(f"EVENT  {e['title']}")
    print(f"  suppliers hit: {', '.join(f'{s['name']} ({s['material']})' for s in hit)}")
    r = next(x for x in replies if x["id"] == reply_id)
    s = suppliers[r["supplier"]]
    got = read_rules(r["text"])
    imp = impact(s, got)
    print(f"  reply from {s['name']}: \"{r['text']}\"")
    print(f"  read as: {got}")
    print(f"  stock lasts {s['stock_days']} days, ship due in {s['eta_days']} + {got['delay_days']} days"
          f"  -> buffer {imp['buffer_days']} days, short {imp['short_tons']:,} tons  => {imp['level']}")
    alts = options(s, {h["id"] for h in hit})
    print("  options: " + ("; ".join(f"{o['name']} ({o['country']}, {o['eta_days']} days)" for o in alts)
                         or "none in list") + "  -> procurement decides")


if __name__ == "__main__":
    demo()
    misses = evaluate(read_rules, "RULE-BASED READER (no AI)")
    print("  examples it got wrong:")
    for i, text, got in misses[:6]:
        print(f"    #{i:<3}{text!r:<64} -> {got}")
    if len(sys.argv) > 1 and sys.argv[1] == "llm":
        evaluate(read_llm, "LLM READER (AI)")
