import re
import json
import asyncio
import hashlib
from typing import Dict, Optional, Any, List, Tuple, Iterable, DefaultDict
from dataclasses import dataclass, field
from collections import defaultdict

import networkx as nx
from langchain_chroma import Chroma

from agents.memo_structure import Sub_memo_layer, MemoStructure
from eval_envs.base_envs import Basic_Recorder
from utils.hire_agent import Agent, Embedding


DIRECTIONS = ["north", "south", "east", "west", "up", "down"]
VERB_WHITELIST = {
    "go", "open", "close", "unlock", "lock", "take", "drop", "put", "insert", "remove",
    "examine", "look", "read", "chop", "dice", "slice", "grill", "fry", "cook", "prepare",
    "wear", "use", "move", "push", "pull", "turn", "switch", "inspect", "boil", "bake",
    "inventory"
}
CREDITABLE_VERBS = {
    "open", "close", "unlock", "lock", "take", "drop", "put", "insert", "remove",
    "read", "wear", "use", "move", "push", "pull", "turn", "switch",
    "chop", "dice", "slice", "grill", "fry", "cook", "prepare", "boil", "bake", "examine"
}
COOKING_VERBS = {"cook", "prepare", "grill", "fry", "boil", "bake", "chop", "dice", "slice"}

# Remove 'exit' from door nouns to avoid duplication with doorless pattern
EXIT_NOUNS = "door|hatch|portal|passageway|gate|archway|staircase|ladder|trapdoor|arch|gateway|entranceway|entrance|doorway|threshold"
KEY_SYNONYMS = {"key", "passkey", "latchkey", "latch key", "pass-code key", "pass code key", "keycard", "key card"}
OBJ_SYNONYMS = {
    "latchkey": "key",
    "latch key": "key",
    "passkey": "key",
    "pass code key": "key",
    "pass-code key": "key",
    "keycard": "key",
    "key card": "key",
}


def _safe_get_action(step: Dict[str, Any]) -> str:
    act = step.get("action_took")
    candidate = ""
    if isinstance(act, list) and act:
        candidate = str(act[-1])
    elif isinstance(act, str):
        candidate = act
    else:
        return ""
    tokens = re.split(r"(?:;|\n| and then )+", candidate, flags=re.IGNORECASE)
    tokens = [t.strip().strip('"').strip("'") for t in tokens if t and t.strip()]
    return tokens[-1].strip() if tokens else ""


def _split_atomic_actions(action: Any) -> List[str]:
    if isinstance(action, list) and action:
        action = action[-1]
    s = str(action or "").strip()
    if not s:
        return []
    toks = re.split(r"(?:;|\n| and then )+", s, flags=re.IGNORECASE)
    toks = [t.strip().strip('"').strip("'") for t in toks if t and t.strip()]
    return toks


def _extract_room_name(obs: str) -> Optional[str]:
    if not obs:
        return None
    m = re.search(r"-=\s*(.+?)\s*=-", obs)
    return m.group(1).strip() if m else None


def _canon_room_name(name: Optional[str]) -> Optional[str]:
    if not name:
        return None
    return re.sub(r"\s+", " ", name.strip()).title()


def _normalize_object_lemma(obj: str) -> str:
    o = (obj or "").strip().lower()
    return OBJ_SYNONYMS.get(o, o)


def _head_lemma(obj: str) -> str:
    """
    Extract head noun lemma from compound names, using last token heuristic and synonyms.
    E.g., 'non-euclidean passkey' -> 'key'; 'rusty iron key' -> 'key'
    """
    s = (obj or "").strip().lower()
    if not s:
        return ""
    tokens = re.split(r"[\s\-]+", s)
    tokens = [t for t in tokens if t and t.isalpha()]
    if not tokens:
        return ""
    head = tokens[-1]
    head = OBJ_SYNONYMS.get(head, head)
    if head in {"key", "door", "gate", "portal", "hatch"}:
        return head
    # fallback: if any token maps to known synonym like key, prefer it
    for t in reversed(tokens):
        tl = OBJ_SYNONYMS.get(t, t)
        if tl in {"key"}:
            return tl
    return head


def _hyphenate_desc(desc: str) -> str:
    d = (desc or "").strip().lower()
    d = re.sub(r"\s+", " ", d)
    d = d.replace(" ", "-")
    d = re.sub(r"-+", "-", d)
    return d


def normalize_command(step: str) -> Optional[str]:
    if not step:
        return None
    s = re.sub(r"\s+", " ", step.strip().lower())

    # Single-word commands
    if s in {"look", "inventory"}:
        return s

    # Movement: "go east", "go east to kitchen"
    m = re.match(r"go\s+(north|south|east|west|up|down)\b", s)
    if m:
        return f"go {m.group(1)}"

    # Unlock: "unlock <desc> <exit-noun> with <key>"
    m = re.match(rf"unlock\s+(?:the\s+)?(?P<desc>[\w\s-]+?)\s+(?P<noun>{EXIT_NOUNS})\s+with\s+(?P<key>[\w\s-]+)$", s)
    if m:
        desc = _hyphenate_desc(m.group("desc"))
        key = re.sub(r"\s+", " ", m.group("key")).strip()
        noun = (m.group("noun") or "door").strip()
        if desc and key:
            return f"unlock {desc} {noun} with {key}"

    # Unlock dir door with key
    m = re.match(r"unlock\s+(north|south|east|west|up|down)\s+(?:door|portal|gate|hatch)\s+with\s+([\w\s-]+)$", s)
    if m:
        return f"unlock {m.group(1)} door with {m.group(2)}"

    # Open/Close desc exit: "open red portal"
    m = re.match(rf"(open|close)\s+(?:the\s+)?(?P<desc>[\w\s-]+?)\s+(?P<noun>{EXIT_NOUNS})$", s)
    if m:
        verb = m.group(1)
        desc = _hyphenate_desc(m.group("desc"))
        noun = (m.group("noun") or "door").strip()
        if desc:
            return f"{verb} {desc} {noun}"

    # Open/Close dir door: "open west door"
    m = re.match(r"(open|close)\s+(north|south|east|west|up|down)\s+(?:door|portal|gate|hatch)$", s)
    if m:
        return f"{m.group(1)} {m.group(2)} door"

    # Take / Examine / Read
    m = re.match(r"(take|examine|read)\s+(?:the\s+|a\s+|an\s+)?([\w\s-]+)$", s)
    if m:
        verb = m.group(1)
        obj = re.sub(r"\s+", " ", m.group(2)).strip()
        if obj:
            return f"{verb} {obj}"

    return None


def _extract_exits_rich(obs: str) -> List[Dict[str, Optional[str]]]:
    """
    Returns list of dicts with keys:
    - dir: direction in DIRECTIONS
    - state: 'open', 'closed', 'locked', or ''
    - door_desc: descriptor tokens normalized (hyphenated)
    - noun: the exit noun used (door, portal, gate, etc.)
    - kind: 'door' or 'exit'
    - locked: bool
    - doorless: bool (true if the text asserts there is no door for that direction)
    - dedup_reason: reason after dedup
    """
    results: List[Dict[str, Optional[str]]] = []
    if not obs:
        return results

    text = obs
    low = text.lower()
    dir_union = "|".join(DIRECTIONS)
    door_nouns = EXIT_NOUNS  # excludes 'exit'

    # Detect explicit "no door" negations
    no_door = set()
    neg_pat = re.compile(
        rf"\b(?:there\s+is|there's)?\s*no\s+(?:{door_nouns})\s+(?:to|leading)\s+(?:the\s+)?(?P<dir>{dir_union})\b",
        re.IGNORECASE,
    )
    for m in neg_pat.finditer(text):
        no_door.add(m.group("dir").lower())

    # Door-like with nouns excluding 'exit'
    door_like = re.compile(
        rf"There is an?\s+(?:(open|closed|locked)\s+)?(?:(?P<desc>[\w\s-]+?)\s+)?(?P<noun>{door_nouns})\s+(?:to|leading)\s+(?:the\s+)?(?P<dir>{dir_union})",
        re.IGNORECASE,
    )

    # Doorless 'exit' phrasing
    doorless = re.compile(
        rf"There is an?\s+(?:(?P<desc>[\w\s-]+?)\s+)?(?P<noun>exit)\s+(?:to|leading)\s+(?:the\s+)?(?P<dir>{dir_union})",
        re.IGNORECASE,
    )

    # Generic hints
    hint_moves = re.compile(
        rf"\b(?:you\s+(?:can|should)\s+)?(?:try\s+)?(?:go|going|head|move|proceed)\s+(?P<dir>{dir_union})\b",
        re.IGNORECASE,
    )

    temp: DefaultDict[str, List[Dict[str, Any]]] = defaultdict(list)

    for m in door_like.finditer(text):
        state = (m.group(1) or "").lower()
        desc_raw = (m.group("desc") or "").strip().lower()
        desc_raw = re.sub(rf"\b(?:{door_nouns})\b", "", desc_raw).strip()
        desc = _hyphenate_desc(desc_raw)
        direction = m.group("dir").lower()
        noun = (m.group("noun") or "door").lower()
        locked = state == "locked"
        span = low[max(0, m.start() - 60): m.end() + 60]
        if ("won't budge" in span or "won’t budge" in span or "lock" in span) and state != "open":
            locked = True
            if not state:
                state = "locked"
        temp[direction].append({
            "dir": direction, "state": state, "door_desc": desc, "noun": noun, "kind": "door",
            "locked": locked, "doorless": direction in no_door, "dedup_reason": ""
        })

    for m in doorless.finditer(text):
        desc_raw = (m.group("desc") or "").strip().lower()
        desc = _hyphenate_desc(desc_raw + " exit" if desc_raw else "exit")
        direction = m.group("dir").lower()
        noun = (m.group("noun") or "exit").lower()
        temp[direction].append({
            "dir": direction, "state": "", "door_desc": desc, "noun": noun, "kind": "exit",
            "locked": False, "doorless": direction in no_door, "dedup_reason": ""
        })

    present_dirs = set(temp.keys())
    for m in hint_moves.finditer(text):
        direction = m.group("dir").lower()
        if direction in present_dirs:
            continue
        temp[direction].append({
            "dir": direction, "state": "", "door_desc": "", "noun": "", "kind": "exit",
            "locked": False, "doorless": False, "dedup_reason": "hint-only"
        })

    # Deduplicate by direction
    for d, entries in temp.items():
        # If explicit no-door asserted, keep a single exit entry and drop doors
        if any(e.get("doorless") for e in entries) or d in no_door:
            # Try to keep an 'exit' entry; create one if only door entries exist
            exit_entries = [e for e in entries if e.get("kind") == "exit"]
            if not exit_entries:
                exit_entries = [{
                    "dir": d, "state": "", "door_desc": "exit", "noun": "exit", "kind": "exit",
                    "locked": False, "doorless": True, "dedup_reason": "no-door-asserted"
                }]
            chosen = exit_entries[0]
            chosen["doorless"] = True
            chosen["dedup_reason"] = "no-door-asserted"
            results.append(chosen)
            continue

        # Otherwise prefer a door if available; prefer described door over plain
        door_entries = [e for e in entries if e.get("kind") == "door"]
        exit_entries = [e for e in entries if e.get("kind") == "exit"]
        if door_entries:
            door_entries.sort(key=lambda e: (0 if e.get("door_desc") else 1, 0 if e.get("state") else 1))
            chosen = door_entries[0]
            chosen["dedup_reason"] = "prefer-door-over-exit" if exit_entries else "unique"
            results.append(chosen)
        elif exit_entries:
            chosen = exit_entries[0]
            chosen["dedup_reason"] = "only-exit"
            results.append(chosen)

    return results


def _hash_text(s: str) -> str:
    return hashlib.sha1(s.encode("utf-8")).hexdigest()[:16]


def _sanitize_and_parse_action(action: str) -> Tuple[str, str]:
    action = (action or "").strip().lower()
    if not action:
        return "", ""

    if action.startswith("pick up"):
        action = "take " + action[len("pick up"):].strip()
    if action.startswith("look at"):
        action = "examine " + action[len("look at"):].strip()

    tokens = action.split()
    if not tokens:
        return "", ""
    verb = tokens[0]
    obj = " ".join(tokens[1:]).strip()

    if "->" in obj:
        return "", ""
    for v in VERB_WHITELIST:
        if re.search(rf"\b{re.escape(v)}\b", obj):
            return "", ""

    if verb not in VERB_WHITELIST:
        return "", ""

    return verb, obj


def _clip_delta(d: float, low: float = -1.0, high: float = 1.0) -> float:
    return max(low, min(high, d))


def _dedup(seq: List[str]) -> List[str]:
    seen = set()
    out = []
    for s in seq:
        if s in seen:
            continue
        seen.add(s)
        out.append(s)
    return out


def extract_objects(init_obs: str) -> Dict[str, List[str]]:
    text = (init_obs or "")
    candidates: List[str] = []

    def _split_items(seg: str) -> List[str]:
        parts = re.split(r",| and |;| but ", seg)
        out = []
        for p in parts:
            p = p.strip()
            p = re.sub(r"^(?:a|an|the)\s+", "", p, flags=re.IGNORECASE)
            p = re.split(r"\b(?:that|which|who|with|without|on|in|at|of)\b", p, flags=re.IGNORECASE)[0].strip()
            if p:
                out.append(p.lower())
        return out

    # "You see ..."
    for m in re.finditer(r"You\s+see\s+(.+?)(?:\.|\n)", text, flags=re.IGNORECASE | re.DOTALL):
        seg = m.group(1)
        candidates.extend(_split_items(seg))

    # "You can see ..."
    for m in re.finditer(r"You\s+can\s+see\s+(.+?)(?:\.|\n)", text, flags=re.IGNORECASE | re.DOTALL):
        seg = m.group(1)
        candidates.extend(_split_items(seg))

    # "on the floor is/are ..."
    for m in re.finditer(r"on\s+the\s+floor\s+(?:is|are)\s+(.+?)(?:\.|\n)", text, flags=re.IGNORECASE | re.DOTALL):
        seg = m.group(1)
        candidates.extend(_split_items(seg))

    # "There is/are ... on the floor."
    for m in re.finditer(r"There\s+(?:is|are)\s+(.+?)\s+on\s+the\s+floor(?:\.|\n)", text, flags=re.IGNORECASE | re.DOTALL):
        seg = m.group(1)
        candidates.extend(_split_items(seg))

    # "There is/are ... here."
    for m in re.finditer(r"There\s+(?:is|are)\s+(?:a|an|the)?\s*(.+?)\s+here(?:\.|\n)", text, flags=re.IGNORECASE | re.DOTALL):
        seg = m.group(1)
        candidates.extend(_split_items(seg))

    # "You (can) make out ..."
    for m in re.finditer(r"You\s+(?:can\s+)?make\s+out\s+(?:a|an|the)\s+(.+?)(?:\.|\n)", text, flags=re.IGNORECASE | re.DOTALL):
        seg = m.group(1)
        candidates.extend(_split_items(seg))

    # Normalize and dedup
    cset = []
    seen = set()
    for c in candidates:
        c = re.sub(r"\s+", " ", c).strip()
        if not c or c in seen:
            continue
        if re.match(r"^(door|exit|passage|north|south|east|west|up|down)$", c, flags=re.IGNORECASE):
            continue
        seen.add(c)
        cset.append(c)

    key_like = []
    for c in cset:
        for ksyn in KEY_SYNONYMS:
            if re.search(rf"\b{re.escape(ksyn)}\b", c):
                key_like.append(c)
                break

    return {"present_objects": cset, "key_like": _dedup(key_like)}


def _canonicalize_verbs_csv(verbs: str) -> str:
    canon_map = {
        "pick-up": "take",
        "pickup": "take",
        "pick up": "take",
        "grab": "take",
        "inspect": "examine",
        "view": "examine",
        "look at": "examine",
        "check": "examine",
        "scan": "examine",
        "enjoy": "",  # drop non-executable
    }
    toks = [t.strip().lower() for t in (verbs or "").split(",") if t.strip()]
    out = []
    for t in toks:
        mapped = canon_map.get(t, t)
        if mapped:
            out.append(mapped)
    seen = set()
    filt = []
    for v in out:
        if v in seen:
            continue
        seen.add(v)
        filt.append(v)
    return ", ".join(filt)


def _extract_target_room(goal: str) -> str:
    g = (goal or "").lower()
    room_patterns = [
        "cellar", "basement", "attic", "loft", "pantry", "kitchen", "bathroom", "garden",
        "yard", "shed", "garage", "hallway", "corridor", "bedroom", "living room", "parlor",
        "dining room", "office", "study"
    ]
    for rp in room_patterns:
        if re.search(rf"\b{re.escape(rp)}\b", g):
            return rp
    m = re.search(r"from\s+the\s+floor\s+of\s+the\s+([\w\s-]+)", g)
    if m:
        return m.group(1).strip().lower()
    return ""


def _is_outdoor_context(room: str, obs: str) -> bool:
    s = " ".join([(room or "").lower(), (obs or "").lower()])
    return bool(re.search(r"\b(garden|backyard|yard|driveway|street|park|lawn|porch|patio|outside|exterior|field)\b", s))


def _is_indoor_room_name(name: str) -> bool:
    s = (name or "").lower()
    return bool(re.search(r"\b(kitchen|cellar|basement|attic|loft|pantry|bedroom|bathroom|living|dining|office|study|hallway|corridor)\b", s))


@dataclass
class TaskSignatureLayer(Sub_memo_layer):
    layer_intro: str = (
        "TaskSignatureLayer stores parsed task intents and signatures. "
        "It uses an LLM to extract key entities (targets, containers, constraints) "
        "and records them in a vector DB for future retrieval."
    )
    database: Optional[Any] = field(default=None)

    def __post_init__(self):
        if self.database is None:
            self.database = Chroma(collection_name="tw_signatures", embedding_function=Embedding())

    async def retrieve(self, **kwargs):
        init: Dict[str, Any] = kwargs.get("init", {})
        goal = (init.get("goal") or "").strip()
        obs = (init.get("obs") or "").strip()
        start_room_raw = _extract_room_name(obs) or ""
        start_room = _canon_room_name(start_room_raw) or ""

        agent = Agent(
            model="gpt-4o-mini",
            system_prompt=(
                "You extract compact task signatures from text-adventure goals and starting observation. "
                "Return short fields only; do not include explanations."
            ),
            output_schema={
                "targets": {"type": "string", "description": "Key items to obtain or manipulate"},
                "constraints": {"type": "string", "description": "Notable constraints like locked doors or required tools"},
                "verbs": {"type": "string", "description": "Core actions likely needed, comma-separated"},
                "start_room": {"type": "string", "description": "Starting room name if available"},
                "target_room": {"type": "string", "description": "Likely room containing the target, if any"},
                "summary": {"type": "string", "description": "One-line intent summary"}
            },
        )
        signature = {
            "targets": "",
            "constraints": "",
            "verbs": "",
            "start_room": start_room,
            "target_room": "",
            "summary": "",
        }
        try:
            sig = await agent.ask(
                f"Goal:\n{goal}\n\nStarting observation:\n{obs}\n\nExtract the signature."
            )
            if isinstance(sig, dict):
                signature.update({k: (sig.get(k) or "") for k in signature.keys()})
                signature["start_room"] = start_room
        except Exception:
            signature["summary"] = goal[:200]
            signature["verbs"] = "go, take, open, examine"
            signature["start_room"] = start_room

        # Canonicalize verbs and target room fallback
        signature["verbs"] = _canonicalize_verbs_csv(signature.get("verbs", ""))
        if not (signature.get("target_room") or "").strip():
            signature["target_room"] = _extract_target_room(goal)

        signature_text = json.dumps(
            {
                "goal": goal,
                "start_room": signature.get("start_room", ""),
                "targets": signature.get("targets", ""),
                "constraints": signature.get("constraints", ""),
                "verbs": signature.get("verbs", ""),
                "target_room": signature.get("target_room", ""),
                "summary": signature.get("summary", ""),
            },
            ensure_ascii=False,
        )

        compact_similar: List[Dict[str, str]] = []
        seen_keys = set()
        try:
            docs = self.database.similarity_search(signature_text, k=6)
            for d in docs:
                txt = d.page_content or ""
                try:
                    obj = json.loads(txt)
                    key = (obj.get("summary", "") or "") + "|" + (obj.get("targets", "") or "") + "|" + (obj.get("constraints", "") or "")
                    hk = _hash_text(key)
                    if hk in seen_keys:
                        continue
                    seen_keys.add(hk)
                    compact_similar.append({
                        "summary": obj.get("summary", "")[:160],
                        "targets": obj.get("targets", ""),
                        "constraints": obj.get("constraints", "")
                    })
                except Exception:
                    continue
        except Exception:
            compact_similar = []

        return {
            "signature": signature,
            "signature_text": signature_text,
            "similar_signatures": compact_similar[:4],
        }

    async def update(self, **kwargs):
        signature_text: str = kwargs.get("signature_text", "")
        reward: float = float(kwargs.get("reward", 0.0))
        meta = {
            "type": "task_signature",
            "reward": reward,
            "hash": _hash_text(signature_text),
        }
        try:
            self.database.add_texts(texts=[signature_text], metadatas=[meta])
        except Exception:
            pass


@dataclass
class StrategyRecallLayer(Sub_memo_layer):
    layer_intro: str = (
        "StrategyRecallLayer stores short strategy snippets and checklists derived from successful or informative episodes. "
        "It retrieves strategies by embedding similarity to the task signature, preferring same-signature hashes, "
        "and applies lightweight domain filtering (e.g., cooking vs non-cooking)."
    )
    database: Optional[Any] = field(default=None)

    def __post_init__(self):
        if self.database is None:
            self.database = Chroma(collection_name="tw_strategies", embedding_function=Embedding())

    async def retrieve(self, **kwargs):
        signature_text: str = kwargs.get("signature_text", "")
        sig_hash: str = kwargs.get("signature_hash", "")
        k = int(kwargs.get("k", 6))
        strategies_all: List[Dict[str, Any]] = []
        checklists_all: List[Dict[str, Any]] = []

        docs = []
        from_fallback = False
        try:
            retriever = self.database.as_retriever(search_kwargs={"k": k, "filter": {"sig": sig_hash, "type": "strategy"}})
            docs = retriever.get_relevant_documents(signature_text)
        except Exception:
            docs = []

        if not docs:
            try:
                docs = self.database.similarity_search(signature_text or "generic strategy", k=k)
                from_fallback = True
            except Exception:
                docs = []

        # Rank by reward descending; drop very negative
        def _reward_of(d) -> float:
            try:
                return float((d.metadata or {}).get("reward", 0.0))
            except Exception:
                return 0.0

        docs = sorted(docs or [], key=_reward_of, reverse=True)
        docs = [d for d in docs if _reward_of(d) >= -0.05]

        cooking_goal = bool(re.search(r"\b(cook|kitchen|meal|recipe|bake|fry|grill|boil|chop|slice|dice)\b", (signature_text or "").lower()))

        for d in docs or []:
            md = d.metadata or {}
            if md.get("type") != "strategy":
                continue
            subtype = md.get("subtype", "generic")
            content = (d.page_content or "").strip()
            if not content:
                continue

            content_lower = content.lower()
            content_is_cooking = any(v in content_lower for v in COOKING_VERBS) or bool(re.search(r"\b(cookbook|recipe|kitchen)\b", content_lower))
            filtered_out = False
            if from_fallback and (not cooking_goal) and content_is_cooking:
                filtered_out = True

            if subtype == "checklist":
                steps_raw = [s.strip() for s in re.split(r"[;\n]+", content) if s and s.strip()]
                steps_norm = []
                for s in steps_raw:
                    c = normalize_command(s)
                    if c:
                        steps_norm.append(c)
                if steps_norm:
                    checklists_all.append({"steps": _dedup(steps_norm), "filtered": filtered_out, "reward": _reward_of(d)})
            else:
                strategies_all.append({"text": content, "filtered": filtered_out, "reward": _reward_of(d)})

        strategies = [s["text"] for s in strategies_all if not s.get("filtered")]
        checklists = [c["steps"] for c in checklists_all if not c.get("filtered")]

        return {
            "strategies_all": strategies_all,
            "checklists_all": checklists_all,
            "strategies": strategies[:3],
            "checklists": checklists[:2],
        }

    async def update(self, **kwargs):
        text: str = kwargs.get("text", "").strip()
        if not text:
            return
        reward: float = float(kwargs.get("reward", 0.0))
        signature_hash: str = kwargs.get("signature_hash", "")
        subtype: str = kwargs.get("subtype", "generic")
        meta = {
            "type": "strategy",
            "reward": reward,
            "sig": signature_hash,
            "subtype": subtype,
        }
        try:
            self.database.add_texts(texts=[text], metadatas=[meta])
        except Exception:
            pass


@dataclass
class AffordanceLayer(Sub_memo_layer):
    layer_intro: str = (
        "AffordanceLayer stores item/verb outcome snippets with score deltas. "
        "It returns candidate actions for mentioned items and global verb priorities."
    )
    database: Optional[Any] = field(default=None)
    verb_stats: Dict[str, Dict[str, float]] = field(default_factory=dict)

    def __post_init__(self):
        if self.database is None:
            self.database = Chroma(collection_name="tw_affordances", embedding_function=Embedding())
        if not self.verb_stats:
            self.verb_stats = {}

    @staticmethod
    def _compute_deltas(steps: List[Dict[str, Any]]) -> List[Tuple[int, float]]:
        out = []
        for i in range(len(steps) - 1):
            s0 = float(steps[i].get("scores", 0.0))
            s1 = float(steps[i + 1].get("scores", 0.0))
            out.append((i, s1 - s0))
        return out

    def _update_verb_stats(self, verb: str, delta: float):
        stats = self.verb_stats.setdefault(verb, {"count": 0.0, "sum": 0.0})
        stats["count"] += 1.0
        stats["sum"] += float(delta)

    def _global_verb_priorities(self) -> List[Tuple[str, float]]:
        sums: Dict[str, float] = {}
        counts: Dict[str, float] = {}
        for v, s in self.verb_stats.items():
            if s["count"] > 0:
                sums[v] = sums.get(v, 0.0) + s["sum"]
                counts[v] = counts.get(v, 0.0) + s["count"]
        try:
            store = self.database.get()
            metas = store.get("metadatas", []) or []
            for m in metas:
                if not isinstance(m, dict):
                    continue
                if m.get("type") == "affordance":
                    v = str(m.get("verb", "")).lower()
                    if v not in CREDITABLE_VERBS:
                        continue
                    d = float(m.get("delta", 0.0))
                    sums[v] = sums.get(v, 0.0) + d
                    counts[v] = counts.get(v, 0.0) + 1.0
        except Exception:
            pass

        avgs = {v: (sums[v] / counts[v]) for v in sums.keys() if counts.get(v, 0.0) > 0}
        items = sorted(avgs.items(), key=lambda kv: kv[1], reverse=True)
        items = [(v, a) for v, a in items if v not in {"look", "examine", "inspect"}]
        return items

    async def retrieve(self, **kwargs):
        signature: Dict[str, str] = kwargs.get("signature", {})
        init_obs: str = kwargs.get("init_obs", "") or ""
        present_extracted = extract_objects(init_obs)
        present_objs = present_extracted.get("present_objects", [])[:6]
        present_lemmas = {_normalize_object_lemma(x) for x in present_objs}
        present_heads = {_head_lemma(x) for x in present_objs}
        items_text = " ".join([signature.get("targets") or "", signature.get("constraints") or ""] + present_objs).strip()
        verbs_text = signature.get("verbs") or ""
        query = (items_text + " " + verbs_text).strip() or "generic affordances"

        grouped: DefaultDict[str, Dict[str, List[float]]] = defaultdict(lambda: defaultdict(list))
        try:
            k = 32
            docs = self.database.similarity_search(query, k=k)
            for d in docs:
                md = d.metadata or {}
                if md.get("type") != "affordance":
                    continue
                verb = str(md.get("verb", "")).lower()
                obj = _normalize_object_lemma(str(md.get("obj", "")).lower())
                if not verb or not obj or verb not in CREDITABLE_VERBS:
                    continue
                # robust object matching: allow head lemma containment
                obj_head = _head_lemma(obj)
                if not (obj in present_lemmas or obj_head in present_heads or any(obj in p or obj_head in p for p in present_lemmas)):
                    # also allow if it appears in signature targets/constraints
                    targets = (signature.get("targets") or "").lower()
                    constraints = (signature.get("constraints") or "").lower()
                    if (obj not in targets) and (obj_head not in targets) and (obj not in constraints) and (obj_head not in constraints):
                        continue
                delta = float(md.get("delta", 0.0))
                key = obj if obj in present_lemmas else obj_head
                grouped[key][verb].append(delta)
        except Exception:
            pass

        grouped_list = []
        for obj, vmap in grouped.items():
            best_actions = []
            for v, deltas in vmap.items():
                if not deltas:
                    continue
                avg = sum(deltas) / max(1, len(deltas))
                best_actions.append({"verb": v, "avg_delta": round(avg, 4)})
            best_actions.sort(key=lambda x: x["avg_delta"], reverse=True)
            if best_actions:
                grouped_list.append({"item": obj, "best_actions": best_actions[:3], "source": "obs_or_sig"})

        grouped_list.sort(
            key=lambda it: -it["best_actions"][0]["avg_delta"] if it["best_actions"] else 0.0
        )

        verb_priorities = self._global_verb_priorities()

        return {
            "grouped": grouped_list[:3],
            "verb_priorities": [{"verb": v, "avg_delta": round(avg, 4)} for v, avg in verb_priorities[:8]],
            "rank_order": "obs_first",
        }

    async def update(self, **kwargs):
        steps: List[Dict[str, Any]] = kwargs.get("steps", [])
        reward: float = float(kwargs.get("reward", 0.0))

        if len(steps) < 2:
            return

        deltas = self._compute_deltas(steps)
        for i, delta in deltas:
            action_field = steps[i].get("action_took")
            atomic_cmds = _split_atomic_actions(action_field)
            if not atomic_cmds:
                # Fallback single parse
                atomic_cmds = [_safe_get_action(steps[i])]
            for raw in atomic_cmds:
                verb, obj = _sanitize_and_parse_action(raw)
                if not verb:
                    continue
                if verb == "go":
                    # do not store movement affordances, but count verb prior if needed
                    continue
                if verb not in CREDITABLE_VERBS:
                    continue
                # attribute whole step delta to each atomic non-movement action (clipped)
                if abs(delta) < 1e-6:
                    # still count for global verb tendencies with zero delta
                    self._update_verb_stats(verb, 0.0)
                    continue
                delta_c = _clip_delta(delta)
                self._update_verb_stats(verb, delta_c)

                room = _canon_room_name(_extract_room_name(steps[i].get("obs", "") or "")) or ""
                content = f"{verb} {obj}"
                meta = {
                    "type": "affordance",
                    "verb": verb,
                    "obj": _normalize_object_lemma(obj),
                    "room": room,
                    "delta": float(delta_c),
                    "reward": reward,
                }
                try:
                    self.database.add_texts(texts=[content], metadatas=[meta])
                except Exception:
                    pass


@dataclass
class SpatialGraphLayer(Sub_memo_layer):
    layer_intro: str = (
        "SpatialGraphLayer stores a global room connectivity graph across episodes. "
        "Nodes: room names. Edges: observed transitions with direction attributes and weights. "
        "Also tracks per-direction average score deltas and generates navigation suggestions."
    )
    database: Optional[Any] = field(default=None)

    def __post_init__(self):
        if self.database is None:
            self.database = {
                "graph": nx.Graph(),
                "dir_stats": {d: {"count": 0.0, "sum": 0.0} for d in DIRECTIONS},
            }

    def _inc_dir_stat(self, direction: str, delta: float):
        ds = self.database["dir_stats"].setdefault(direction, {"count": 0.0, "sum": 0.0})
        ds["count"] += 1.0
        ds["sum"] += float(delta)

    def _dir_avg(self, direction: str) -> float:
        ds = self.database["dir_stats"].get(direction, {"count": 0.0, "sum": 0.0})
        return (ds["sum"] / ds["count"]) if ds["count"] else 0.0

    @staticmethod
    def _dir_inverse(direction: str) -> str:
        return {
            "north": "south",
            "south": "north",
            "east": "west",
            "west": "east",
            "up": "down",
            "down": "up",
        }.get(direction, "")

    @staticmethod
    def _bias_from_target_room(target_room: str) -> Dict[str, float]:
        tr = (target_room or "").lower()
        bias = {d: 0.0 for d in DIRECTIONS}
        if not tr:
            return bias
        # Lightweight, generalizable priors (stronger bias)
        if re.search(r"\b(cellar|basement)\b", tr):
            bias["down"] += 0.15
        if re.search(r"\b(attic|loft)\b", tr):
            bias["up"] += 0.15
        return bias

    def _nav_suggestions(
        self,
        start_room: str,
        init_obs: str,
        observed: List[Dict[str, Any]],
        key_like: Optional[List[str]],
        target_bias: Optional[Dict[str, float]],
        known_neighbors: Optional[List[Dict[str, Any]]] = None,
        target_room: Optional[str] = "",
        room_exit_stats: Optional[Dict[str, Dict[str, float]]] = None,
    ) -> List[str]:
        suggestions: List[str] = []
        target_bias = target_bias or {d: 0.0 for d in DIRECTIONS}
        known_neighbors = known_neighbors or []
        tr_low = (target_room or "").lower()
        is_outdoor = _is_outdoor_context(start_room, init_obs)
        indoor_target = _is_indoor_room_name(tr_low)

        # If any known neighbor name contains target_room token, prioritize that direction
        if tr_low:
            for nb in known_neighbors:
                room_name = (nb.get("room") or "").lower()
                via = nb.get("via") or ""
                if via and room_name and tr_low in room_name:
                    nc = normalize_command(f"go {via}")
                    if nc:
                        suggestions.append(nc)
                        break  # single top-priority suggestion

        # Prefer handling locked/closed exits first with descriptor-first variants
        for e in observed:
            kind = (e.get("kind") or "").lower()
            d = e["dir"]
            if kind == "exit" or e.get("doorless"):
                # doorless/exit: do not try open/unlock
                continue
            if (e.get("state") in {"closed", "locked"}) or e.get("locked"):
                desc = _hyphenate_desc(e.get("door_desc") or "")
                noun = (e.get("noun") or "door").strip() or "door"
                key_name = (key_like[0] if key_like else None)
                cands = []
                if desc:
                    if key_name and (e.get("locked") or e.get("state") == "locked"):
                        cands.append(f"unlock {desc} {noun} with {key_name}")
                    cands.append(f"open {desc} {noun}")
                # Fallback to direction-based if no desc
                if key_name and (e.get("locked") or e.get("state") == "locked"):
                    cands.append(f"unlock {d} {noun} with {key_name}")
                cands.append(f"open {d} {noun}")
                cands.append(f"go {d}")
                normed = []
                for c in cands:
                    nc2 = normalize_command(c)
                    if nc2:
                        normed.append(nc2)
                return _dedup(normed)

        # Otherwise, choose top exits ranked by per-room stats + global + small target bias + novelty + outdoor indoor bias
        dir_scores = {}
        per_room = room_exit_stats or {}
        for d in DIRECTIONS:
            pr = per_room.get(d, {})
            pr_count = float(pr.get("count", 0.0))
            pr_avg = float(pr.get("sum", 0.0) / pr_count) if pr_count > 0 else 0.0
            novelty_bonus = 0.08 if pr_count == 0 else 0.0
            door_bias = 0.0
            if is_outdoor and indoor_target:
                # favor doors over exits when outdoors aiming indoors
                # will add later per observed kind
                door_bias = 0.05
            dir_scores[d] = pr_avg + self._dir_avg(d) + float(target_bias.get(d, 0.0)) + novelty_bonus + door_bias

        # Re-rank observed exits; add door preference when outdoor
        ranked = sorted(
            [e for e in observed if e.get("dir")],
            key=lambda x: (
                dir_scores.get(x["dir"], 0.0)
                + (0.05 if (is_outdoor and indoor_target and (x.get("kind") == "door")) else 0.0)
            ),
            reverse=True,
        )

        for e in ranked[:2]:
            nc = normalize_command(f"go {e['dir']}")
            if nc:
                suggestions.append(nc)
        if not suggestions and observed:
            nc = normalize_command(f"go {observed[0]['dir']}")
            if nc:
                suggestions.append(nc)
        return _dedup(suggestions)

    async def retrieve(self, **kwargs):
        start_room: str = _canon_room_name(kwargs.get("start_room", "") or "") or ""
        init_obs: str = kwargs.get("init_obs", "") or ""
        key_like: List[str] = kwargs.get("key_like", []) or []
        target_room: str = (kwargs.get("target_room", "") or "").strip().lower()
        G: nx.Graph = self.database["graph"]

        observed_exits_rich = _extract_exits_rich(init_obs)
        # add explicit doorless/dedup_reason and hyphenated descs already included
        observed_dir_info = [{
            "dir": e["dir"],
            "state": e.get("state", "") or "",
            "door_desc": _hyphenate_desc(e.get("door_desc", "") or ""),
            "noun": e.get("noun", "") or "",
            "kind": e.get("kind", "") or "",
            "locked": bool(e.get("locked", False)),
            "doorless": bool(e.get("doorless", False)),
            "dedup_reason": e.get("dedup_reason", "") or "",
        } for e in observed_exits_rich]
        observed_dirs = {e["dir"] for e in observed_exits_rich}

        known_neighbors = []
        archived_known_exits = []
        known_exits = []
        per_room_exit_stats = {}
        if start_room and start_room in G:
            neighbors = []
            for nbr in G.neighbors(start_room):
                if nbr == start_room:
                    continue
                if G.has_edge(start_room, nbr):
                    edge = G.edges[start_room, nbr]
                    dir_map = edge.get("dir_map", {})
                    if start_room in dir_map:
                        neighbors.append((nbr, edge.get("weight", 0.0), dir_map.get(start_room, "")))
            neighbors.sort(key=lambda t: t[1], reverse=True)
            for nbr, _, via in neighbors:
                known_neighbors.append({"room": nbr, "via": via})

            node_exits = G.nodes[start_room].get("exits", {})
            # Reconcile with current observations
            obs_by_dir = {e["dir"]: e for e in observed_exits_rich}
            for d, info in node_exits.items():
                entry_mem = {
                    "dir": d,
                    "state": (info or {}).get("state", "") or "",
                    "door_desc": _hyphenate_desc((info or {}).get("door_desc", "") or ""),
                    "noun": (info or {}).get("noun", "") or "",
                    "kind": (info or {}).get("kind", "") or "",
                    "doorless": bool((info or {}).get("doorless", False)),
                }
                if d in observed_dirs:
                    ob = obs_by_dir[d]
                    # If observed says exit/doorless, override any prior door memory
                    if (ob.get("kind") == "exit") or ob.get("doorless"):
                        entry = {
                            "dir": d,
                            "state": "",
                            "door_desc": _hyphenate_desc(ob.get("door_desc", "") or ""),
                            "locked": False,
                            "noun": ob.get("noun", "") or "exit",
                            "kind": "exit",
                            "doorless": bool(ob.get("doorless", False)),
                            "source": "observed",
                        }
                    else:
                        entry = {
                            "dir": d,
                            "state": ob.get("state", "") or entry_mem["state"],
                            "door_desc": _hyphenate_desc(ob.get("door_desc", "") or entry_mem["door_desc"]),
                            "locked": bool(ob.get("locked", False)),
                            "noun": ob.get("noun", "") or entry_mem["noun"] or "door",
                            "kind": "door",
                            "doorless": False,
                            "source": "observed",
                        }
                    known_exits.append(entry)
                else:
                    entry_mem["locked"] = False
                    entry_mem["source"] = "memory"
                    archived_known_exits.append(entry_mem)

            # pull per-room exit stats
            exit_stats = G.nodes[start_room].get("exit_stats", {}) or {}
            # to flat avg list
            per_room_exit_stats = {
                d: {"count": (exit_stats.get(d, {}) or {}).get("count", 0.0), "sum": (exit_stats.get(d, {}) or {}).get("sum", 0.0)}
                for d in DIRECTIONS
            }

        dir_priority_raw = [{"dir": d, "avg_delta": round(self._dir_avg(d), 4)} for d in DIRECTIONS]
        nonzero = any(abs(x["avg_delta"]) > 1e-6 for x in dir_priority_raw)
        dir_priority = sorted(dir_priority_raw, key=lambda x: x["avg_delta"], reverse=True) if nonzero else []

        # per-room direction priority
        dir_priority_room = []
        for d in DIRECTIONS:
            cnt = float(per_room_exit_stats.get(d, {}).get("count", 0.0))
            sm = float(per_room_exit_stats.get(d, {}).get("sum", 0.0))
            avg = (sm / cnt) if cnt > 0 else 0.0
            dir_priority_room.append({"dir": d, "avg_delta": round(avg, 4), "count": int(cnt)})
        if any(x["count"] > 0 for x in dir_priority_room):
            dir_priority_room.sort(key=lambda x: (x["avg_delta"], -x["count"]), reverse=True)

        nav_suggestions = self._nav_suggestions(
            start_room,
            init_obs,
            observed_exits_rich,
            key_like,
            self._bias_from_target_room(target_room),
            known_neighbors=known_neighbors,
            target_room=target_room,
            room_exit_stats=per_room_exit_stats,
        )

        return {
            "start_room": start_room,
            "observed_exits": observed_dir_info,
            "known_neighbors": known_neighbors,
            "known_exits": known_exits,
            "archived_known_exits": archived_known_exits,
            "direction_priority": dir_priority,
            "direction_priority_room": dir_priority_room,
            "nav_suggestions": nav_suggestions[:3],
        }

    async def update(self, **kwargs):
        steps: List[Dict[str, Any]] = kwargs.get("steps", [])
        if len(steps) == 0:
            return

        G: nx.Graph = self.database["graph"]

        # First pass: record nodes and exits with richer info
        for step in steps:
            room_raw = _extract_room_name(step.get("obs", "") or "")
            room = _canon_room_name(room_raw)
            if not room:
                continue
            if room not in G:
                G.add_node(room, exits={}, exit_stats={})
            exits_rich = _extract_exits_rich(step.get("obs", "") or "")
            node_exits = dict(G.nodes[room].get("exits", {}))
            for e in exits_rich:
                d = e["dir"]
                info = node_exits.get(d, {}) or {}
                # If doorless/exit observed, override to exit and clear door state
                if e.get("kind") == "exit" or e.get("doorless"):
                    node_exits[d] = {
                        "state": "",
                        "door_desc": _hyphenate_desc(e.get("door_desc", "") or ""),
                        "noun": "exit",
                        "kind": "exit",
                        "doorless": True,
                    }
                else:
                    state = e.get("state", "") or info.get("state", "")
                    door_desc = _hyphenate_desc((e.get("door_desc", "") or info.get("door_desc", "")).strip())
                    noun = (e.get("noun", "") or info.get("noun", "") or "door")
                    node_exits[d] = {"state": state, "door_desc": door_desc, "noun": noun, "kind": "door", "doorless": False}
            nx.set_node_attributes(G, {room: {"exits": node_exits}})

        # Second pass: transitions and direction deltas; also per-room exit stats
        for i in range(len(steps) - 1):
            # split atomic to capture last movement too; but transition remains based on room change
            atomic_cmds = _split_atomic_actions(steps[i].get("action_took"))
            chosen_move_dir = None
            for cmd in reversed(atomic_cmds):
                vn, on = _sanitize_and_parse_action(cmd)
                if vn == "go" and on in DIRECTIONS:
                    chosen_move_dir = on
                    break

            room_a = _canon_room_name(_extract_room_name(steps[i].get("obs", "") or ""))
            room_b = _canon_room_name(_extract_room_name(steps[i + 1].get("obs", "") or ""))
            if not room_a or not room_b:
                continue
            if room_a == room_b:
                continue
            if chosen_move_dir:
                direction = chosen_move_dir
                if room_a not in G:
                    G.add_node(room_a, exits={}, exit_stats={})
                if room_b not in G:
                    G.add_node(room_b, exits={}, exit_stats={})
                if not G.has_edge(room_a, room_b):
                    G.add_edge(room_a, room_b, weight=0.0, dir_map={})
                edge = G.edges[room_a, room_b]
                edge["weight"] = float(edge.get("weight", 0.0)) + 1.0
                dir_map = dict(edge.get("dir_map", {}))
                dir_map[room_a] = direction
                inv_dir = self._dir_inverse(direction)
                if inv_dir:
                    dir_map[room_b] = inv_dir
                edge["dir_map"] = dir_map

                s0 = float(steps[i].get("scores", 0.0))
                s1 = float(steps[i + 1].get("scores", 0.0))
                d = s1 - s0
                self._inc_dir_stat(direction, d)

                # per-room exit stats for originating room
                node_stats = dict(G.nodes[room_a].get("exit_stats", {}))
                ds = node_stats.get(direction, {"count": 0.0, "sum": 0.0})
                ds["count"] = float(ds.get("count", 0.0)) + 1.0
                ds["sum"] = float(ds.get("sum", 0.0)) + float(d)
                node_stats[direction] = ds
                nx.set_node_attributes(G, {room_a: {"exit_stats": node_stats}})


def _categorize_command(cmd: str) -> str:
    if not cmd:
        return "other"
    if cmd.startswith("go "):
        return "move"
    if cmd.startswith("open ") or cmd.startswith("unlock ") or cmd.startswith("close "):
        return "door"
    if cmd.startswith("take "):
        return "take"
    if cmd in {"look", "inventory"} or cmd.startswith("examine ") or cmd.startswith("read "):
        return "info"
    return "other"


def _extract_obj_from_cmd(cmd: str) -> str:
    if not cmd:
        return ""
    m = re.match(r"(?:take|examine|read)\s+(.+)$", cmd)
    if m:
        return m.group(1).strip()
    m = re.match(r"(?:open|close|unlock)\s+(.+)$", cmd)
    if m:
        return m.group(1).strip()
    return ""


def _door_cmd_matches_observed(cmd: str, observed_exits: List[Dict[str, Any]]) -> bool:
    cmd_l = (cmd or "").lower()
    is_door_op = any(cmd_l.startswith(v) for v in ["open ", "close ", "unlock "])
    for e in observed_exits:
        d = e.get("dir", "")
        desc = (e.get("door_desc") or "").lower()
        kind = (e.get("kind") or "").lower()
        doorless = bool(e.get("doorless", False))
        if is_door_op and (kind == "exit" or doorless):
            # never match door ops to doorless exits
            continue
        if d and re.search(rf"\b{re.escape(d)}\b", cmd_l):
            return True
        if desc and desc in cmd_l:
            return True
    return False


class TextWorldMemo(MemoStructure):
    def __init__(self):
        super().__init__()
        self.sig_layer = TaskSignatureLayer()
        self.strategy_layer = StrategyRecallLayer()
        self.aff_layer = AffordanceLayer()
        self.spatial_layer = SpatialGraphLayer()

    async def general_retrieve(self, recorder: Basic_Recorder) -> Dict:
        if not recorder or not hasattr(recorder, "init"):
            raise ValueError("Recorder with init is required for retrieval.")
        init = recorder.init or {}
        goal = (init.get("goal") or "").strip()
        obs = (init.get("obs") or "").strip()

        # 1) Signature
        sig_res = await self.sig_layer.retrieve(init={"goal": goal, "obs": obs})
        signature = sig_res.get("signature", {})
        signature_text = sig_res.get("signature_text", "")
        similar_signatures = sig_res.get("similar_signatures", [])
        start_room = _canon_room_name(signature.get("start_room") or _extract_room_name(obs) or "") or ""
        target_room = (signature.get("target_room") or "").strip().lower()
        sig_hash = _hash_text(signature_text)

        # 2) Present objects and keys
        present = extract_objects(obs)
        present_objects = present.get("present_objects", [])
        key_like = present.get("key_like", [])

        # 3) Spatial memory + nav (bias with target room)
        spatial_res = await self.spatial_layer.retrieve(
            start_room=start_room, init_obs=obs, key_like=key_like, target_room=target_room
        )

        # 4) Strategies/checklists retrieval (but only convert to compact hints)
        strat_res = await self.strategy_layer.retrieve(signature_text=signature_text, signature_hash=sig_hash, k=6)
        checklists = strat_res.get("checklists", [])

        # 5) Affordances with synonym grounding
        aff_res = await self.aff_layer.retrieve(signature=signature, init_obs=obs)

        # Build next actions with normalization and gating
        candidates: List[str] = []
        candidates_meta: Dict[str, Dict[str, Any]] = {}

        # Always suggest taking a detected key first
        if key_like:
            take_key_cmd = normalize_command(f"take {key_like[0]}")
            if take_key_cmd:
                candidates.append(take_key_cmd)
                candidates_meta.setdefault(take_key_cmd, {"category": _categorize_command(take_key_cmd), "source": "perception", "priority": 0.92})

        # From spatial suggestions
        for c in spatial_res.get("nav_suggestions", []):
            nc = normalize_command(c)
            if nc:
                candidates.append(nc)
                candidates_meta.setdefault(nc, {"category": _categorize_command(nc), "source": "nav", "priority": 1.0})

        observed_exits = spatial_res.get("observed_exits", []) or []
        locked_or_closed = [
            e for e in observed_exits
            if e.get("kind") != "exit" and not e.get("doorless") and (e.get("locked") or (e.get("state") in {"closed", "locked"}))
        ]
        chosen_exit = None
        if locked_or_closed:
            chosen_exit = sorted(locked_or_closed, key=lambda x: 0 if (x.get("door_desc") or "") else 1)[0]
        key_name = key_like[0] if key_like else None
        if key_name and chosen_exit:
            desc = _hyphenate_desc(chosen_exit.get("door_desc") or "")
            d = chosen_exit.get("dir")
            noun = chosen_exit.get("noun") or "door"
            unlock_cmd = normalize_command(f"unlock {desc} {noun} with {key_name}") if desc else normalize_command(f"unlock {d} {noun} with {key_name}")
            if unlock_cmd:
                candidates.append(unlock_cmd)
                candidates_meta.setdefault(unlock_cmd, {"category": _categorize_command(unlock_cmd), "source": "nav-door", "priority": 0.95})

        # Strategy hints from checklists (normalized + gated), suppress movement to avoid conflicts
        def _gate_checklist_step(step: str) -> Optional[str]:
            step_n = normalize_command(step)
            if not step_n:
                return None
            cat = _categorize_command(step_n)
            if cat == "move":
                return None  # suppress movement hints
            if cat == "door":
                if _door_cmd_matches_observed(step_n, observed_exits):
                    return step_n
                return None
            # object-based commands
            obj = _extract_obj_from_cmd(step_n)
            if not obj:
                return None
            obj_l = obj.lower()
            in_present = any(re.search(rf"\b{re.escape(obj_l)}\b", it.lower()) for it in present_objects)
            in_targets = any(re.search(rf"\b{re.escape(tok.strip().lower())}\b", obj_l) for tok in (signature.get("targets") or "").split(","))
            return step_n if (in_present or in_targets) else None

        strategy_hints: List[str] = []
        for cl in checklists:
            for step in cl:
                g = _gate_checklist_step(step)
                if g:
                    strategy_hints.append(g)
                if len(strategy_hints) >= 3:
                    break
            if len(strategy_hints) >= 3:
                break

        for sh in strategy_hints:
            candidates.append(sh)
            candidates_meta.setdefault(sh, {"category": _categorize_command(sh), "source": "strategy", "priority": 0.55})

        # Opportunistic target action if target visible
        targets = [t.strip() for t in (signature.get("targets") or "").split(",") if t.strip()]
        obs_lower = obs.lower()
        for t in targets:
            t_l = t.lower()
            if t_l and re.search(rf"\b{re.escape(t_l)}\b", obs_lower):
                for v in ["take", "open", "read", "examine"]:
                    nc = normalize_command(f"{v} {t}")
                    if nc:
                        candidates.append(nc)
                        candidates_meta.setdefault(nc, {"category": _categorize_command(nc), "source": "target", "priority": 0.85})
                        break
                break

        # Dedup and category cap: at most one move, one door, one take, plus one info
        capped: List[str] = []
        capped_meta: List[Dict[str, Any]] = []
        seen_cat = {"move": 0, "door": 0, "take": 0, "info": 0}
        # Sort candidates by priority metadata (higher first), default lower
        unique_candidates = _dedup([x for x in candidates if x])
        unique_candidates.sort(key=lambda cmd: float(candidates_meta.get(cmd, {}).get("priority", 0.3)), reverse=True)
        for c in unique_candidates:
            cat = _categorize_command(c)
            if cat in seen_cat and seen_cat[cat] >= 1:
                continue
            seen_cat[cat] = seen_cat.get(cat, 0) + 1
            capped.append(c)
            md = dict(candidates_meta.get(c, {}))
            md["cmd"] = c
            capped_meta.append(md)
            if len(capped) >= 5:
                break

        if not capped and not observed_exits:
            capped.append("look")
            capped_meta.append({"cmd": "look", "category": "info", "source": "fallback", "priority": 0.2})

        return {
            "task_signature": {
                "start_room": start_room,
                "summary": signature.get("summary", ""),
                "targets": signature.get("targets", ""),
                "constraints": signature.get("constraints", ""),
                "verbs": signature.get("verbs", ""),
                "target_room": target_room,
                "similar_signatures": similar_signatures[:3],
            },
            "navigation": {
                "start_room": spatial_res.get("start_room", ""),
                "observed_exits": spatial_res.get("observed_exits", []),
                "known_neighbors": spatial_res.get("known_neighbors", []),
                "known_exits": spatial_res.get("known_exits", []),
                "direction_priority": spatial_res.get("direction_priority", []),
                "direction_priority_room": spatial_res.get("direction_priority_room", []),
            },
            "perception": {
                "present_objects": present_objects[:12],
                "key_like": key_like[:3],
            },
            "affordances": {
                "grouped": aff_res.get("grouped", []),
                "verb_priorities": aff_res.get("verb_priorities", []),
                "rank_order": aff_res.get("rank_order", "obs_first"),
            },
            "strategy_hints": _dedup([s for s in strategy_hints if s and _categorize_command(s) != "move"])[:3],
            "next_actions": capped,
            "next_actions_meta": capped_meta,
        }

    async def general_update(self, recorder: Basic_Recorder) -> None:
        if not recorder or not hasattr(recorder, "init") or not hasattr(recorder, "steps"):
            raise ValueError("Recorder with init and steps is required for update.")

        init = recorder.init or {}
        goal = (init.get("goal") or "").strip()
        obs0 = (init.get("obs") or "").strip()
        start_room = _canon_room_name(_extract_room_name(obs0) or "") or ""
        steps: List[Dict[str, Any]] = recorder.steps or []
        reward = float(getattr(recorder, "reward", 0.0) or 0.0)

        transitions: List[str] = []
        positive_events: List[str] = []
        action_verbs_seen: List[str] = []
        for i in range(len(steps)):
            # multi-action awareness in digest
            action_field = steps[i].get("action_took")
            atomic_cmds = _split_atomic_actions(action_field)
            if not atomic_cmds:
                atomic_cmds = [_safe_get_action(steps[i])]
            verbs_in_step = []
            for cmd in atomic_cmds:
                v, _ = _sanitize_and_parse_action(cmd)
                if v:
                    verbs_in_step.append(v)
                    action_verbs_seen.append(v)
            room = _canon_room_name(_extract_room_name(steps[i].get("obs", "") or "")) or ""
            if i < len(steps) - 1:
                s0 = float(steps[i].get("scores", 0.0))
                s1 = float(steps[i + 1].get("scores", 0.0))
                d = s1 - s0
                if abs(d) > 1e-6:
                    joined_cmd = "; ".join(atomic_cmds) if len(atomic_cmds) > 1 else (atomic_cmds[0] if atomic_cmds else "")
                    positive_events.append(f"{joined_cmd} in {room} -> delta {d:.4f}")
            transitions.append(f"{room}: " + ("; ".join(atomic_cmds) if atomic_cmds else "").strip())

        digest_text = (
            f"Goal: {goal}\nStart: {start_room}\n"
            f"Transitions:\n" + "\n".join(transitions[:120]) + "\n"
            f"Score-affecting events:\n" + "\n".join(positive_events[:40])
        )

        general_strategy_text = ""
        if reward >= 0.2 or len(positive_events) >= 2:
            try:
                agent = Agent(
                    model="gpt-4o-mini",
                    system_prompt=(
                        "Given a text-adventure episode digest, produce a short, reusable strategy. "
                        "Keep it compact and generalizable; avoid specific coordinates and keep object references generic."
                    ),
                    output_schema={"strategy": {"type": "string", "description": "Compact strategy text"}},
                )
                res = await agent.ask(
                    "Episode digest:\n"
                    + digest_text
                    + "\n\nReturn only the strategy text."
                )
                if isinstance(res, dict):
                    general_strategy_text = (res.get("strategy") or "").strip()
            except Exception:
                general_strategy_text = ""

        checklist_text = ""
        verbs_seen = set(action_verbs_seen)
        looks_like_cooking = bool(verbs_seen & COOKING_VERBS) or bool(re.search(r"\bcookbook|recipe|meal|kitchen\b", digest_text, re.IGNORECASE))
        if looks_like_cooking and (reward >= 0.1 or len(positive_events) >= 1):
            try:
                agent2 = Agent(
                    model="gpt-4o-mini",
                    system_prompt=(
                        "Extract a short, reusable checklist (3-7 steps) from a text-adventure episode digest. "
                        "Steps should be generic (no coordinates), imperative verbs, separated clearly."
                    ),
                    output_schema={"steps": {"type": "string", "description": "Checklist steps separated by semicolons"}},
                )
                res2 = await agent2.ask("Episode digest:\n" + digest_text + "\n\nReturn steps separated by semicolons ';'.")
                if isinstance(res2, dict):
                    checklist_text = (res2.get("steps") or "").strip()
            except Exception:
                checklist_text = ""

        # 1) Update spatial graph first (map and direction stats)
        await self.spatial_layer.update(steps=steps)

        # 2) Update affordance experiences
        await self.aff_layer.update(steps=steps, reward=reward)

        # 3) Update task signature store
        try:
            sig_res = await self.sig_layer.retrieve(init={"goal": goal, "obs": obs0})
            signature_text = sig_res.get("signature_text", "")
        except Exception:
            signature_text = json.dumps({"goal": goal, "start_room": start_room}, ensure_ascii=False)
        await self.sig_layer.update(signature_text=signature_text, reward=reward)
        sig_hash = _hash_text(signature_text)

        # 4) Update strategy library
        if general_strategy_text:
            await self.strategy_layer.update(
                text=general_strategy_text,
                reward=reward,
                signature_hash=sig_hash,
                subtype="generic",
            )
        if checklist_text:
            await self.strategy_layer.update(
                text=checklist_text,
                reward=reward,
                signature_hash=sig_hash,
                subtype="checklist",
            )