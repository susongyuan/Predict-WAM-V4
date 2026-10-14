"""Instruction grounding contract: instruction + visible objects + skill catalog -> candidate skill programs.

The VLM/LLM backend is pluggable and untrusted: its output is parsed against a strict JSON schema and every
skill/object is checked against the trained skill catalog and the perceived object list. Nothing here calls
the simulator or reads BDDL ground truth at run time; the goal text comes from the task instruction only.

Ambiguity is decided at the *action* level: identical skill prefixes can execute first. The orchestrator
also requires agreement on the next manipulation predicate used to verify a shared navigation step.

    python3 -m harness.grounding      # synthetic self-test with the mock backend (CPU only)
"""
from dataclasses import dataclass, field
import json
import re
from pathlib import Path

from harness import task_profile  # P2_TASK_ID selects configs/task{id:03d}_skill_program.json (13 unchanged)

ROOT = Path(__file__).resolve().parents[1]
PROGRAM = ROOT / f"configs/task{task_profile.TASK_ID:03d}_skill_program.json"


def category(obj):
    """'can_of_soda@2' -> 'can_of_soda'. An '@' tag distinguishes instances of one category inside a program (three
    identical cans); contract keys (WAM condition, budgets, memory) and the catalog stay category-level."""
    return obj.split("@", 1)[0]

SCHEMA = {
    "type": "object",
    "required": ["candidates"],
    "candidates": {"interpretation": "str", "program": [{"skill": "str", "objects": ["str"]}],
                   "preconditions": ["str"], "postconditions": ["str"]},
}


@dataclass(frozen=True)
class Step:
    skill: str
    objects: tuple

    def key(self):
        return f"{self.skill}({','.join(category(o) for o in self.objects)})"


@dataclass
class CandidateProgram:
    interpretation: str
    steps: list
    preconditions: list = field(default_factory=list)
    postconditions: list = field(default_factory=list)

    def next_key(self, index):
        return self.steps[index].key() if index < len(self.steps) else "done"


class Catalog:
    """Skills (with allowed object-argument patterns) the WAM has actually been trained on."""

    def __init__(self, path=PROGRAM):
        prog = json.loads(Path(path).read_text())
        self.classes = prog["skill_classes"]
        self.patterns = {}
        for row in prog["skills"]:
            if row["count"] >= 5:  # ignore one-off annotation noise
                self.patterns.setdefault(row["skill"], set()).add(tuple(row["objects"]))
        self.categories = sorted({o for pats in self.patterns.values() for p in pats for o in p})

    def allows(self, step: Step):
        return step.skill in self.patterns and tuple(category(o) for o in step.objects) in self.patterns[step.skill]

    def describe(self):
        return "\n".join(f"- {s}: " + ", ".join(f"{s}({','.join(p)})" for p in sorted(pats)) for s, pats in sorted(self.patterns.items()))


def assign_instance_tags(steps, instance_counts):
    """Deterministic instance binding for repeated object categories (Task 1: three identical `can_of_soda`).

    The VLM writes plain category names, and compile_programs treats every untagged `can_of_soda` as ONE object
    (after the first can is placed into the carried bin, the next `move to(can_of_soda)` is a carried-navigation
    error). `instance_counts` gives per-category instance counts from the BDDL task definition (identical at
    evaluation), e.g. {"can_of_soda": 3, "trash_can": 1}; list movable object categories only (support surfaces such
    as floors are locations). Only categories with count >= 2 are tagged, so unique objects keep their bare name as
    in configs/task001_fixed_program.json. Untagged mentions of such a category C with n instances bind in order:

      pick up from(C, s)  the k-th pick of C binds C@k (a new instance); k > n -> too_many_instances:C>n
      move to(C)          binds the instance of the NEXT pick, C@(k+1) (the approach precedes its pick); k+1 > n -> reject
      place in(C, x)      binds the most recently picked instance C@k; none picked yet or C@k already placed -> reject
      any other mention   (C as source/container, other skills) -> unresolved_instance (HOLD rather than guess)

    Explicit tags C@j are kept but must be integers 1..n and cannot be mixed with untagged mentions of C. Re-picks
    after failures never pass through here (the orchestrator's replanner reuses already-tagged steps). Returns a new
    step list or raises ValueError(reason); Step.key() stays category-level.
    """
    counts = {c: int(n) for c, n in (instance_counts or {}).items() if int(n) >= 2}
    kinds = {}
    for s in steps:
        for o in s.objects:
            if category(o) in counts:
                kinds.setdefault(category(o), set()).add("@" in o)
    for c, k in kinds.items():
        if k == {True, False}:
            raise ValueError("mixed_instance_tags:" + c)
        for o in {o for s in steps for o in s.objects if "@" in o and category(o) == c}:
            tag = o.split("@", 1)[1]
            if not tag.isdigit() or not 1 <= int(tag) <= counts[c]:
                raise ValueError("invalid_instance_tag:" + o)
    picks = {c: 0 for c, k in kinds.items() if k == {False}}
    placed, out = set(), []
    for s in steps:
        objs = list(s.objects)
        for i, c in enumerate(objs):
            if c not in picks:
                continue
            if i == 0 and s.skill in ("pick up from", "move to"):
                k = picks[c] + 1
                if k > counts[c]:
                    raise ValueError(f"too_many_instances:{c}>{counts[c]}")
                if s.skill == "pick up from":
                    picks[c] = k
                objs[i] = f"{c}@{k}"
            elif i == 0 and s.skill == "place in" and picks[c] and f"{c}@{picks[c]}" not in placed:
                objs[i] = f"{c}@{picks[c]}"
                placed.add(objs[i])
            else:
                raise ValueError("unresolved_instance:" + s.key())
        out.append(Step(s.skill, tuple(objs)))
    return out


def compilable_skill_forms(catalog: Catalog):
    """Catalog forms for which compile_programs defines a skill contract (each form probed as a one-step program).
    Unary contact skills such as press and binary support placement are included only after the compiler defines
    their spatial/carried-object semantics; unknown trained labels remain hidden rather than inviting a HOLD."""
    forms = {}
    for skill, patterns in catalog.patterns.items():
        for p in patterns:
            _, errors = compile_programs([CandidateProgram("probe", [Step(skill, tuple(p))])], catalog, list(catalog.categories))
            if not any(e.startswith(("invalid_skill_binding:", "undefined_skill_contract:")) for e in errors):
                forms.setdefault(skill, set()).add(tuple(p))
    return forms


def _task_facts_prompt(instruction, visible_objects, catalog, hints, instance_counts, held_containers):
    """build_prompt when the task definition supplies instance counts / held containers (Task 1). Relative to the
    legacy (Task 13) text it adds two facts -- repeated objects are separate instances, a held container can receive
    objects -- and fixes lines that are wrong for another catalog: the skill list keeps only compilable forms, the
    format example uses a catalog object, and the Task-13 glosses appear only when their categories exist."""
    forms = compilable_skill_forms(catalog)
    skills = "\n".join(f"- {s}: " + ", ".join(f"{s}({','.join(p)})" for p in sorted(pats)) for s, pats in sorted(forms.items()))
    example = min((p[0] for p in forms.get("move to", ())), default="car")
    cats = set(catalog.categories)
    counts = {c: int(n) for c, n in sorted(instance_counts.items())}
    for h in held_containers:
        if h not in cats:
            raise ValueError(f"held container {h!r} is not a catalog category")
    lines = ["You plan for a mobile bimanual household robot. Output ONLY JSON.",
             f'Instruction from the user: "{instruction}"',
             f"Objects currently perceived (category names): {', '.join(visible_objects)}",
             "Skills the robot can execute (skill(object arguments)); use ONLY these exact forms:",
             skills,
             'In each JSON step, "skill" contains ONLY the skill name, and "objects" contains its arguments.',
             f'Example: {{"skill": "move to", "objects": ["{example}"]}}. Never put "move to({example})" in "skill".']
    if {"digital", "tennis", "car"} <= cats:
        lines.append('Catalog abbreviations: "digital" means digital camera; "tennis" means tennis racket; the trunk lid belongs to "car".')
    if counts:
        lines.append("Object instances in the scene: " + ", ".join(f"{n} {c}" for c, n in counts.items()) + ".")
    for c, n in counts.items():
        if n > 1:
            lines.append(f'The {n} {c} are separate objects: each {c} to handle needs its own steps. Write the plain name "{c}" '
                         f"every time (the system numbers them in pick-up order). Never pick up more than {n} {c}.")
    for h in held_containers:
        lines.append(f"A container the robot holds can receive objects: the robot can pick up the {h} with one hand and keep holding "
                     f"it while the other hand picks up objects one at a time and places them into it. place in(...,{h}) is trained "
                     f"only with the {h} held, and placing into a held container needs no move to.")
    camera = (" A camera placed into a toy box is not automatically the requested camera placed into a car."
              if {"digital", "toy_box", "car"} <= cats else "")
    lines += ["Verified past experience for this scene:", hints,
              "Produce every plausible interpretation of the instruction that leads to a DIFFERENT sequence of skills.",
              "If the instruction is unambiguous, give exactly one candidate. Do not invent objects or skills.",
              'Include required navigation and manipulation in "program"; listing them as preconditions does not execute them.',
              "Never change a requested object or destination just to fit the skill catalog. If the requested effect is unsupported,",
              'return {"candidates": []}.' + camera,
              'JSON schema: {"candidates": [{"interpretation": str, "program": [{"skill": str, "objects": [str]}],',
              '"preconditions": [str], "postconditions": [str]}]}']
    return "\n".join(lines)


def build_prompt(instruction, visible_objects, catalog: Catalog, memory_hints=(), instance_counts=None, held_containers=()):
    hints = "\n".join(f"- {h}" for h in memory_hints) or "- none"
    if instance_counts or held_containers:   # Task 1 facts; without them the legacy (Task 13) text below is unchanged
        return _task_facts_prompt(instruction, visible_objects, catalog, hints, instance_counts or {}, tuple(held_containers))
    return f"""You plan for a mobile bimanual household robot. Output ONLY JSON.
Instruction from the user: "{instruction}"
Objects currently perceived (category names): {", ".join(visible_objects)}
Skills the robot can execute (skill(object arguments)); use ONLY these exact forms:
{catalog.describe()}
In each JSON step, "skill" contains ONLY the skill name, and "objects" contains its arguments.
Example: {{"skill": "move to", "objects": ["car"]}}. Never put "move to(car)" in "skill".
Catalog abbreviations: "digital" means digital camera; "tennis" means tennis racket; the trunk lid belongs to "car".
Verified past experience for this scene:
{hints}
Produce every plausible interpretation of the instruction that leads to a DIFFERENT sequence of skills.
If the instruction is unambiguous, give exactly one candidate. Do not invent objects or skills.
Include required navigation and manipulation in "program"; listing them as preconditions does not execute them.
Never change a requested object or destination just to fit the skill catalog. If the requested effect is unsupported,
return {{"candidates": []}}. A camera placed into a toy box is not automatically the requested camera placed into a car.
JSON schema: {{"candidates": [{{"interpretation": str, "program": [{{"skill": str, "objects": [str]}}],
"preconditions": [str], "postconditions": [str]}}]}}"""


def compile_programs(candidates, catalog: Catalog, visible_objects=None, *, completed_prefix=(), restart=False):
    """Validate bindings and insert catalog-supported approaches without changing any requested effect.

    This is a *plan*, not arrival/grasp evidence: text preconditions are never trusted as observations.
    Containment and carried objects are propagated only as expected effects of preceding steps. The
    runtime must still verify each step. Replans preserve the executed prefix and re-establish proximity.
    Returns (valid candidates, rejections); execution callers must HOLD if any candidate was rejected.
    """
    if not isinstance(candidates, (list, tuple)):
        return [], ["malformed_candidates"]
    visible = list(catalog.categories if visible_objects is None else visible_objects)
    out, rejected, seen = [], [], set()
    prefix = list(completed_prefix)
    for c in candidates:
        if (not isinstance(c, CandidateProgram) or not isinstance(c.interpretation, str) or not c.interpretation.strip()
                or not isinstance(c.steps, list) or not c.steps
                or any(not isinstance(s, Step) or not isinstance(s.skill, str) or not isinstance(s.objects, tuple)
                       or not all(isinstance(o, str) for o in s.objects) for s in c.steps)
                or any(not isinstance(v, list) or not all(isinstance(s, str) for s in v)
                       for v in (c.preconditions, c.postconditions))):
            rejected.append("malformed_candidate")
            continue
        if restart and len(c.steps) <= len(prefix):
            rejected.append("empty_replan_suffix")
            continue
        unseen = sorted({category(o) for s in c.steps for o in s.objects} - set(visible))
        bad = [s.key() for s in c.steps if not catalog.allows(s)]
        ambiguous = sorted({category(o) for s in c.steps for o in s.objects
                            if visible.count(category(o)) != 1 and category(o) in visible})
        if unseen or bad or ambiguous or c.steps[:len(prefix)] != prefix:
            rejected.append("unperceived_object:" + ";".join(unseen) if unseen else
                            "untrained_skill:" + ";".join(bad) if bad else
                            "ambiguous_object_binding:" + ";".join(ambiguous) if ambiguous else "changed_completed_prefix")
            continue
        steps, at, inside, carried = [], None, {}, set()

        def location(obj):
            visited = set()
            while obj in inside:
                if obj in visited:
                    raise ValueError("cyclic_object_binding")
                visited.add(obj)
                obj = inside[obj]
            return obj

        try:
            for index, step in enumerate(c.steps):
                if restart and index == len(prefix):
                    at = None   # a failed attempt can move the robot away from its previous approach
                if len(step.objects) != (2 if step.skill in ("pick up from", "place in", "place on", "place on next to", "push to") else 1):
                    raise ValueError("invalid_skill_binding:" + step.key())
                if step.skill in ("move to", "open lid", "close lid", "open door", "pick up from", "press", "push to"):
                    target = step.objects[0]
                elif step.skill in ("place in", "place on", "place on next to"):
                    target = step.objects[1]
                else:
                    raise ValueError("undefined_skill_contract:" + step.key())
                destination = location(target)
                # A container held by the robot (Task 1: the trash can in one hand) travels with it: placing into
                # it needs no approach. Navigating to a carried object is still an invalid program.
                held_container = step.skill == "place in" and destination in carried
                operates_on_held = step.skill == "press" and target in carried
                if destination in carried and not held_container and not operates_on_held:
                    raise ValueError("carried_navigation_target:" + target)
                if step.skill != "move to" and at != destination and not held_container and not operates_on_held:
                    nav = Step("move to", (target,))
                    if index < len(prefix):
                        raise ValueError("incomplete_completed_prefix")
                    if not catalog.allows(nav):
                        raise ValueError("untrained_navigation:" + nav.key())
                    steps.append(nav)
                    at = destination
                steps.append(step)
                if step.skill == "move to":
                    at = destination
                elif step.skill == "pick up from":
                    obj, source = step.objects
                    if obj in inside and location(inside[obj]) != location(source):
                        raise ValueError("conflicting_pickup_source:" + step.key())
                    at = location(source)
                    inside.pop(obj, None)
                    carried.add(obj)
                elif step.skill == "place on next to":
                    # Task90: release the bin on support NEXT TO the sink, not inside it.
                    # Do not add a containment edge or treat later approach-to-bin as approach-to-sink.
                    obj, reference = step.objects
                    if obj not in carried or obj == reference:
                        raise ValueError("invalid_adjacent_placement:" + step.key())
                    carried.remove(obj)
                    inside.pop(obj, None)
                    at = None
                elif step.skill in ("place in", "place on"):
                    obj, container = step.objects
                    inside[obj] = container
                    location(obj)  # reject impossible containment cycles
                    carried.discard(obj)
                elif step.skill == "push to":
                    inside[step.objects[0]] = step.objects[1]
                    location(step.objects[0])
                    at = None   # pushing need not leave the robot at the object's destination
        except ValueError as error:
            rejected.append(str(error))
            continue
        signature = tuple(s.key() for s in steps)
        if signature not in seen:
            seen.add(signature)
            out.append(CandidateProgram(c.interpretation, steps, list(c.preconditions), list(c.postconditions)))
    return out, rejected


def parse(text, catalog: Catalog, visible_objects, instance_counts=None):
    """Strict parse and spatial compilation; invalid candidates are dropped and reported.
    With `instance_counts` (Task 1), repeated categories are bound to instances first (assign_instance_tags)."""
    m = re.search(r"\{.*\}", text, re.S)
    if not m:
        return [], ["no_json"]
    try:
        data = json.loads(m.group(0))
    except json.JSONDecodeError as e:
        return [], [f"bad_json:{e.msg}"]
    if not isinstance(data, dict) or not isinstance(data.get("candidates"), list):
        return [], ["malformed_candidates"]
    out, rejected = [], []
    for c in data.get("candidates", []):
        try:
            if not isinstance(c, dict) or not isinstance(c.get("program"), list) or not isinstance(c.get("interpretation"), str):
                raise TypeError
            for s in c["program"]:
                if not isinstance(s, dict) or not isinstance(s.get("skill"), str) or not isinstance(s.get("objects"), list):
                    raise TypeError
                if not all(isinstance(o, str) for o in s["objects"]):
                    raise TypeError
            for key in ("preconditions", "postconditions"):
                if not isinstance(c.get(key, []), list) or not all(isinstance(v, str) for v in c.get(key, [])):
                    raise TypeError
            steps = [Step(s["skill"], tuple(s["objects"])) for s in c["program"]]
            interp = c["interpretation"].strip()
        except (KeyError, TypeError, AttributeError):
            rejected.append("malformed_candidate")
            continue
        if not steps or not interp:
            rejected.append("empty_candidate")
        else:
            out.append(CandidateProgram(interp, steps, list(c.get("preconditions", [])), list(c.get("postconditions", []))))
    if instance_counts:
        tagged = []
        for c in out:
            try:
                tagged.append(CandidateProgram(c.interpretation, assign_instance_tags(c.steps, instance_counts),
                                               c.preconditions, c.postconditions))
            except ValueError as error:
                rejected.append(str(error))
        out = tagged
    compiled, errors = compile_programs(out, catalog, visible_objects)
    return compiled, rejected + errors


def divergence(candidates, index):
    """Where do the candidates first disagree at or after step `index`? Returns (common_prefix_len, choices)."""
    if not candidates:
        return 0, ()
    i = index
    while True:
        keys = {c.next_key(i) for c in candidates}
        if len(keys) > 1 or "done" in keys:
            return i - index, tuple(c.interpretation for c in candidates) if len(keys) > 1 else ()
        i += 1


class MockBackend:
    def __init__(self, responses):
        self.responses = responses

    def complete(self, prompt):
        for needle, text in self.responses.items():
            if needle in prompt:
                return text
        return "{}"


class QwenVLBackend:
    """Qwen2.5-VL via transformers; loads lazily so importing this module never touches the GPU.
    quant='4bit' needs bitsandbytes; device_map='cpu' works without a GPU but is slow."""

    def __init__(self, model_id="Qwen/Qwen2.5-VL-7B-Instruct", quant="4bit", device_map="auto", max_new_tokens=600):
        self.model_id, self.quant, self.device_map, self.max_new_tokens = model_id, quant, device_map, max_new_tokens
        self._model = self._proc = None

    def _load(self):
        import torch
        from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
        kw = {"device_map": self.device_map}
        if self.quant == "4bit":
            from transformers import BitsAndBytesConfig
            kw["quantization_config"] = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.bfloat16,
                                                            bnb_4bit_quant_type="nf4")
        else:
            kw["dtype"] = torch.bfloat16
        self._model = Qwen2_5_VLForConditionalGeneration.from_pretrained(self.model_id, **kw)
        self._proc = AutoProcessor.from_pretrained(self.model_id)

    def complete(self, prompt, image=None):
        if self._model is None:
            self._load()
        content = ([{"type": "image", "image": image}] if image is not None else []) + [{"type": "text", "text": prompt}]
        msgs = [{"role": "user", "content": content}]
        text = self._proc.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        inputs = self._proc(text=[text], images=[image] if image is not None else None, return_tensors="pt").to(self._model.device)
        out = self._model.generate(**inputs, max_new_tokens=self.max_new_tokens, do_sample=False)
        self.last_text = self._proc.batch_decode(out[:, inputs["input_ids"].shape[1]:], skip_special_tokens=True)[0]
        return self.last_text  # kept on the instance so probes can log the raw (pre-parse) model output


def ground(backend, instruction, visible_objects, catalog=None, memory_hints=(), image=None, instance_counts=None,
           held_containers=()):
    catalog = catalog or Catalog()
    prompt = build_prompt(instruction, visible_objects, catalog, memory_hints, instance_counts, held_containers)
    text = backend.complete(prompt, image) if image is not None else backend.complete(prompt)
    candidates, rejected = parse(text, catalog, visible_objects, instance_counts)
    # Dropping an unsupported interpretation is not evidence that the remaining interpretation is intended.
    return ([] if rejected else candidates), rejected, prompt


def _self_test():
    cat = Catalog()
    visible = ["car", "door", "toy_box", "floors", "digital", "tennis", "coffee_table"]
    cam = [{"skill": "move to", "objects": ["digital"]}, {"skill": "pick up from", "objects": ["digital", "coffee_table"]}]
    rac = [{"skill": "move to", "objects": ["tennis"]}, {"skill": "pick up from", "objects": ["tennis", "coffee_table"]}]
    tail = [{"skill": "move to", "objects": ["car"]}, {"skill": "place in", "objects": ["tennis", "car"]},
            {"skill": "place in", "objects": ["digital", "toy_box"]}]
    common = [{"skill": "move to", "objects": ["car"]}, {"skill": "open lid", "objects": ["car"]}]
    ambiguous = json.dumps({"candidates": [
        {"interpretation": "camera first", "program": common + cam + rac + tail},
        {"interpretation": "racket first", "program": common + rac + cam + tail},
        {"interpretation": "duplicate of camera first", "program": common + cam + rac + tail},
        {"interpretation": "literal: camera straight into the car (untrained)", "program": common + cam + [{"skill": "move to", "objects": ["car"]}, {"skill": "place in", "objects": ["digital", "car"]}]},
        {"interpretation": "hallucinated skill", "program": [{"skill": "fly to", "objects": ["car"]}]},
        {"interpretation": "unseen object", "program": common + [{"skill": "move to", "objects": ["sofa"]}]}]})
    backend = MockBackend({"pack the small items": ambiguous,
                           "close the trunk": json.dumps({"candidates": [{"interpretation": "close the car lid", "program": common[:1] + [{"skill": "close lid", "objects": ["car"]}]}]})})
    cands, rejected = parse(ambiguous, cat, visible)
    assert [c.interpretation for c in cands] == ["camera first", "racket first"], cands
    assert rejected == ["untrained_skill:place in(digital,car)", "untrained_skill:fly to(car)", "unperceived_object:sofa"], rejected
    prefix, choices = divergence(cands, 0)
    assert prefix == 2 and len(choices) == 2, (prefix, choices)  # shared: move to car, open lid -> execute first; ask at step 2
    assert divergence(cands, 2) == (0, choices)
    blocked, rejections, _ = ground(backend, "pack the small items into the car", visible, cat)
    assert blocked == [] and rejections == rejected
    one, rej, _ = ground(backend, "close the trunk", visible, cat)
    assert len(one) == 1 and rej == [] and divergence(one, 0) == (2, ())
    assert parse("no json here", cat, visible) == ([], ["no_json"])
    for malformed in ('{}', '{"candidates": null}', '{"candidates": [null]}',
                      '{"candidates": [{"interpretation": 7, "program": []}]}'):
        assert parse(malformed, cat, visible)[1]
    print("PASS (synthetic, CPU only): strict parsing, catalog/perception rejection, dedupe, action-level divergence")


def _self_test_instance_tags():
    """Task 1 instance binding on the Task 1 catalog (explicit path, independent of P2_TASK_ID)."""
    cat = Catalog(ROOT / "configs/task001_skill_program.json")
    visible, counts = ["trash_can", "can_of_soda", "floors"], {"can_of_soda": 3, "trash_can": 1}
    ref = [(s["skill"], tuple(s["objects"])) for s in json.loads((ROOT / "configs/task001_fixed_program.json").read_text())["program"]]

    def step(skill, *objects):
        return {"skill": skill, "objects": list(objects)}

    def text(program):
        return json.dumps({"candidates": [{"interpretation": "cans into the held bin", "program": program}]})

    bin_first = [step("move to", "trash_can"), step("pick up from", "trash_can", "floors")]
    cycle = [step("move to", "can_of_soda"), step("pick up from", "can_of_soda", "floors"), step("place in", "can_of_soda", "trash_can")]
    for program in (bin_first + cycle * 3, [s for s in bin_first + cycle * 3 if s["skill"] != "move to"]):
        cands, rej = parse(text(program), cat, visible, counts)
        assert rej == [] and len(cands) == 1 and [(s.skill, s.objects) for s in cands[0].steps] == ref, (rej, cands)
        assert [s.key() for s in cands[0].steps] == [f"{s}({','.join(category(o) for o in objs)})" for s, objs in ref]
    assert parse(text(bin_first + cycle * 3), cat, visible)[1] == ["carried_navigation_target:can_of_soda"]  # why tags exist
    assert parse(text(bin_first + cycle * 4), cat, visible, counts)[1] == ["too_many_instances:can_of_soda>3"]
    picks_then_places = bin_first + [cycle[1]] * 3 + [cycle[2]] * 3
    assert parse(text(picks_then_places), cat, visible, counts)[1] == ["unresolved_instance:place in(can_of_soda,trash_can)"]
    assert parse(text(bin_first + cycle[2:] + cycle), cat, visible, counts)[1] == ["unresolved_instance:place in(can_of_soda,trash_can)"]
    tagged = [dict(s, objects=[o + "@2" if o == "can_of_soda" else o for o in s["objects"]]) for s in bin_first + cycle]
    assert parse(text(tagged), cat, visible, counts)[1] == []
    assert parse(text(tagged + cycle), cat, visible, counts)[1] == ["mixed_instance_tags:can_of_soda"]
    assert parse(text([dict(s, objects=[o.replace("@2", "@4") for o in s["objects"]]) for s in tagged]), cat, visible,
                 counts)[1] == ["invalid_instance_tag:can_of_soda@4"]
    floor_bin, rej = parse(text(cycle[1:] * 3), cat, visible, counts)  # carries each can to a bin on the floor: compiles,
    assert rej == [] and [(s.skill, s.objects) for s in floor_bin[0].steps] != ref  # but the trained place-in needs the held bin
    prompt = build_prompt("Throw the soda cans into the trash can.", visible, cat, instance_counts=counts, held_containers=("trash_can",))
    assert "3 can_of_soda" in prompt and "place on" in prompt and '"car"' not in prompt and "held" in prompt
    assert build_prompt("x", visible, cat) == build_prompt("x", visible, cat, instance_counts={}, held_containers=())
    backend = MockBackend({"soda cans": text(bin_first + cycle * 3)})
    cands, rej, _ = ground(backend, "Throw the soda cans into the trash can.", visible, cat, instance_counts=counts,
                           held_containers=("trash_can",))
    assert rej == [] and [(s.skill, s.objects) for s in cands[0].steps] == ref
    print("PASS (synthetic, CPU only): Task 1 instance tags (k-th pick -> @k, approach/place propagation, count limit), "
          "held-container prompt")


if __name__ == "__main__":
    _self_test()
    _self_test_instance_tags()
