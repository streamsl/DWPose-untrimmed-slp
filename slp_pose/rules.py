"""Primary-signer rules by name: the framework's own, and a dataset's.

A rule is a select.FrameRule (per frame: the GPU workers pose with one, and `derive` can re-pick
with one) or a signer.VideoRule (the whole video: `derive` applies it on the CPU). Records, hashes,
logs and the CLI name rules. The framework's rules are generic: `largest_bbox`, `highest_score`
(select.PRIMARY_RULES) and `signer_track` (signer.VIDEO_RULES). A rule that serves one dataset only
is defined in that dataset's file and listed in its Dataset.rules, and Dataset.rule(name) resolves
the dataset's names and the framework's; a dataset never reuses a framework rule's name.

Functions taking a rule accept a rule object (e.g. from Dataset.rule) or a framework rule's name (a
dataset's own rule is passed as the object). Pure numpy; must not import torch.
"""
from __future__ import annotations

from typing import List, Union

from .select import PRIMARY_RULES, FrameRule
from .signer import VIDEO_RULES, VideoRule

Rule = Union[FrameRule, VideoRule]
RuleLike = Union[str, FrameRule, VideoRule]


def framework_names() -> List[str]:
    """The framework's rule names: per-frame, then video-level."""
    return sorted(PRIMARY_RULES) + sorted(VIDEO_RULES)


def is_framework_name(name: str) -> bool:
    return name in PRIMARY_RULES or name in VIDEO_RULES


def framework_rule(name: str) -> Rule:
    """The framework's rule `name`; KeyError for another name."""
    rule = PRIMARY_RULES.get(name) or VIDEO_RULES.get(name)
    if rule is None:
        raise KeyError(f"unknown rule {name!r}; the framework's rules: {framework_names()} (a dataset's own rules "
                       f'resolve through its Dataset.rule)')
    return rule


def as_rule(rule: RuleLike) -> Rule:
    """`rule` itself (a FrameRule or VideoRule), or the framework's rule of that name (KeyError for
    another name). A rule object may not carry a framework rule's name unless it is that rule
    (ValueError)."""
    if isinstance(rule, str):
        return framework_rule(rule)
    if not isinstance(rule, (FrameRule, VideoRule)):
        raise TypeError(f'expected a rule name, FrameRule or VideoRule, got {rule!r}')
    if is_framework_name(rule.name) and rule != framework_rule(rule.name):
        raise ValueError(f'{rule.name!r} is the name of a framework rule; give this rule another name')
    return rule


def as_frame_rule(rule: RuleLike) -> FrameRule:
    """as_rule, which must be per-frame (KeyError for a video-level rule)."""
    out = as_rule(rule)
    if not isinstance(out, FrameRule):
        raise KeyError(f'{out.name!r} is a video-level rule, not a per-frame rule')
    return out
