#!/usr/bin/env python3
"""Frozen 30k WAM/braking with an explicit causal-history restoration IPC."""
from pathlib import Path
import sys
import time
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from scripts import eval_zhuhai_precision_official_braking as upstream
from harness.zhuhai_dagger_segmented import POLICY_PROTOCOL, restore_policy


def serve(channel, policy):
    b = upstream.base
    policy.runtime.update(segmented_policy_support=POLICY_PROTOCOL,
                          segmented_worker_source_sha256=b.identity(__file__)['sha256'])
    channel.send(b.envelope('ready', 'startup', 0, runtime=policy.runtime, pid=__import__('os').getpid()), 10)
    while True:
        message = channel.recv(600)
        episode, step = message.get('episode'), message.get('step')
        b.ids(episode, step)
        if message.get('version') != b.PROTOCOL:
            raise b.InfrastructureError('Unsupported native16 protocol')
        kind = message['kind']
        started = time.monotonic()
        if kind == 'reset':
            b.strict_keys(message, ('version', 'kind', 'episode', 'step', 'seed'))
            if step != 0:
                raise ValueError('Reset must start at zero')
            policy.reset(episode, message['seed'])
            reply = b.envelope('reset_ok', episode, step)
        elif kind == 'restore_history':
            b.strict_keys(message, ('version', 'kind', 'episode', 'step', 'states', 'commands', 'visuals', 'control'))
            frames = {int(k): b.decode_observation(v) for k, v in message['visuals'].items()}
            result = restore_policy(policy, episode, step, message['states'], message['commands'], frames, message['control'])
            reply = b.envelope('history_restored', episode, step, **{k: v for k, v in result.items() if k != 'next_step'}, next_step=step)
        elif kind == 'clear_actions':
            b.strict_keys(message, ('version', 'kind', 'episode', 'step', 'reason'))
            reply = b.envelope('cleared', episode, step, **policy.clear_actions(episode, step, message['reason']))
        elif kind == 'observe':
            b.strict_keys(message, ('version', 'kind', 'episode', 'step', 'observation'))
            result = policy.observe(episode, step, b.decode_observation(message['observation']))
            for key in ('raw_action', 'model_raw_action', 'smoothed_action', 'limited_action'):
                b.vector(result[key], 16, key)
            reply = b.envelope('action', episode, step, **result, policy_seconds=time.monotonic()-started)
        else:
            raise ValueError('Unknown segmented worker request')
        channel.send(reply, 60)


def main():
    upstream.configure()
    upstream.base.serve_native16 = serve
    return upstream.base.main()


if __name__ == '__main__':
    raise SystemExit(main())
