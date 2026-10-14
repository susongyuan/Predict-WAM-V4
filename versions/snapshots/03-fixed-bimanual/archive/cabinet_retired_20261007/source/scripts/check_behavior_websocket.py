#!/usr/bin/env python3
"""CPU loopback of the actual port's health, handshake, reset and 23D action protocol.

Run with .conda-envs/behavior6/bin/python scripts/check_behavior_websocket.py.
Only the simulator-triggering macros import is excluded; networking code is unchanged.
"""
import ast
import asyncio
import json
import os
from pathlib import Path
from types import SimpleNamespace

os.environ["CUDA_VISIBLE_DEVICES"] = ""
ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "third_party/BEHAVIOR-1K-isaac6/OmniGibson/omnigibson/eval/utils/network_utils.py"
tree = ast.parse(SOURCE.read_text(), filename=str(SOURCE))
tree.body = [node for node in tree.body
             if not (isinstance(node, ast.ImportFrom) and node.module == "omnigibson.macros")]
network = {"__name__": "isolated_behavior_network", "gm": SimpleNamespace(DEBUG=False)}
exec(compile(tree, str(SOURCE), "exec"), network)
np, torch = network["np"], network["th"]
torch.set_num_threads(1)


class Policy:
    resets = 0

    def reset(self):
        self.resets += 1

    def act(self, obs):
        assert obs["state"].shape == (61,)
        assert obs["rgb"].shape == (224, 224, 3) and obs["rgb"].dtype == np.uint8
        return torch.from_numpy(obs["state"][:23].copy()) + self.resets


async def main():
    policy = Policy()
    server = network["WebsocketPolicyServer"](policy, metadata={"action_dim": 23})
    async with network["_server"].serve(server._handler, "127.0.0.1", 0, compression=None,
                                        max_size=None, process_request=network["_health_check"]) as socket_server:
        port = socket_server.sockets[0].getsockname()[1]

        def client_check():
            client = network["WebsocketClientPolicy"](host="127.0.0.1", port=port)
            try:
                for reset_count in (1, 2):
                    client.reset()
                    assert client.get_server_metadata() == {"action_dim": 23}
                    action = client.act({"state": np.arange(61, dtype=np.float32),
                                         "rgb": np.zeros((224, 224, 3), dtype=np.uint8)})
                    assert action.shape == (23,) and action.dtype == torch.float32
                    assert torch.equal(action, torch.arange(23, dtype=torch.float32) + reset_count)
            finally:
                if client._ws is not None:
                    client._ws.close()

        await asyncio.wait_for(asyncio.to_thread(client_check), timeout=15)
    assert policy.resets == 2 and not torch.cuda.is_initialized()
    print(json.dumps({"status": "passed", "websockets": network["websockets"].__version__,
                      "health_handshake": True, "resets": policy.resets, "actions": 2,
                      "state_dim": 61, "action_dim": 23, "cpu_only": True,
                      "source": str(SOURCE.relative_to(ROOT))}, indent=2))


if __name__ == "__main__":
    asyncio.run(main())
