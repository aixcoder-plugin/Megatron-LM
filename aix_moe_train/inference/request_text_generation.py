#!/usr/bin/env python3
"""Send a text-generation request to the Megatron HTTP inference server."""

import argparse
import json
import sys
import urllib.error
import urllib.request


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Send a prompt to the Megatron text-generation server."
    )
    parser.add_argument("--host", default="127.0.0.1", help="Server host.")
    parser.add_argument("--port", type=int, default=5000, help="Server port.")
    parser.add_argument("--prompt", required=True, help="Input prompt text.")
    parser.add_argument(
        "--tokens-to-generate",
        type=int,
        default=128,
        help="Number of output tokens to generate.",
    )
    parser.add_argument(
        "--temperature", type=float, default=1.0, help="Sampling temperature."
    )
    parser.add_argument("--top-k", type=int, default=0, help="Top-k sampling value.")
    parser.add_argument("--top-p", type=float, default=0.0, help="Top-p sampling value.")
    parser.add_argument(
        "--random-seed",
        type=int,
        default=-1,
        help="Sampling seed. Use -1 to let the server decide.",
    )
    parser.add_argument(
        "--add-bos",
        action="store_true",
        help="Ask the server to prepend BOS for empty prompts.",
    )
    parser.add_argument(
        "--logprobs",
        action="store_true",
        help="Request token log probabilities from the server.",
    )
    parser.add_argument(
        "--raw",
        action="store_true",
        help="Print the full JSON response instead of only the generated text.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    url = f"http://{args.host}:{args.port}/api"
    payload = {
        "prompts": [args.prompt],
        "tokens_to_generate": args.tokens_to_generate,
        "temperature": args.temperature,
        "top_k": args.top_k,
        "top_p": args.top_p,
        "add_BOS": args.add_bos,
        "logprobs": args.logprobs,
    }
    if args.random_seed >= 0:
        payload["random_seed"] = args.random_seed

    request = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="PUT",
    )

    try:
        with urllib.request.urlopen(request) as response:
            body = response.read().decode("utf-8")
    except urllib.error.HTTPError as exc:
        error_body = exc.read().decode("utf-8", errors="replace")
        sys.stderr.write(f"HTTP {exc.code}: {error_body}\n")
        return 1
    except urllib.error.URLError as exc:
        sys.stderr.write(f"Request failed: {exc.reason}\n")
        return 1

    data = json.loads(body)
    if args.raw:
        print(json.dumps(data, ensure_ascii=False, indent=2))
        return 0

    texts = data.get("text")
    if isinstance(texts, list) and texts:
        print(texts[0])
        return 0

    print(json.dumps(data, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
