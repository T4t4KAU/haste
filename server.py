"""HTTP server for exposing Haste as a text generation service."""

from __future__ import annotations

import argparse
import json
import signal
import threading
import time
import uuid
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

from haste import LLM, SamplingParams
from haste.utils.profiling import build_profile_report


class APIError(Exception):
    """HTTP-facing API error."""

    def __init__(self, status: HTTPStatus, message: str):
        super().__init__(message)
        self.status = status
        self.message = message


def build_parser() -> argparse.ArgumentParser:
    """Build command line argument parser."""
    parser = argparse.ArgumentParser(description="Serve Haste as an HTTP API.")
    parser.add_argument(
        "--mode",
        choices=("ar", "spec_sync", "spec_async"),
        default="spec_async",
        help="Decoding mode: autoregressive (`ar`), synchronous speculative (`spec_sync`), or asynchronous speculative (`spec_async`).",
    )
    parser.add_argument("--target-model-path", required=True, help="Path to the target model.")
    parser.add_argument(
        "--draft-model-path",
        default="",
        help="Path to the draft model. Required for speculative modes.",
    )
    parser.add_argument("--host", default="0.0.0.0", help="Host to bind the HTTP server.")
    parser.add_argument("--port", type=int, default=8000, help="Port to bind the HTTP server.")
    parser.add_argument("--max-num-seqs", type=int, default=32, help="Maximum number of concurrent sequences.")
    parser.add_argument(
        "--max-num-batched-tokens",
        type=int,
        default=4096,
        help="Maximum number of batched tokens for the scheduler.",
    )
    parser.add_argument("--max-model-len", type=int, default=4096, help="Maximum model context length.")
    parser.add_argument(
        "--default-max-new-tokens",
        type=int,
        default=128,
        help="Default maximum number of newly generated tokens when the request does not override it.",
    )
    parser.add_argument("--speculate-k", type=int, default=7, help="Speculative lookahead budget.")
    parser.add_argument("--async-fan-out", type=int, default=3, help="Async speculative fan-out budget.")
    parser.add_argument(
        "--auto-tune-kf",
        action="store_true",
        help="Dynamically search and adjust speculative lookahead/fan-out at runtime.",
    )
    parser.add_argument(
        "--enforce-eager",
        action="store_true",
        help="Disable CUDA graph capture and force eager execution.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose runtime logs.",
    )
    return parser


def validate_mode_args(args: argparse.Namespace) -> None:
    """Validate command line argument combinations."""
    if args.mode in {"spec_sync", "spec_async"} and not args.draft_model_path:
        raise ValueError("--draft-model-path is required when --mode is spec_sync or spec_async.")
    if args.auto_tune_kf and args.mode != "spec_async":
        raise ValueError("--auto-tune-kf is only supported when --mode is spec_async.")


def build_llm_kwargs(args: argparse.Namespace) -> dict[str, Any]:
    """Build LLM keyword arguments from CLI arguments."""
    speculate = args.mode != "ar"
    draft_async = args.mode == "spec_async"
    kwargs: dict[str, Any] = {
        "model": args.target_model_path,
        "speculate": speculate,
        "speculate_k": args.speculate_k,
        "draft_async": draft_async,
        "async_fan_out": args.async_fan_out,
        "async_auto_tune": args.auto_tune_kf,
        "verbose": args.verbose,
        "enforce_eager": args.enforce_eager,
        "max_num_seqs": args.max_num_seqs,
        "max_num_batched_tokens": args.max_num_batched_tokens,
        "max_model_len": args.max_model_len,
    }
    if speculate:
        kwargs["draft_model"] = args.draft_model_path
    return kwargs


def render_chat_prompt(messages: list[dict[str, Any]], tokenizer) -> str:
    """Render chat messages to a prompt string."""
    if not isinstance(messages, list) or not messages:
        raise APIError(HTTPStatus.BAD_REQUEST, "`messages` must be a non-empty list.")

    normalized_messages: list[dict[str, str]] = []
    for idx, message in enumerate(messages):
        if not isinstance(message, dict):
            raise APIError(HTTPStatus.BAD_REQUEST, f"`messages[{idx}]` must be an object.")
        role = message.get("role")
        content = message.get("content")
        if not isinstance(role, str) or not role:
            raise APIError(HTTPStatus.BAD_REQUEST, f"`messages[{idx}].role` must be a non-empty string.")
        if not isinstance(content, str):
            raise APIError(HTTPStatus.BAD_REQUEST, f"`messages[{idx}].content` must be a string.")
        normalized_messages.append({"role": role, "content": content})

    if hasattr(tokenizer, "apply_chat_template"):
        return tokenizer.apply_chat_template(
            normalized_messages,
            tokenize=False,
            add_generation_prompt=True,
        )

    prompt_lines = []
    for message in normalized_messages:
        prompt_lines.append(f"{message['role'].upper()}: {message['content']}")
    prompt_lines.append("ASSISTANT:")
    return "\n".join(prompt_lines)


def parse_prompt_inputs(body: dict[str, Any]) -> tuple[list[str | list[int]], bool]:
    """Parse prompt inputs from a request body."""
    if "prompt" in body:
        prompt = body["prompt"]
        if not isinstance(prompt, str):
            raise APIError(HTTPStatus.BAD_REQUEST, "`prompt` must be a string.")
        return [prompt], True

    if "prompts" in body:
        prompts = body["prompts"]
        if not isinstance(prompts, list) or not prompts or not all(isinstance(prompt, str) for prompt in prompts):
            raise APIError(HTTPStatus.BAD_REQUEST, "`prompts` must be a non-empty list of strings.")
        return prompts, False

    if "prompt_token_ids" in body:
        prompt_token_ids = body["prompt_token_ids"]
        if (
            not isinstance(prompt_token_ids, list)
            or not prompt_token_ids
            or not all(isinstance(token_id, int) for token_id in prompt_token_ids)
        ):
            raise APIError(HTTPStatus.BAD_REQUEST, "`prompt_token_ids` must be a non-empty list of integers.")
        return [prompt_token_ids], True

    if "prompt_token_ids_batch" in body:
        prompt_token_ids_batch = body["prompt_token_ids_batch"]
        if (
            not isinstance(prompt_token_ids_batch, list)
            or not prompt_token_ids_batch
            or not all(isinstance(row, list) and all(isinstance(token_id, int) for token_id in row) for row in prompt_token_ids_batch)
        ):
            raise APIError(
                HTTPStatus.BAD_REQUEST,
                "`prompt_token_ids_batch` must be a non-empty list of integer lists.",
            )
        return prompt_token_ids_batch, False

    raise APIError(
        HTTPStatus.BAD_REQUEST,
        "Request must include one of `prompt`, `prompts`, `prompt_token_ids`, or `prompt_token_ids_batch`.",
    )


def _sampling_params_from_payload(payload: dict[str, Any], *, default_max_new_tokens: int) -> SamplingParams:
    """Build SamplingParams from a request payload."""
    if not isinstance(payload, dict):
        raise APIError(HTTPStatus.BAD_REQUEST, "`sampling_params` entries must be objects.")

    max_new_tokens = payload.get("max_new_tokens", payload.get("max_tokens", default_max_new_tokens))
    if not isinstance(max_new_tokens, int) or max_new_tokens <= 0:
        raise APIError(HTTPStatus.BAD_REQUEST, "`max_new_tokens`/`max_tokens` must be a positive integer.")

    temperature = payload.get("temperature", 0.0)
    draft_temperature = payload.get("draft_temperature", 0.0)
    ignore_eos = payload.get("ignore_eos", False)

    if not isinstance(temperature, (int, float)):
        raise APIError(HTTPStatus.BAD_REQUEST, "`temperature` must be a number.")
    if draft_temperature is not None and not isinstance(draft_temperature, (int, float)):
        raise APIError(HTTPStatus.BAD_REQUEST, "`draft_temperature` must be a number or null.")
    if not isinstance(ignore_eos, bool):
        raise APIError(HTTPStatus.BAD_REQUEST, "`ignore_eos` must be a boolean.")

    return SamplingParams(
        temperature=float(temperature),
        draft_temperature=None if draft_temperature is None else float(draft_temperature),
        max_new_tokens=max_new_tokens,
        ignore_eos=ignore_eos,
    )


def build_sampling_params_list(
    body: dict[str, Any],
    *,
    count: int,
    default_max_new_tokens: int,
) -> list[SamplingParams]:
    """Build per-request sampling params."""
    if "sampling_params" not in body:
        params = _sampling_params_from_payload(body, default_max_new_tokens=default_max_new_tokens)
        return [params] * count

    payload = body["sampling_params"]
    if isinstance(payload, dict):
        params = _sampling_params_from_payload(payload, default_max_new_tokens=default_max_new_tokens)
        return [params] * count

    if not isinstance(payload, list) or len(payload) != count:
        raise APIError(
            HTTPStatus.BAD_REQUEST,
            "`sampling_params` must be an object or a list whose length matches the number of prompts.",
        )
    return [
        _sampling_params_from_payload(item, default_max_new_tokens=default_max_new_tokens)
        for item in payload
    ]


def infer_finish_reason(output: dict[str, Any], sampling_params: SamplingParams) -> str:
    """Infer finish reason from output length."""
    return "length" if len(output["token_ids"]) >= sampling_params.max_new_tokens else "stop"


class HasteService:
    """Thread-safe Haste inference service."""

    def __init__(self, llm: LLM, args: argparse.Namespace):
        self.llm = llm
        self.args = args
        self.lock = threading.Lock()
        self.started_at = int(time.time())
        self.mode = args.mode
        self.model_path = str(Path(args.target_model_path).expanduser().resolve())
        self.draft_model_path = (
            str(Path(args.draft_model_path).expanduser().resolve()) if args.draft_model_path else None
        )
        self.model_id = Path(self.model_path).name or self.model_path
        self.default_max_new_tokens = args.default_max_new_tokens

    @property
    def tokenizer(self):
        return self.llm.tokenizer

    def close(self) -> None:
        """Release model resources."""
        self.llm.shutdown()

    def model_metadata(self) -> dict[str, Any]:
        """Return metadata for the loaded model."""
        return {
            "id": self.model_id,
            "object": "model",
            "created": self.started_at,
            "owned_by": "haste",
            "mode": self.mode,
            "target_model_path": self.model_path,
            "draft_model_path": self.draft_model_path,
            "speculate": self.args.mode != "ar",
            "draft_async": self.args.mode == "spec_async",
            "speculate_k": self.args.speculate_k,
            "async_fan_out": self.args.async_fan_out,
            "auto_tune_kf": self.args.auto_tune_kf,
        }

    def _count_prompt_tokens(self, prompt: str | list[int]) -> int:
        if isinstance(prompt, list):
            return len(prompt)
        return len(self.tokenizer.encode(prompt))

    def _usage_summary(self, prompts: list[str | list[int]], outputs: list[dict[str, Any]]) -> dict[str, int]:
        prompt_tokens = sum(self._count_prompt_tokens(prompt) for prompt in prompts)
        completion_tokens = sum(len(output["token_ids"]) for output in outputs)
        return {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": prompt_tokens + completion_tokens,
        }

    def generate(
        self,
        prompts: list[str | list[int]],
        sampling_params: list[SamplingParams],
        *,
        return_metrics: bool,
    ) -> tuple[list[dict[str, Any]], dict[str, Any] | None]:
        """Run generation under a lock and optionally build a profile report."""
        start = time.perf_counter()
        with self.lock:
            outputs, metrics = self.llm.generate(
                prompts,
                sampling_params,
                use_tqdm=False,
                log_metrics=False,
            )
        elapsed = time.perf_counter() - start

        if not return_metrics:
            return outputs, None

        generated_new_tokens = sum(len(output["token_ids"]) for output in outputs)
        requested_new_tokens = sum(params.max_new_tokens for params in sampling_params)
        profile = build_profile_report(
            metrics,
            wall_time_sec=elapsed,
            generated_new_tokens=generated_new_tokens,
            requested_new_tokens=requested_new_tokens,
            speculate_k=self.llm.config.speculate_k,
            metadata={
                "mode": self.mode,
                "target_model_path": self.model_path,
                "draft_model_path": self.draft_model_path,
                "request_count": len(prompts),
            },
            include_raw_metrics=False,
        )
        return outputs, profile

    def handle_generate_request(self, body: dict[str, Any]) -> dict[str, Any]:
        """Handle text generation requests."""
        prompts, is_single = parse_prompt_inputs(body)
        sampling_params = build_sampling_params_list(
            body,
            count=len(prompts),
            default_max_new_tokens=self.default_max_new_tokens,
        )
        return_metrics = bool(body.get("return_metrics", False))

        outputs, metrics = self.generate(
            prompts,
            sampling_params,
            return_metrics=return_metrics,
        )
        created = int(time.time())
        usage = self._usage_summary(prompts, outputs)
        formatted_outputs = [
            {
                "index": idx,
                "text": output["text"],
                "token_ids": output["token_ids"],
                "finish_reason": infer_finish_reason(output, sampling_params[idx]),
            }
            for idx, output in enumerate(outputs)
        ]

        response = {
            "id": f"gen-{uuid.uuid4().hex}",
            "object": "text_generation",
            "created": created,
            "model": self.model_id,
            "mode": self.mode,
            "outputs": formatted_outputs,
            "usage": usage,
        }
        if is_single:
            response["text"] = formatted_outputs[0]["text"]
            response["token_ids"] = formatted_outputs[0]["token_ids"]
            response["finish_reason"] = formatted_outputs[0]["finish_reason"]
        if metrics is not None:
            response["metrics"] = metrics
        return response

    def handle_chat_completion_request(self, body: dict[str, Any]) -> dict[str, Any]:
        """Handle OpenAI-style chat completion requests."""
        if bool(body.get("stream", False)):
            raise APIError(
                HTTPStatus.BAD_REQUEST,
                "`stream=true` is not supported by this server yet.",
            )

        prompt = render_chat_prompt(body.get("messages"), self.tokenizer)
        sampling_params = build_sampling_params_list(
            body,
            count=1,
            default_max_new_tokens=self.default_max_new_tokens,
        )[0]
        return_metrics = bool(body.get("return_metrics", False))

        outputs, metrics = self.generate([prompt], [sampling_params], return_metrics=return_metrics)
        output = outputs[0]
        usage = self._usage_summary([prompt], outputs)
        finish_reason = infer_finish_reason(output, sampling_params)
        created = int(time.time())

        response = {
            "id": f"chatcmpl-{uuid.uuid4().hex}",
            "object": "chat.completion",
            "created": created,
            "model": body.get("model") or self.model_id,
            "mode": self.mode,
            "choices": [
                {
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": output["text"],
                    },
                    "finish_reason": finish_reason,
                }
            ],
            "usage": usage,
        }
        if metrics is not None:
            response["metrics"] = metrics
        return response


class HasteHTTPServer(ThreadingHTTPServer):
    """HTTP server carrying a Haste service instance."""

    daemon_threads = True

    def __init__(self, server_address: tuple[str, int], service: HasteService):
        super().__init__(server_address, HasteRequestHandler)
        self.service = service


class HasteRequestHandler(BaseHTTPRequestHandler):
    """HTTP request handler for Haste."""

    server: HasteHTTPServer
    protocol_version = "HTTP/1.1"

    def log_message(self, format: str, *args) -> None:
        if self.server.service.args.verbose:
            super().log_message(format, *args)

    def _send_json(self, status: HTTPStatus, payload: dict[str, Any]) -> None:
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(status.value)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _send_error(self, status: HTTPStatus, message: str) -> None:
        self._send_json(
            status,
            {
                "error": {
                    "type": status.phrase.lower().replace(" ", "_"),
                    "message": message,
                    "code": status.value,
                }
            },
        )

    def _read_json_body(self) -> dict[str, Any]:
        content_length = self.headers.get("Content-Length")
        if content_length is None:
            raise APIError(HTTPStatus.LENGTH_REQUIRED, "Missing Content-Length header.")
        try:
            length = int(content_length)
        except ValueError as exc:
            raise APIError(HTTPStatus.BAD_REQUEST, "Invalid Content-Length header.") from exc
        raw = self.rfile.read(length)
        try:
            payload = json.loads(raw.decode("utf-8"))
        except json.JSONDecodeError as exc:
            raise APIError(HTTPStatus.BAD_REQUEST, f"Invalid JSON body: {exc.msg}") from exc
        if not isinstance(payload, dict):
            raise APIError(HTTPStatus.BAD_REQUEST, "JSON body must be an object.")
        return payload

    def do_GET(self) -> None:
        path = urlparse(self.path).path
        try:
            if path in {"/", ""}:
                self._send_json(
                    HTTPStatus.OK,
                    {
                        "service": "haste",
                        "status": "ok",
                        "endpoints": [
                            "GET /health",
                            "GET /v1/models",
                            "POST /v1/generate",
                            "POST /v1/chat/completions",
                        ],
                    },
                )
                return

            if path == "/health":
                self._send_json(
                    HTTPStatus.OK,
                    {
                        "status": "ok",
                        "service": "haste",
                        "model": self.server.service.model_metadata(),
                    },
                )
                return

            if path == "/v1/models":
                self._send_json(
                    HTTPStatus.OK,
                    {
                        "object": "list",
                        "data": [self.server.service.model_metadata()],
                    },
                )
                return

            self._send_error(HTTPStatus.NOT_FOUND, f"Unknown endpoint: {path}")
        except APIError as exc:
            self._send_error(exc.status, exc.message)
        except Exception as exc:  # pragma: no cover - defensive guard
            self._send_error(HTTPStatus.INTERNAL_SERVER_ERROR, str(exc))

    def do_POST(self) -> None:
        path = urlparse(self.path).path
        try:
            body = self._read_json_body()

            if path in {"/v1/generate", "/generate"}:
                response = self.server.service.handle_generate_request(body)
                self._send_json(HTTPStatus.OK, response)
                return

            if path in {"/v1/chat/completions", "/chat/completions"}:
                response = self.server.service.handle_chat_completion_request(body)
                self._send_json(HTTPStatus.OK, response)
                return

            self._send_error(HTTPStatus.NOT_FOUND, f"Unknown endpoint: {path}")
        except APIError as exc:
            self._send_error(exc.status, exc.message)
        except Exception as exc:  # pragma: no cover - defensive guard
            self._send_error(HTTPStatus.INTERNAL_SERVER_ERROR, str(exc))


def install_signal_handlers(server: HasteHTTPServer) -> None:
    """Install signal handlers for graceful shutdown."""

    def _handle_signal(signum, _frame):
        print(f"Received signal {signum}, shutting down server...", flush=True)
        server.shutdown()

    for sig in (signal.SIGINT, signal.SIGTERM):
        signal.signal(sig, _handle_signal)


def main() -> None:
    """CLI entry point."""
    args = build_parser().parse_args()
    validate_mode_args(args)

    print("Initializing Haste service...", flush=True)
    llm = LLM(**build_llm_kwargs(args))
    service = HasteService(llm, args)
    server = HasteHTTPServer((args.host, args.port), service)
    install_signal_handlers(server)

    print(
        f"Haste server listening on http://{args.host}:{args.port} "
        f"(mode={args.mode}, model={service.model_id})",
        flush=True,
    )
    try:
        server.serve_forever()
    finally:
        server.server_close()
        service.close()
        print("Haste server stopped.", flush=True)


if __name__ == "__main__":
    main()
