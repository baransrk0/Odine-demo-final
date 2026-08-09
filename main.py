"""FastAPI lifecycle and HTTP/SSE surface for the voice assistant."""

import asyncio
from collections.abc import Awaitable, Callable
from contextlib import asynccontextmanager
from io import BytesIO
import inspect
import logging
from pathlib import Path
import shutil
import tempfile
import time
from typing import Any
from uuid import UUID, uuid4
import wave

from fastapi import FastAPI, File, Header, Request, UploadFile
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse, Response
from sse_starlette import EventSourceResponse
from starlette.datastructures import Headers
from starlette.exceptions import HTTPException as StarletteHTTPException
from starlette.staticfiles import StaticFiles
from starlette.types import ASGIApp, Message, Receive, Scope, Send

from app.audio.conversion import convert_to_stt_wav, probe_duration
from app.audio.gpio import ListeningIndicator
from app.audio.input_mode import InputModeController
from app.audio.listening import ListeningBroadcaster
from app.audio.output import AudioOutputController, list_output_devices
from app.audio.rf_discovery import RFDiscoveryBuffer
from app.audio.rf_service import RFInputService
from app.audio.storage import AudioStorage
from app.audio.validation import save_upload, validate_upload
from app.config import Settings
from app.errors import ApiError
from app.schemas import (
    AudioInputRequest,
    AudioInputState,
    AudioOutputRequest,
    AudioOutputState,
    AudioOutputsResponse,
    ErrorBody,
    TurnCreated,
)
from app.turns.manager import TurnManager
from app.turns.metrics import RecentMetrics
from app.turns.submitter import TurnSubmitter


logger = logging.getLogger(__name__)
AudioToolsValidator = Callable[[], Any]
AudioProbe = Callable[[Path, float | None], float]
AudioConverter = Callable[[Path, Path], None]
Sleep = Callable[[float], Awaitable[None]]
RFServiceFactory = Callable[
    [
        Settings,
        AudioStorage,
        TurnSubmitter,
        RFDiscoveryBuffer,
        ListeningIndicator,
        ListeningBroadcaster,
    ],
    RFInputService,
]

_MULTIPART_OVERHEAD_BYTES = 64 * 1024
_STATE_CHANGING_METHODS = frozenset({"POST", "PUT", "PATCH", "DELETE"})
_DISABLED_DOCUMENTATION_PREFIXES = (
    "docs",
    "openapi.json",
    "redoc",
)
_DEFAULT_FRONTEND_DIST = Path(__file__).resolve().parents[2] / "frontend" / "dist"
_CONTENT_SUFFIXES = {
    "audio/webm": ".webm",
    "audio/ogg": ".ogg",
    "audio/wav": ".wav",
    "audio/mp4": ".mp4",
    "audio/mpeg": ".mp3",
}


def _safe_error_response(
    *,
    status_code: int,
    code: str,
    message: str,
    headers: dict[str, str] | None = None,
) -> JSONResponse:
    return JSONResponse(
        status_code=status_code,
        headers=headers,
        content=ErrorBody(code=code, message=message).model_dump(),
    )


class _RawTurnBodyLimitMiddleware:
    """Bound raw multipart input before Starlette can parse or spool uploads."""

    def __init__(self, app: ASGIApp, *, max_audio_bytes: int) -> None:
        self.app = app
        self.max_body_bytes = max_audio_bytes + _MULTIPART_OVERHEAD_BYTES

    async def __call__(
        self,
        scope: Scope,
        receive: Receive,
        send: Send,
    ) -> None:
        if not self._applies(scope):
            await self.app(scope, receive, send)
            return

        headers = Headers(scope=scope)
        content_length = headers.get("content-length")
        if content_length is not None:
            try:
                declared_length = int(content_length)
            except ValueError:
                declared_length = None
            if (
                declared_length is not None
                and declared_length > self.max_body_bytes
            ):
                await self._reject(scope, receive, send)
                return

        buffered: list[Message] = []
        received_bytes = 0
        while True:
            message = await receive()
            if message["type"] == "http.request":
                received_bytes += len(message.get("body", b""))
                if received_bytes > self.max_body_bytes:
                    await self._reject(scope, receive, send)
                    return
            buffered.append(message)
            if message["type"] == "http.disconnect" or (
                message["type"] == "http.request"
                and not message.get("more_body", False)
            ):
                break

        next_message = 0

        async def replay() -> Message:
            nonlocal next_message
            if next_message < len(buffered):
                message = buffered[next_message]
                next_message += 1
                return message
            return {"type": "http.disconnect"}

        await self.app(scope, replay, send)

    @staticmethod
    def _applies(scope: Scope) -> bool:
        if scope["type"] != "http":
            return False
        if scope.get("method") != "POST" or scope.get("path") != "/api/turns":
            return False
        content_type = Headers(scope=scope).get("content-type", "")
        return content_type.lower().startswith("multipart/form-data")

    @staticmethod
    async def _reject(
        scope: Scope,
        receive: Receive,
        send: Send,
    ) -> None:
        response = _safe_error_response(
            status_code=413,
            code="audio_too_large",
            message="Ses kaydı boyut sınırını aşıyor.",
        )
        await response(scope, receive, send)


class _SafeCORSMiddleware(CORSMiddleware):
    """Reject disallowed state changes and keep CORS errors safely structured."""

    async def __call__(
        self,
        scope: Scope,
        receive: Receive,
        send: Send,
    ) -> None:
        if scope["type"] == "http":
            method = scope.get("method", "")
            origin = Headers(scope=scope).get("origin")
            if (
                method in _STATE_CHANGING_METHODS
                and origin is not None
                and not self.is_allowed_origin(origin=origin)
            ):
                response = _safe_error_response(
                    status_code=403,
                    code="cors_not_allowed",
                    message="Bu kaynaktan erişime izin verilmiyor.",
                )
                await response(scope, receive, send)
                return
        await super().__call__(scope, receive, send)

    def preflight_response(self, request_headers: Headers) -> Response:
        response = super().preflight_response(request_headers)
        if response.status_code < 400:
            return response
        headers = {
            key: value
            for key, value in response.headers.items()
            if key.lower() not in {"content-length", "content-type"}
        }
        return _safe_error_response(
            status_code=response.status_code,
            headers=headers,
            code="cors_not_allowed",
            message="Bu kaynaktan erişime izin verilmiyor.",
        )


def _validate_audio_tools() -> None:
    """Fail startup safely when required local audio tools are unavailable."""
    if shutil.which("ffmpeg") is None or shutil.which("ffprobe") is None:
        raise RuntimeError("Required audio tools are unavailable.")


def _validate_hardware_audio_tools(settings: Settings) -> None:
    required = ["amixer", "arecord"]
    if settings.local_audio_playback:
        required.append("aplay")
    missing = [tool for tool in required if shutil.which(tool) is None]
    if missing:
        raise RuntimeError(f"Required RF audio tools are unavailable: {', '.join(missing)}")


async def _load_speech_runtime(runtime: object, label: str) -> None:
    load = getattr(runtime, "load", None)
    if not callable(load):
        return
    try:
        result = load()
        if inspect.isawaitable(result):
            await result
    except Exception:
        setattr(runtime, "ready", False)
        logger.warning("%s runtime could not be loaded.", label)


async def _probe_runtime(runtime: object, label: str) -> None:
    health = getattr(runtime, "health", None)
    if not callable(health):
        return
    try:
        ready = health()
        if inspect.isawaitable(ready):
            ready = await ready
        setattr(runtime, "ready", bool(ready))
    except Exception:
        setattr(runtime, "ready", False)
        logger.warning("%s runtime could not be probed.", label)


async def _expiry_loop(
    storage: AudioStorage,
    manager: TurnManager,
    retention_seconds: int,
    *,
    sleep: Sleep = asyncio.sleep,
) -> None:
    interval = max(min(retention_seconds / 2, 30), 0.25)
    while True:
        await sleep(interval)
        try:
            await asyncio.to_thread(storage.expire)
        except Exception:
            logger.warning("Audio storage expiry failed.")
        try:
            await manager.expire()
        except Exception:
            logger.warning("Turn event expiry failed.")


async def _cleanup_request_upload(
    upload: UploadFile | object,
    turn_directory: Path,
    *,
    remove_directory: bool,
) -> None:
    close = getattr(upload, "close", None)
    if callable(close):
        try:
            closed = close()
            if inspect.isawaitable(closed):
                await closed
        except Exception:
            logger.warning(
                "Rejected upload close failed."
                if remove_directory
                else "Upload close failed."
            )

    if not remove_directory:
        return
    try:
        await asyncio.to_thread(shutil.rmtree, turn_directory)
    except FileNotFoundError:
        return
    except Exception:
        logger.warning("Rejected upload deletion failed.")


async def _finish_request_upload(
    upload: UploadFile | object,
    turn_directory: Path,
    *,
    remove_directory: bool,
) -> None:
    """Finish close/delete even if the owning request is cancelled."""
    finalizer = asyncio.create_task(
        _cleanup_request_upload(
            upload,
            turn_directory,
            remove_directory=remove_directory,
        )
    )
    cancellation: asyncio.CancelledError | None = None
    while not finalizer.done():
        try:
            await asyncio.shield(finalizer)
        except asyncio.CancelledError as error:
            if cancellation is None:
                cancellation = error
        except Exception:
            break

    try:
        finalizer.result()
    except asyncio.CancelledError:
        raise
    except Exception:
        logger.warning("Upload cleanup finalization failed.")
    if cancellation is not None:
        raise cancellation


def create_app(
    settings: Settings | None = None,
    *,
    storage: AudioStorage | None = None,
    validate_audio_tools: AudioToolsValidator = _validate_audio_tools,
    probe_audio: AudioProbe = probe_duration,
    convert_audio: AudioConverter = convert_to_stt_wav,
    frontend_dist: Path | None = None,
    rf_discovery: RFDiscoveryBuffer | None = None,
    rf_service_factory: RFServiceFactory = RFInputService,
) -> FastAPI:
    """Build an application with injectable local runtimes for deterministic tests."""
    configured_settings = settings or Settings()

    @asynccontextmanager
    async def lifespan(application: FastAPI):
        validation_result = validate_audio_tools()
        if inspect.isawaitable(validation_result):
            validation_result = await validation_result
        if validation_result is False:
            raise RuntimeError("Required audio tools are unavailable.")
        if configured_settings.audio_input_mode == "rf_i2s":
            _validate_hardware_audio_tools(configured_settings)

        temporary_audio: tempfile.TemporaryDirectory[str] | None = None
        active_storage = storage
        if active_storage is None:
            temporary_audio = tempfile.TemporaryDirectory(
                prefix="orin-voice-audio-"
            )
            active_storage = AudioStorage(
                Path(temporary_audio.name),
                retention_seconds=configured_settings.audio_retention_seconds,
            )

        recent_metrics = RecentMetrics(configured_settings)
        turn_manager = TurnManager(
            configured_settings.audio_retention_seconds,
        )
        audio_output = AudioOutputController(
            mode="device" if configured_settings.local_audio_playback else "browser",
            device=configured_settings.speaker_device,
        )
        submitter = TurnSubmitter(
            active_storage,
            turn_manager,
            recent_metrics,
            configured_settings,
        )
        active_rf_discovery = rf_discovery or RFDiscoveryBuffer(
            configured_settings.audio_retention_seconds
        )
        listening_broadcaster = ListeningBroadcaster()
        listening_indicator = ListeningIndicator(configured_settings)
        listening_indicator.setup()
        expiry_task = asyncio.create_task(
            _expiry_loop(
                active_storage,
                turn_manager,
                configured_settings.audio_retention_seconds,
            ),
            name="audio-expiry",
        )

        application.state.settings = configured_settings
        application.state.storage = active_storage
        application.state.recent_metrics = recent_metrics
        application.state.turn_manager = turn_manager
        application.state.turn_submitter = submitter
        application.state.rf_discovery = active_rf_discovery
        application.state.listening = listening_broadcaster
        application.state.listening_indicator = listening_indicator
        application.state.audio_output = audio_output

        def make_rf_service() -> RFInputService:
            return rf_service_factory(
                configured_settings,
                active_storage,
                submitter,
                active_rf_discovery,
                listening_indicator,
                listening_broadcaster,
            )

        input_mode = InputModeController(
            mode=configured_settings.audio_input_mode,
            rf_factory=make_rf_service,
            validate=lambda: _validate_hardware_audio_tools(configured_settings),
        )
        application.state.input_mode = input_mode

        try:
            input_mode.start()
            yield
        finally:
            await input_mode.shutdown()
            listening_indicator.cleanup()
            expiry_task.cancel()
            await asyncio.gather(expiry_task, return_exceptions=True)
            await turn_manager.shutdown()
            if temporary_audio is not None:
                temporary_audio.cleanup()

    application = FastAPI(
        lifespan=lifespan,
        openapi_url=None,
        docs_url=None,
        redoc_url=None,
        swagger_ui_oauth2_redirect_url=None,
    )
    application.add_middleware(
        _RawTurnBodyLimitMiddleware,
        max_audio_bytes=configured_settings.audio_max_bytes,
    )
    application.add_middleware(
        _SafeCORSMiddleware,
        allow_origins=configured_settings.cors_origins,
        allow_credentials=False,
        allow_methods=["GET", "POST", "OPTIONS"],
        allow_headers=["Content-Type", "Last-Event-ID"],
    )

    @application.exception_handler(ApiError)
    async def api_error_handler(
        request: Request,
        error: ApiError,
    ) -> JSONResponse:
        del request
        return JSONResponse(
            status_code=error.status_code,
            content=ErrorBody(
                code=error.code,
                message=error.message,
            ).model_dump(),
        )

    @application.exception_handler(RequestValidationError)
    async def validation_error_handler(
        request: Request,
        error: RequestValidationError,
    ) -> JSONResponse:
        del request, error
        return JSONResponse(
            status_code=422,
            content=ErrorBody(
                code="invalid_request",
                message="İstek bilgileri geçersiz.",
            ).model_dump(),
        )

    @application.exception_handler(StarletteHTTPException)
    async def http_error_handler(
        request: Request,
        error: StarletteHTTPException,
    ) -> JSONResponse:
        del request
        if error.status_code == 404:
            body = ErrorBody(
                code="not_found",
                message="İstenen kaynak bulunamadı.",
            )
        else:
            body = ErrorBody(
                code="request_failed",
                message="İstek tamamlanamadı.",
            )
        return JSONResponse(status_code=error.status_code, content=body.model_dump())

    @application.exception_handler(Exception)
    async def unexpected_error_handler(
        request: Request,
        error: Exception,
    ) -> JSONResponse:
        del request, error
        logger.error("Unhandled API exception.")
        return JSONResponse(
            status_code=500,
            content=ErrorBody(
                code="internal_error",
                message="İşlem tamamlanamadı, tekrar deneyin.",
            ).model_dump(),
        )

    @application.get("/api/health")
    async def health(request: Request) -> dict[str, object]:
        settings = request.app.state.settings
        return {
            "status": "ok",
            "audio_input_mode": request.app.state.input_mode.mode,
            "local_audio_playback": request.app.state.settings.local_audio_playback,
            "audio_output_mode": request.app.state.audio_output.mode,
            "audio_output_device": request.app.state.audio_output.device,
            "orchestrator_base_url": "configured" if settings.orchestrator_base_url.strip() else "unset",
        }

    @application.get("/api/rf/turns/events")
    async def rf_turn_events(
        request: Request,
        last_event_id: str | None = Header(
            default=None,
            alias="Last-Event-ID",
        ),
    ) -> EventSourceResponse:
        if request.app.state.input_mode.mode != "rf_i2s":
            raise ApiError(
                "rf_input_disabled",
                "RF ses girişi etkin değil.",
                409,
            )

        cursor: int | None = None
        if last_event_id is not None:
            try:
                cursor = int(last_event_id)
            except ValueError:
                raise ApiError(
                    "invalid_request",
                    "İstek bilgileri geçersiz.",
                    422,
                ) from None

        async def stream():
            async for event in request.app.state.rf_discovery.subscribe(cursor):
                yield event.as_sse()

        return EventSourceResponse(stream())

    @application.get("/api/rf/listening/events")
    async def rf_listening_events(
        request: Request,
        last_event_id: str | None = Header(
            default=None,
            alias="Last-Event-ID",
        ),
    ) -> EventSourceResponse:
        if request.app.state.input_mode.mode != "rf_i2s":
            raise ApiError(
                "rf_input_disabled",
                "RF ses girişi etkin değil.",
                409,
            )

        cursor: int | None = None
        if last_event_id is not None:
            try:
                cursor = int(last_event_id)
            except ValueError:
                raise ApiError(
                    "invalid_request",
                    "İstek bilgileri geçersiz.",
                    422,
                ) from None

        async def stream():
            async for event in request.app.state.listening.subscribe(cursor):
                yield event.as_sse()

        return EventSourceResponse(stream())

    @application.get("/api/audio/input", response_model=AudioInputState)
    async def audio_input(request: Request) -> AudioInputState:
        return AudioInputState(mode=request.app.state.input_mode.mode)

    @application.post("/api/audio/input", response_model=AudioInputState)
    async def set_audio_input(
        request: Request,
        body: AudioInputRequest,
    ) -> AudioInputState:
        controller = request.app.state.input_mode
        try:
            await controller.set_mode(body.mode)
        except ApiError:
            raise
        except Exception as error:
            raise ApiError(
                "rf_start_failed",
                "RF girişi başlatılamadı.",
                503,
            ) from error
        return AudioInputState(mode=controller.mode)

    @application.get("/api/audio/outputs", response_model=AudioOutputsResponse)
    async def audio_outputs(request: Request) -> AudioOutputsResponse:
        controller = request.app.state.audio_output
        devices = await asyncio.to_thread(list_output_devices)
        return AudioOutputsResponse(
            current=AudioOutputState(mode=controller.mode, device=controller.device),
            devices=devices,
        )

    @application.post("/api/audio/output", response_model=AudioOutputState)
    async def set_audio_output(
        request: Request,
        body: AudioOutputRequest,
    ) -> AudioOutputState:
        controller = request.app.state.audio_output
        if body.mode == "device":
            device = body.device or controller.device
            if not device.strip():
                raise ApiError(
                    "invalid_request",
                    "İstek bilgileri geçersiz.",
                    422,
                )
            controller.configure(mode="device", device=device)
        else:
            controller.configure(mode="browser")
        return AudioOutputState(mode=controller.mode, device=controller.device)

    @application.post(
        "/api/turns",
        response_model=TurnCreated,
        status_code=202,
    )
    async def create_turn(
        request: Request,
        audio: UploadFile = File(...),
    ) -> TurnCreated:
        turn_id = uuid4()
        started_ns = time.perf_counter_ns()
        content_type = (
            audio.content_type.split(";", 1)[0].strip().lower()
            if audio.content_type
            else None
        )
        suffix = _CONTENT_SUFFIXES.get(content_type or "", ".upload")
        turn_storage: AudioStorage = request.app.state.storage
        source_path = turn_storage.upload_path(turn_id, suffix)
        input_path = turn_storage.intermediate_path(turn_id)
        admitted = False

        try:
            byte_count = await save_upload(
                audio,
                source_path,
                request.app.state.settings.audio_max_bytes,
            )
            validate_upload(
                content_type,
                byte_count,
                request.app.state.settings.audio_max_bytes,
            )
            normalize_started_ns = time.perf_counter_ns()
            await asyncio.to_thread(convert_audio, source_path, input_path)
            duration = await asyncio.to_thread(
                probe_audio,
                input_path,
                request.app.state.settings.audio_max_seconds,
            )
            normalize_ms = max(time.perf_counter_ns() - normalize_started_ns, 0) / 1_000_000
            created = await request.app.state.turn_submitter.submit_normalized_wav(
                turn_id,
                input_path,
                source_path=source_path,
                duration_seconds=duration,
                byte_count=byte_count,
                content_type=content_type,
                normalize_ms=normalize_ms,
                started_ns=started_ns,
            )
            admitted = True
            return created
        finally:
            await _finish_request_upload(
                audio,
                turn_storage.turn_dir(turn_id),
                remove_directory=not admitted,
            )

    @application.get("/api/turns/{turn_id}/events")
    async def turn_events(
        request: Request,
        turn_id: UUID,
        last_event_id: str | None = Header(
            default=None,
            alias="Last-Event-ID",
        ),
    ) -> EventSourceResponse:
        turn = request.app.state.turn_manager.get(turn_id)
        if turn is None:
            raise ApiError(
                "turn_not_found",
                "İşlem bulunamadı veya süresi doldu.",
                404,
            )

        cursor: int | None = None
        if last_event_id is not None:
            try:
                cursor = int(last_event_id)
            except ValueError:
                raise ApiError(
                    "invalid_request",
                    "İstek bilgileri geçersiz.",
                    422,
                ) from None

        async def stream():
            async for event in turn.events.subscribe(cursor):
                yield event.as_sse()

        return EventSourceResponse(stream())

    @application.get("/api/metrics/recent")
    async def recent_metrics(request: Request) -> list[dict[str, object]]:
        return [
            metric.model_dump(mode="json")
            for metric in request.app.state.recent_metrics.snapshot()
        ]

    _mount_frontend(application, frontend_dist)
    return application


def _mount_frontend(
    application: FastAPI,
    frontend_dist: Path | None,
) -> None:
    """Serve a built Vite app without shadowing the API or disabled consoles."""
    if frontend_dist is None:
        return
    resolved_dist = frontend_dist.resolve()
    index_path = resolved_dist / "index.html"
    if not index_path.is_file():
        logger.info("Frontend build not found; static UI is disabled.")
        return

    assets_path = resolved_dist / "assets"
    if assets_path.is_dir():
        application.mount(
            "/assets",
            StaticFiles(directory=assets_path),
            name="frontend-assets",
        )

    @application.get("/{frontend_path:path}", include_in_schema=False)
    async def frontend_app(frontend_path: str) -> FileResponse:
        normalized = frontend_path.lstrip("/")
        if (
            normalized == "api"
            or normalized.startswith("api/")
            or _is_disabled_documentation_path(normalized)
            or normalized == "assets"
            or normalized.startswith("assets/")
        ):
            raise StarletteHTTPException(status_code=404)

        candidate = (resolved_dist / normalized).resolve()
        if candidate.is_relative_to(resolved_dist) and candidate.is_file():
            return FileResponse(candidate)
        return FileResponse(index_path, media_type="text/html")


def _is_disabled_documentation_path(path: str) -> bool:
    return any(
        path == prefix or path.startswith(f"{prefix}/")
        for prefix in _DISABLED_DOCUMENTATION_PREFIXES
    )


def _audio_not_found() -> ApiError:
    return ApiError(
        "audio_not_found",
        "Ses kaydı bulunamadı veya süresi doldu.",
        404,
    )


def _combine_wav_chunks(chunks: list[Path]) -> bytes:
    """Combine compatible PCM chunks, with a safe fallback for adapter doubles."""
    output = BytesIO()
    try:
        parameters: tuple[int, int, int, str, str] | None = None
        frames: list[bytes] = []
        for path in chunks:
            with wave.open(str(path), "rb") as source:
                current = (
                    source.getnchannels(),
                    source.getsampwidth(),
                    source.getframerate(),
                    source.getcomptype(),
                    source.getcompname(),
                )
                if parameters is None:
                    parameters = current
                elif current != parameters:
                    raise wave.Error("incompatible WAV chunks")
                frames.append(source.readframes(source.getnframes()))

        if parameters is None:
            raise wave.Error("empty WAV chunks")
        with wave.open(output, "wb") as target:
            target.setnchannels(parameters[0])
            target.setsampwidth(parameters[1])
            target.setframerate(parameters[2])
            target.setcomptype(parameters[3], parameters[4])
            for frame in frames:
                target.writeframes(frame)
        return output.getvalue()
    except (EOFError, OSError, wave.Error):
        return b"".join(path.read_bytes() for path in chunks)


app = create_app(frontend_dist=_DEFAULT_FRONTEND_DIST)
