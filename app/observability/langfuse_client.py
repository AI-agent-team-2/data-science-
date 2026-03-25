from __future__ import annotations

import logging
from typing import Any
from uuid import uuid4

from app.config import settings
from app.observability.sanitize import sanitize_payload

logger = logging.getLogger(__name__)

_langfuse_client: Any | None = None
_client_init_attempted = False
_callback_handler_class: Any | None = None
_callback_init_error: str | None = None


class _LangfuseObservationAdapter:
    """
    Адаптер observation Langfuse v3 под интерфейс текущего проекта.

    Объект инкапсулирует создание и завершение observation через
    `start_as_current_observation`, но предоставляет привычные методы
    `span`, `update`, `end`, `event`.
    """

    def __init__(
        self,
        *,
        client: Any,
        name: str,
        trace_id: str | None = None,
        parent_span_id: str | None = None,
        input_payload: Any | None = None,
        metadata: dict[str, Any] | None = None,
        as_type: str = "span",
    ) -> None:
        self._client = client
        self._name = name
        self._trace_id = str(trace_id).strip() if trace_id else None
        self._parent_span_id = (
            str(parent_span_id).strip() if parent_span_id else None
        )
        self._input_payload = sanitize_payload(input_payload)
        self._metadata = sanitize_payload(metadata or {})
        self._as_type = as_type

        self._ctx_manager: Any | None = None
        self._observation: Any | None = None
        self._ended = False

        self.id: str | None = None

    @property
    def trace_id(self) -> str | None:
        """Возвращает trace_id текущего observation."""
        return self._trace_id

    def start(self) -> "_LangfuseObservationAdapter":
        """Создает observation и сохраняет его идентификаторы."""
        trace_context: dict[str, str] = {}
        if self._trace_id:
            trace_context["trace_id"] = self._trace_id
        if self._parent_span_id:
            trace_context["parent_span_id"] = self._parent_span_id

        kwargs: dict[str, Any] = {
            "name": self._name,
            "as_type": self._as_type,
            "input": self._input_payload,
            "metadata": self._metadata,
            "end_on_exit": False,
        }
        if trace_context:
            kwargs["trace_context"] = trace_context

        self._ctx_manager = self._client.start_as_current_observation(**kwargs)
        self._observation = self._ctx_manager.__enter__()

        current_trace_id = self._client.get_current_trace_id()
        current_observation_id = self._client.get_current_observation_id()
        self._trace_id = current_trace_id or self._trace_id
        self.id = current_observation_id
        return self

    def span(
        self,
        *,
        name: str,
        input: Any | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> "_LangfuseObservationAdapter":
        """
        Создает дочерний span в том же trace.

        Parameters
        ----------
        name : str
            Имя дочернего span.
        input : Any | None
            Входные данные span.
        metadata : dict[str, Any] | None
            Дополнительные метаданные.
        """
        child = _LangfuseObservationAdapter(
            client=self._client,
            name=name,
            trace_id=self._trace_id,
            parent_span_id=self.id,
            input_payload=input,
            metadata=metadata,
            as_type="span",
        )
        return child.start()

    def event(
        self,
        *,
        name: str,
        level: str | None = None,
        input: Any | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        """
        Записывает событие как короткий дочерний span.

        Langfuse v3 не использует отдельный тип `event` в этом wrapper,
        поэтому событие фиксируется через дочерний observation.
        """
        event_metadata = dict(metadata or {})
        event_metadata["event_name"] = name
        if level:
            event_metadata["event_level"] = level

        child = self.span(
            name=f"event:{name}",
            input=input,
            metadata=event_metadata,
        )
        child.end(
            output=input,
            metadata=event_metadata,
            level=level,
            status_message=str(event_metadata.get("message", "")) or None,
        )

    def update(
        self,
        *,
        input: Any | None = None,
        output: Any | None = None,
        metadata: dict[str, Any] | None = None,
        level: str | None = None,
        status_message: str | None = None,
    ) -> None:
        """Обновляет текущий observation безопасным payload."""
        if self._ended or self._observation is None:
            return

        update_kwargs: dict[str, Any] = {}
        if input is not None:
            update_kwargs["input"] = sanitize_payload(input)
        if output is not None:
            update_kwargs["output"] = sanitize_payload(output)
        if metadata:
            update_kwargs["metadata"] = sanitize_payload(metadata)
        if level:
            update_kwargs["level"] = level
        if status_message:
            update_kwargs["status_message"] = sanitize_payload(status_message)

        if not update_kwargs:
            return

        updater = getattr(self._observation, "update", None)
        if callable(updater):
            updater(**update_kwargs)

    def end(
        self,
        *,
        output: Any | None = None,
        metadata: dict[str, Any] | None = None,
        level: str | None = None,
        status_message: str | None = None,
    ) -> None:
        """Завершает observation и закрывает связанный контекст."""
        if self._ended:
            return

        self.update(
            output=output,
            metadata=metadata,
            level=level,
            status_message=status_message,
        )

        ender = getattr(self._observation, "end", None)
        if callable(ender):
            try:
                ender()
            except Exception:
                logger.debug("Observation уже завершен или недоступен для end().")

        if self._ctx_manager is not None:
            try:
                self._ctx_manager.__exit__(None, None, None)
            except Exception:
                logger.debug("Не удалось корректно закрыть контекст observation.")

        self._ended = True


class _LangfuseTraceAdapter:
    """
    Адаптер trace, который инкапсулирует root observation.

    Нужен, чтобы `create_trace` создавал реальный trace-entity в Langfuse,
    но код проекта продолжал работать с объектом, похожим на observation
    (`span`, `update`, `end`, `event`).
    """

    def __init__(
        self,
        *,
        trace_id: str,
        root_observation: _LangfuseObservationAdapter,
        trace_entity: Any | None = None,
    ) -> None:
        self.trace_id = trace_id
        self.id = root_observation.id
        self._root = root_observation
        self._trace_entity = trace_entity

    def span(
        self,
        *,
        name: str,
        input: Any | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> _LangfuseObservationAdapter:
        return self._root.span(name=name, input=input, metadata=metadata)

    def update(
        self,
        *,
        input: Any | None = None,
        output: Any | None = None,
        metadata: dict[str, Any] | None = None,
        level: str | None = None,
        status_message: str | None = None,
    ) -> None:
        self._root.update(
            input=input,
            output=output,
            metadata=metadata,
            level=level,
            status_message=status_message,
        )

        # Поддерживаем синхронизацию trace-level полей, если trace-entity доступен.
        if self._trace_entity is None:
            return

        trace_update: dict[str, Any] = {}
        if input is not None:
            trace_update["input"] = sanitize_payload(input)
        if output is not None:
            trace_update["output"] = sanitize_payload(output)
        if metadata:
            trace_update["metadata"] = sanitize_payload(metadata)
        if not trace_update:
            return

        updater = getattr(self._trace_entity, "update", None)
        if callable(updater):
            try:
                updater(**trace_update)
            except Exception:
                logger.debug("Не удалось обновить trace-entity через trace.update().")

    def end(
        self,
        *,
        output: Any | None = None,
        metadata: dict[str, Any] | None = None,
        level: str | None = None,
        status_message: str | None = None,
    ) -> None:
        self._root.end(
            output=output,
            metadata=metadata,
            level=level,
            status_message=status_message,
        )

    def event(
        self,
        *,
        name: str,
        level: str | None = None,
        input: Any | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        self._root.event(name=name, level=level, input=input, metadata=metadata)


def _is_enabled() -> bool:
    """Проверяет, включена ли интеграция Langfuse и заданы ли ключи."""
    return bool(
        settings.langfuse_enabled
        and settings.langfuse_public_key
        and settings.langfuse_secret_key
    )


def get_langfuse_client() -> Any | None:
    """
    Возвращает singleton-клиент Langfuse.

    Returns
    -------
    Any | None
        Клиент Langfuse или `None`, если интеграция отключена/недоступна.
    """
    global _langfuse_client, _client_init_attempted
    if _client_init_attempted:
        return _langfuse_client

    _client_init_attempted = True
    if not _is_enabled():
        logger.info("Langfuse отключен или не сконфигурирован.")
        return None

    try:
        from langfuse import Langfuse  # type: ignore

        _langfuse_client = Langfuse(
            public_key=settings.langfuse_public_key,
            secret_key=settings.langfuse_secret_key,
            host=settings.langfuse_host,
        )
    except Exception:
        logger.exception("Не удалось инициализировать Langfuse, observability отключена.")
        _langfuse_client = None

    return _langfuse_client


def get_langchain_callback_handler(
    trace_id: str | None = None,
    session_id: str | None = None,
    user_id: str | None = None,
    parent_observation_id: str | None = None,
    tags: list[str] | None = None,
) -> Any | None:
    """
    Возвращает singleton `CallbackHandler` для LangChain.

    Parameters
    ----------
    trace_id : str | None
        Аргумент сохранен для совместимости сигнатуры.
    session_id : str | None
        Аргумент сохранен для совместимости сигнатуры.
    user_id : str | None
        Аргумент сохранен для совместимости сигнатуры.
    parent_observation_id : str | None
        Идентификатор родительского observation для явной привязки callback.
    tags : list[str] | None
        Теги trace/наблюдения для callback handler.

    Returns
    -------
    Any | None
        Экземпляр callback handler или `None`, если инициализация не удалась.
    """
    _ = (trace_id, session_id, user_id, parent_observation_id, tags)
    global _callback_handler_class, _callback_init_error
    if not _is_enabled():
        return None

    _ = get_langfuse_client()
    try:
        if _callback_handler_class is None:
            from langfuse.langchain import CallbackHandler as LangfuseCallbackHandler  # type: ignore

            _callback_handler_class = LangfuseCallbackHandler

        constructor_kwargs: dict[str, Any] = {}
        if trace_id:
            constructor_kwargs["trace_id"] = str(trace_id).strip()
        if session_id:
            constructor_kwargs["session_id"] = str(session_id).strip()
        if user_id:
            constructor_kwargs["user_id"] = str(user_id).strip()
        if parent_observation_id:
            constructor_kwargs["parent_observation_id"] = str(parent_observation_id).strip()
        if tags:
            constructor_kwargs["tags"] = tags

        callback_handler: Any
        if constructor_kwargs:
            try:
                callback_handler = _callback_handler_class(**constructor_kwargs)
            except TypeError:
                # Фолбэк на старые/ограниченные сигнатуры CallbackHandler.
                callback_handler = _callback_handler_class()
        else:
            callback_handler = _callback_handler_class()
        _callback_init_error = None
        logger.debug("Langfuse CallbackHandler успешно создан для запроса.")
        return callback_handler
    except Exception as exc:
        _callback_handler_class = None
        _callback_init_error = str(exc)
        logger.error(
            "Не удалось инициализировать Langfuse CallbackHandler: %s. "
            "Проверьте совместимость версий langfuse/langchain.",
            _callback_init_error,
        )
        return None


def get_callback_init_error() -> str | None:
    """
    Возвращает последнюю ошибку инициализации callback.

    Returns
    -------
    str | None
        Текст ошибки или `None`, если ошибок не было.
    """
    return _callback_init_error


def _safe_call(obj: Any, method_name: str, **kwargs: Any) -> Any | None:
    """Безопасно вызывает метод объекта по имени."""
    method = getattr(obj, method_name, None)
    if callable(method):
        return method(**kwargs)
    return None


def create_trace(
    name: str,
    session_id: str,
    trace_id: str | None = None,
    input_payload: Any | None = None,
    metadata: dict[str, Any] | None = None,
) -> Any | None:
    """
    Создает trace в Langfuse.

    Returns
    -------
    Any | None
        Объект trace или `None`, если создать trace не удалось.
    """
    client = get_langfuse_client()
    if client is None:
        return None

    try:
        trace_metadata = dict(metadata or {})
        if session_id:
            trace_metadata["langfuse_session_id"] = session_id

        effective_trace_id = str(trace_id or "").strip() or uuid4().hex

        # 1) Пытаемся создать trace-entity как первичную сущность.
        trace_entity: Any | None = None
        try:
            trace_entity = _safe_call(
                client,
                "trace",
                id=effective_trace_id,
                name=name,
                input=sanitize_payload(input_payload),
                metadata=sanitize_payload(trace_metadata),
            )
        except Exception:
            logger.debug("Не удалось создать trace-entity через client.trace(); используем root observation.")

        # 2) Создаем root observation внутри trace.
        root_span = _LangfuseObservationAdapter(
            client=client,
            name=name,
            trace_id=effective_trace_id,
            parent_span_id=None,
            input_payload=input_payload,
            metadata=trace_metadata,
            as_type="span",
        ).start()

        trace_adapter = _LangfuseTraceAdapter(
            trace_id=effective_trace_id,
            root_observation=root_span,
            trace_entity=trace_entity,
        )

        logger.debug("Создан trace %s с root observation %s", trace_adapter.trace_id, trace_adapter.id)
        return trace_adapter
    except Exception:
        logger.exception("Не удалось создать Langfuse trace: %s", name)
        return None


def create_span(
    parent: Any | None,
    name: str,
    input_payload: Any | None = None,
    metadata: dict[str, Any] | None = None,
) -> Any | None:
    """
    Создает дочерний span для текущего trace/span.

    Returns
    -------
    Any | None
        Объект span или `None` при ошибке.
    """
    try:
        if parent is not None:
            if isinstance(parent, _LangfuseObservationAdapter):
                return parent.span(
                    name=name,
                    input=input_payload,
                    metadata=metadata or {},
                )

            return _safe_call(
                parent,
                "span",
                name=name,
                input=sanitize_payload(input_payload),
                metadata=sanitize_payload(metadata or {}),
            )

        client = get_langfuse_client()
        if client is None:
            return None

        current_trace_id = _safe_call(client, "get_current_trace_id")
        current_observation_id = _safe_call(client, "get_current_observation_id")
        if not current_trace_id:
            return None

        adapter = _LangfuseObservationAdapter(
            client=client,
            name=name,
            trace_id=str(current_trace_id),
            parent_span_id=str(current_observation_id) if current_observation_id else None,
            input_payload=input_payload,
            metadata=metadata,
            as_type="span",
        )
        return adapter.start()
    except Exception:
        logger.exception("Не удалось создать Langfuse span: %s", name)
        return None


def end_observation(
    target: Any | None,
    output_payload: Any | None = None,
    metadata: dict[str, Any] | None = None,
) -> None:
    """Завершает span или trace с очищенным payload."""
    if target is None:
        return

    sanitized_output = sanitize_payload(output_payload)
    sanitized_meta = sanitize_payload(metadata or {})

    try:
        ended = _safe_call(
            target,
            "end",
            output=sanitized_output,
            metadata=sanitized_meta,
        )
        if ended is None:
            _safe_call(target, "update", output=sanitized_output, metadata=sanitized_meta)
    except Exception:
        logger.exception("Не удалось завершить observation.")


def capture_error(
    target: Any | None,
    error: Exception | str,
    metadata: dict[str, Any] | None = None,
) -> None:
    """Отправляет информацию об ошибке в Langfuse без влияния на бизнес-логику."""
    if target is None:
        return

    error_text = str(error)
    sanitized_meta = sanitize_payload(metadata or {})
    try:
        event_input = {"message": sanitize_payload(error_text)}
        emitted = _safe_call(
            target,
            "event",
            name="error",
            level="ERROR",
            input=event_input,
            metadata=sanitized_meta,
        )
        if emitted is None:
            _safe_call(
                target,
                "update",
                status_message=sanitize_payload(error_text),
                metadata=sanitized_meta,
                level="ERROR",
            )
    except Exception:
        logger.exception("Не удалось отправить ошибку в Langfuse.")


def flush_if_available() -> None:
    """Принудительно отправляет буфер observability, если клиент доступен."""
    client = get_langfuse_client()
    if client is None:
        return

    try:
        _safe_call(client, "flush")
    except Exception:
        logger.exception("Не удалось выполнить Langfuse.flush().")
