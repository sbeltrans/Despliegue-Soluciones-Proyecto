import logging
import sys
from types import FrameType
from typing import List, cast

from loguru import logger
from pydantic import AnyHttpUrl, BaseSettings

# Nivel del logger
class LoggingSettings(BaseSettings):
    LOGGING_LEVEL: int = logging.INFO

# Configuración de raíz de la ruta, logger, CORS, nombre
class Settings(BaseSettings):
    API_V1_STR: str = "/api/v1"

    # Meta
    logging: LoggingSettings = LoggingSettings()

    # Lista separada por comas de origins
    BACKEND_CORS_ORIGINS: List[AnyHttpUrl] = [
        "http://localhost:3000",
        "http://localhost:8000",
        "https://localhost:3000",
        "https://localhost:8000",
    ]

    PROJECT_NAME: str = "SP500 Stock Recommender API"

    class Config:
        case_sensitive = True

# Intercepción de mensajes de loggers
class InterceptHandler(logging.Handler):
    def emit(self, record: logging.LogRecord) -> None:
        # Obtenemos el nivel de Loguru correspondiente si existe
        try:
            level = logger.level(record.levelname).name
        except ValueError:
            level = str(record.levelno)

        # Encontramos el caller de donde se originó el mensaje
        frame, depth = logging.currentframe(), 2
        while frame.f_code.co_filename == logging.__file__:
            frame = cast(FrameType, frame.f_back)
            depth += 1

        logger.opt(depth=depth, exception=record.exc_info).log(
            level,
            record.getMessage(),
        )

# Configuración de loggers usando uvicorn
def setup_app_logging(config: Settings) -> None:
    LOGGERS = ("uvicorn.asgi", "uvicorn.access")
    logging.getLogger().handlers = [InterceptHandler()]
    for logger_name in LOGGERS:
        logging_logger = logging.getLogger(logger_name)
        logging_logger.handlers = [InterceptHandler(level=config.logging.LOGGING_LEVEL)]

    logger.configure(
        handlers=[{"sink": sys.stderr, "level": config.logging.LOGGING_LEVEL}]
    )


settings = Settings()
