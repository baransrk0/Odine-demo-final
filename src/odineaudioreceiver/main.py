from dotenv import load_dotenv

try:
    load_dotenv(".env")
except FileNotFoundError:
    raise FileNotFoundError(".env hasn't been created")
except OSError:
    raise OSError("access denied for .env")
except ValueError:
    raise ValueError(".env file has an invalid format")

from pathlib import Path
import asyncio
import logging

from odineaudioreceiver.config import Settings
from odineaudioreceiver.audio.gpio import ListeningIndicator
from odineaudioreceiver.audio.storage import AudioStorage
from odineaudioreceiver.audio.rf_service import RFInputService
from odineaudioreceiver._bridge import deliver_audio_to_apa

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


async def run_recorder():
    settings = Settings()
    base_rec_dir = Path(settings.base_directory)
    base_rec_dir.mkdir(parents=True, exist_ok=True)

    indicator = ListeningIndicator(settings)
    storage = AudioStorage(base_rec_dir, settings.audio_retention_seconds)

    rf_svc = RFInputService(
        settings=settings,
        storage=storage,
        submitter=deliver_audio_to_apa,  # Run once the recording has finished.
        indicator=indicator,
    )

    try:
        rf_svc.start()
    except asyncio.CancelledError:
        logger.info("Shutting down recorder...")
    finally:
        await rf_svc.stop()


if __name__ == "__main__":
    try:
        asyncio.run(run_recorder())
    except KeyboardInterrupt:
        logger.info("Recorder stopped manually.")
