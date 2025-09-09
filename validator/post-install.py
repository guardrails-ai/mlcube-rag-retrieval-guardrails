from dotenv import load_dotenv
import logging

logger = logging.getLogger(__name__)
logger.info("post-install starting...")
logger.info("Loading environment variables...")
load_dotenv()
logger.info("post-install finished.")
