import logging

# Step 1: Define the logging format
logging_format_str = '%(asctime)s - %(levelname)s - %(message)s'

# Step 2: Configure logging
logging.basicConfig(level=logging.INFO, format=logging_format_str)

# Step 3: Create a logger object
confnavigator_logger = logging.getLogger(__name__)

# Step 4: Optional - Add handlers (e.g., FileHandler)
file_handler = logging.FileHandler('confnavigator_logfile.log')
log_file_format = logging.Formatter(logging_format_str)

file_handler.setFormatter(log_file_format)

confnavigator_logger.addHandler(file_handler)


