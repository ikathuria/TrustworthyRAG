import logging

class Logger:
	def __init__(self, name=__name__):
		self.logger = self.get_logger(name)

	@staticmethod
	def get_logger(name):
		logging.basicConfig(
			level=logging.INFO,
			format='%(asctime)s [%(name)s] [%(levelname)s] %(message)s',
			handlers=[
				logging.FileHandler(f'logs/{name}.log'),
				logging.StreamHandler()
			]
		)
		return logging.getLogger(name)

	def log_info(self, message):
		self.logger.info(message)

	def log_error(self, message):
		self.logger.error(message)

	def log_debug(self, message):
		self.logger.debug(message)

	def log_warning(self, message):
		self.logger.warning(message)

if __name__ == "__main__":
	logger = Logger("TEST_Logger")
	logger.log_info("This is an info message")
	logger.log_error("This is an error message")
	logger.log_debug("This is a debug message")
	logger.log_warning("This is a warning message")
