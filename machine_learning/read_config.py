import os
import yaml
import logging
from pathlib import Path


class Config:
    """Class that loads hyper parameters from a yaml file.
    Example:
    ```
    params = Params(config_file_path)
    print(params.learning_rate)
    params.learning_rate = 0.5  # change the value of learning_rate in params
    ```
    """

    def __init__(self, config_file_path):
        self.update(config_file_path)
        self.base_dir = os.path.dirname(os.path.abspath(config_file_path))
        self._additional_parameter()

        # Initially it will be false
        # In the hyper parameter searching file it will be set to True
        self.is_hyper_parameter_searching = False
        # This file is in the root directory of the project.

        # Delete .DS_Store file from all the directory
        # so that tensorflow data pipeline won't have to deal with that
        command = "find " + self.base_dir + " -name '*.DS_Store' -type f -delete"
        os.system(command)
        self._make_absolute_directory()
        # set logger
        self.logger = self.get_logger()
        self.verbose = 1 if self.log_level == 'info' else 0
        self.logger.info("Finish reading the configuration file")

    def _additional_parameter(self):
        pass

    def _make_absolute_directory(self):
        # prepend this base directory with other parameter so that we won't get any error for the path
        # As those directory will be accessed from different file. which are in different location
        self.dataset_dir = os.path.join(self.base_dir, self.dataset_dir)
        self.train_data = os.path.join(self.dataset_dir, self.train_data)
        self.valid_data = os.path.join(self.dataset_dir, self.valid_data)

        self.output_dir = os.path.join(self.base_dir, self.output_dir)

        self.checkpoint_dir = os.path.join(self.output_dir, "checkpoint")
        self.log_dir = os.path.join(self.output_dir, "logs")

    def save(self, json_path):
        """Saves parameters to json file"""
        with open(json_path, 'w') as f:
            yaml.dump(self.__dict__, f, indent=4)

    def update(self, config_file_path):
        with open(config_file_path) as f:
            params = yaml.load(f, Loader=yaml.FullLoader)
            self.__dict__.update(params)

    @property
    def dict(self):
        """Gives dict-like access to Params instance by `params.dict['learning_rate']`"""
        return self.__dict__

    def get_logger(self):

        logger = logging.getLogger(__name__)
        # we will either write the log information to file or console
        # Usually we don't need to log in both location
        if self.log_to == "file":
            # If log file does not exist then create one
            log_path = Path(self.log_dir) / "logging.log"
            log_path.parent.mkdir(parents=True, exist_ok=True)
            if not log_path.exists():
                with open(log_path, "w") as file:
                    logger.info("Log file created")
            handler = logging.FileHandler(log_path)
            handler.setFormatter(logging.Formatter(fmt='%(asctime)s %(levelname)-8s %(message)s',
                                                   datefmt='%Y-%m-%d %H:%M:%S'))
        else:
            handler = logging.StreamHandler()
            handler.setFormatter(logging.Formatter("%(levelname)s: %(message)s"))
        # set the log level
        if self.log_level == 'info':
            logger.setLevel(logging.INFO)
        elif self.log_level == 'debug':
            logger.setLevel(logging.DEBUG)
        elif self.log_level == 'warning':
            logger.setLevel(logging.WARNING)
        else:
            logger.setLevel(logging.ERROR)

        # remove the default handler
        logger.handlers = []
        # get the tensorflow logger
        logger.addHandler(handler)
        return logger