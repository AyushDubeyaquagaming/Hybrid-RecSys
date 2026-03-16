from pipeline.config import PipelineSettings
from pipeline.flow import training_flow
from pipeline.logging_utils import configure_logging

if __name__ == "__main__":
    configure_logging(PipelineSettings().LOG_LEVEL)
    training_flow()
