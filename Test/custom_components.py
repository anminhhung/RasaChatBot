# custom_components.py

from typing import Any, Dict, List, Text
from transformers import AutoModel, AutoTokenizer
import numpy as np
import torch
import logging
from rasa.engine.graph import GraphComponent, ExecutionContext
from rasa.engine.storage.resource import Resource
from rasa.engine.storage.storage import ModelStorage
from rasa.shared.nlu.constants import TEXT
from rasa.shared.nlu.training_data.message import Message
from rasa.shared.nlu.training_data.training_data import TrainingData
from rasa.engine.recipes.default_recipe import DefaultV1Recipe

logger = logging.getLogger(__name__)

@DefaultV1Recipe.register(DefaultV1Recipe.ComponentType.MESSAGE_FEATURIZER, is_trainable=False)
class BARTPhoFeaturizer(GraphComponent):
    """A custom feature extractor using BARTPho model."""

    @staticmethod
    def get_default_config() -> Dict[Text, Any]:
        return {}

    def __init__(self, config: Dict[Text, Any]) -> None:
        self.config = config
        # print("CONFIG: ", config)
        self.tokenizer = AutoTokenizer.from_pretrained("vinai/bartpho-word")
        self.model = AutoModel.from_pretrained("vinai/bartpho-word")

    @classmethod
    def create(
        cls,
        config: Dict[Text, Any],
        model_storage: ModelStorage,
        resource: Resource,
        execution_context: ExecutionContext,
    ) -> GraphComponent:
        return cls(config)

    def process_training_data(self, training_data: TrainingData) -> TrainingData:
        """Process the training data and add features to it."""
        print("######### train data ########:", training_data.training_examples)
        for message in training_data.training_examples:
            print("######MESSAGE#######:", message)
            self._add_features_to_message(message)
        return training_data

    def process(self, messages: List[Message]) -> List[Message]:
        for message in messages:
            self._add_features_to_message(message)
        return messages

    def _add_features_to_message(self, message: Message) -> None:
        text = message.get(TEXT)
        # if not text:
        #     logger.warning(f"Message does not have a text attribute: {message}")
        #     return

        # Make sure the text is in the correct format for the tokenizer
        inputs = self.tokenizer(text=text, return_tensors="pt")
        inputs.pop('token_type_ids', None)
        with torch.no_grad():
            outputs = self.model(**inputs)
        features = outputs.last_hidden_state.mean(dim=1).cpu().numpy()
        message.set("custom_features", features)

    def persist(self, storage: ModelStorage, resource: Resource) -> None:
        """Persist this component."""
        pass  # This component does not persist any data

    @classmethod
    def load(
        cls,
        config: Dict[Text, Any],
        model_storage: ModelStorage,
        resource: Resource,
        execution_context: ExecutionContext,
    ) -> GraphComponent:
        """Load this component."""
        return cls(config)
