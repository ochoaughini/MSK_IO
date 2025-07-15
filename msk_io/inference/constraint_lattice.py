import os
import json
from typing import Dict, Any, List, Optional
from msk_io.errors import ProcessingError, ConfigurationError
from msk_io.utils.log_config import get_logger
from msk_io.utils.decorators import handle_errors, log_method_entry_exit
from msk_io.schema._pydantic_base import MSKIOBaseModel

logger = get_logger(__name__)

class ConstraintLattice:
    def __init__(self, config, rules_file_path: Optional[str] = None):
        self.config = config
        self.rules: Dict[str, Any] = {}
        self.rules_file_path = rules_file_path
        self._load_rules()
        logger.info(f"Constraint Lattice initialized. Rules loaded from: {rules_file_path or 'dummy rules'}")

    @handle_errors
    @log_method_entry_exit
    def _load_rules(self) -> None:
        if self.rules_file_path:
            if not os.path.exists(self.rules_file_path):
                raise ConfigurationError(f"Constraint rules file not found: {self.rules_file_path}")
            try:
                with open(self.rules_file_path, 'r', encoding='utf-8') as f:
                    self.rules = json.load(f)
                logger.info(f"Loaded {len(self.rules)} rules from {self.rules_file_path}.")
            except json.JSONDecodeError as e:
                raise ConfigurationError(f"Invalid JSON in rules file {self.rules_file_path}: {e}") from e
            except Exception as e:
                raise ConfigurationError(f"Failed to load rules from {self.rules_file_path}: {e}") from e
        else:
            logger.warning("No specific rules file provided. Using dummy constraint rules.")
            self.rules = {
                "dummy_rule_1": {
                    "description": "Detect large simulated lesions",
                    "conditions": [
                        {"data_type": "image_analysis", "field": "lesion_volume_mm3", "operator": ">", "value": 500000}
                    ],
                    "outcome": {"recommendation": "Consider biopsy for large lesion.", "severity_increase": "HIGH"}
                },
                "dummy_rule_2": {
                    "description": "Confirm LLM finding with image evidence",
                    "conditions": [
                        {"data_type": "llm_analysis", "field": "extracted_findings.category", "operator": "contains", "value": "Anomaly"},
                        {"data_type": "image_analysis", "field": "regions_of_interest", "operator": "not_empty"}
                    ],
                    "outcome": {"confidence_boost": 0.1, "message": "LLM finding corroborated by image ROI."}
                }
            }

    @handle_errors
    @log_method_entry_exit
    def evaluate_constraints(self, processed_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        logger.info("Evaluating constraints against processed data.")
        triggered_rules = []
        for rule_id, rule_def in self.rules.items():
            conditions_met = True
            for condition in rule_def.get("conditions", []):
                data_type = condition.get("data_type")
                field = condition.get("field")
                operator = condition.get("operator")
                value = condition.get("value")
                if data_type not in processed_data:
                    logger.debug(f"Rule '{rule_id}': Missing data type '{data_type}'. Condition not met.")
                    conditions_met = False
                    break
                current_data = processed_data[data_type]
                field_path = field.split('.')
                field_value = current_data
                for path_part in field_path:
                    if isinstance(field_value, dict) and path_part in field_value:
                        field_value = field_value[path_part]
                    elif isinstance(field_value, list):
                        found = False
                        for item in field_value:
                            if isinstance(item, dict) and path_part in item and (operator == "contains" and value in item[path_part]):
                                found = True
                                break
                            elif isinstance(item, MSKIOBaseModel) and hasattr(item, path_part) and (operator == "contains" and value in getattr(item, path_part)):
                                found = True
                                break
                        if found:
                            field_value = True
                        else:
                            field_value = False
                            break
                    elif isinstance(field_value, MSKIOBaseModel) and hasattr(field_value, path_part):
                        field_value = getattr(field_value, path_part)
                    else:
                        logger.debug(f"Rule '{rule_id}': Field '{field}' not found in {data_type}.")
                        conditions_met = False
                        break
                if not conditions_met:
                    break
                if operator == ">":
                    if not (isinstance(field_value, (int, float)) and field_value > value):
                        conditions_met = False
                elif operator == "<":
                    if not (isinstance(field_value, (int, float)) and field_value < value):
                        conditions_met = False
                elif operator == "==":
                    if not (field_value == value):
                        conditions_met = False
                elif operator == "contains":
                    if not (isinstance(field_value, str) and value in field_value) and not (isinstance(field_value, list) and value in field_value):
                        conditions_met = False
                elif operator == "not_empty":
                    if not (field_value is not None and len(field_value) > 0):
                        conditions_met = False
                else:
                    logger.warning(f"Rule '{rule_id}': Unknown operator '{operator}'. Skipping condition.")
                    conditions_met = False
                if not conditions_met:
                    break
            if conditions_met:
                logger.info(f"Rule '{rule_id}' triggered. Outcome: {rule_def.get('outcome')}")
                triggered_rules.append({"rule_id": rule_id, "outcome": rule_def.get("outcome", {})})
        return triggered_rules
