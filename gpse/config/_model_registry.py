"""YAML-driven model registry for GPSE.

Loads model definitions from ``gpse/config/models.yaml`` (with optional user
override), resolves import paths lazily, injects thread parameters, and compiles
inline search-space DSL into Optuna-compatible callables.
"""

from __future__ import annotations

import importlib
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

from loguru import logger

from gpse.utils.configuration import load_models_config


@dataclass
class ModelRegistryEntry:
    """Resolved model definition from the registry YAML."""

    name: str
    task: str
    import_path: str
    thread_strategy: List[str]
    default_params: Dict[str, Any]
    search_space: Optional[List[Dict[str, Any]]] = None
    param_func: Optional[Callable] = None
    extra: Dict[str, Any] = field(default_factory=dict)


def _resolve_placeholders(value: Any, context: Dict[str, Any]) -> Any:
    """Recursively substitute ``{random_seed}``-style placeholders."""
    if isinstance(value, str) and value.startswith("{") and value.endswith("}"):
        key = value[1:-1]
        if key in context:
            return context[key]
        return value
    if isinstance(value, dict):
        return {k: _resolve_placeholders(v, context) for k, v in value.items()}
    if isinstance(value, list):
        return [_resolve_placeholders(v, context) for v in value]
    return value


def _resolve_import_path(import_path: str) -> type:
    """Dynamically import a class from a dotted module path."""
    module_path, _, class_name = import_path.rpartition(".")
    if not module_path:
        raise ImportError(f"Invalid import_path (no module component): {import_path}")
    module = importlib.import_module(module_path)
    cls = getattr(module, class_name, None)
    if cls is None:
        raise ImportError(f"Class '{class_name}' not found in module '{module_path}'")
    return cls


def _inject_threads(params: Dict[str, Any], strategy: List[str], n_threads: int) -> Dict[str, Any]:
    """Inject thread-count parameters according to the model's strategy."""
    for key in strategy:
        params[key] = n_threads
    return params


def _build_inline_param_func(
    search_space: List[Dict[str, Any]], context: Dict[str, Any]
) -> Callable:
    """Compile a YAML search-space DSL into an Optuna param function."""

    def param_func(trial) -> Dict[str, Any]:
        params: Dict[str, Any] = {}
        for spec in search_space:
            name = spec["name"]
            ptype = spec["type"]
            if ptype == "fixed":
                params[name] = _resolve_placeholders(spec["value"], context)
            elif ptype == "int":
                kwargs: Dict[str, Any] = {}
                if spec.get("log"):
                    kwargs["log"] = True
                if spec.get("step") is not None:
                    kwargs["step"] = spec["step"]
                params[name] = trial.suggest_int(name, spec["low"], spec["high"], **kwargs)
            elif ptype == "float":
                kwargs = {}
                if spec.get("log"):
                    kwargs["log"] = True
                params[name] = trial.suggest_float(name, spec["low"], spec["high"], **kwargs)
            elif ptype == "categorical":
                choices = spec["choices"]
                params[name] = trial.suggest_categorical(name, choices)
            else:
                raise ValueError(f"Unknown search-space type '{ptype}' for param '{name}'")
        return params

    return param_func


def _resolve_param_func_ref(
    ref: str, builtin_funcs: Dict[str, Callable]
) -> Callable:
    """Resolve a param_func reference string to a callable.

    Formats:
        "builtin:_xgboost_reg_params"  → lookup in builtin_funcs
        "my.module:my_func"            → dynamic import
    """
    if ref.startswith("builtin:"):
        func_name = ref[len("builtin:"):]
        if func_name not in builtin_funcs:
            raise ValueError(
                f"Builtin param func '{func_name}' not found. "
                f"Available: {sorted(builtin_funcs.keys())}"
            )
        return builtin_funcs[func_name]

    module_path, _, func_name = ref.partition(":")
    if not module_path or not func_name:
        raise ValueError(
            f"Invalid param_func reference '{ref}'. "
            "Expected 'builtin:_name' or 'module.path:func_name'."
        )
    module = importlib.import_module(module_path)
    func = getattr(module, func_name, None)
    if func is None:
        raise ImportError(f"Function '{func_name}' not found in module '{module_path}'")
    return func


def _normalize_thread_strategy(raw: Any) -> List[str]:
    """Normalize thread_strategy to a list of param names."""
    if raw is None or raw == "none":
        return []
    if isinstance(raw, str):
        return [raw]
    if isinstance(raw, list):
        return [s for s in raw if s != "none"]
    return []


class ModelRegistry:
    """YAML-driven model registry with dynamic import and thread injection."""

    def __init__(
        self,
        user_config_path: Optional[str] = None,
        random_seed: Optional[int] = 42,
        n_threads: int = 1,
        n_classes: Optional[int] = None,
        catboost_train_dir: Optional[str] = None,
        builtin_param_funcs: Optional[Dict[str, Callable]] = None,
        task_filter: Optional[str] = None,
    ):
        self.random_seed = random_seed
        self.n_threads = n_threads
        self.n_classes = n_classes
        self.catboost_train_dir = catboost_train_dir
        self._builtin_funcs = builtin_param_funcs or {}
        self._task_filter = task_filter

        self._context: Dict[str, Any] = {
            "random_seed": random_seed,
            "n_threads": n_threads,
            "n_classes": n_classes,
        }

        cfg = load_models_config(user_config_path)
        raw_models = cfg.get("models", [])
        self._entries: Dict[str, ModelRegistryEntry] = {}

        for raw in raw_models:
            task = raw.get("task", "")
            if task_filter and task != task_filter:
                continue
            entry = self._parse_entry(raw)
            self._entries[entry.name] = entry

        logger.debug(
            "ModelRegistry loaded: {} models (task_filter={})",
            len(self._entries),
            task_filter or "all",
        )

    def _parse_entry(self, raw: Dict[str, Any]) -> ModelRegistryEntry:
        """Parse and validate a single model entry from YAML."""
        name = raw["name"]
        task = raw.get("task", "regression")
        import_path = raw["import_path"]
        thread_strategy = _normalize_thread_strategy(raw.get("thread_strategy"))
        default_params = raw.get("default_params", {})
        extra = raw.get("extra", {})

        search_space_raw = raw.get("search_space")
        param_func_ref = raw.get("param_func")

        if search_space_raw and param_func_ref:
            raise ValueError(
                f"Model '{name}': search_space and param_func are mutually exclusive"
            )

        search_space = None
        param_func = None

        if search_space_raw:
            search_space = search_space_raw
            param_func = _build_inline_param_func(search_space_raw, self._context)
        elif param_func_ref:
            param_func = _resolve_param_func_ref(param_func_ref, self._builtin_funcs)
        else:
            raise ValueError(
                f"Model '{name}': must define either search_space or param_func"
            )

        return ModelRegistryEntry(
            name=name,
            task=task,
            import_path=import_path,
            thread_strategy=thread_strategy,
            default_params=default_params,
            search_space=search_space,
            param_func=param_func,
            extra=extra,
        )

    @property
    def model_configs(self) -> Dict[str, ModelRegistryEntry]:
        """Backward-compatible dict interface (keys = model names)."""
        return self._entries

    def get_available_models(self) -> List[str]:
        return list(self._entries.keys())

    def get_param_func(self, model_name: str) -> Callable:
        if model_name not in self._entries:
            raise ValueError(f"Model {model_name} not found in configurations")
        return self._entries[model_name].param_func

    def get_default_params(self, model_name: str) -> Dict[str, Any]:
        if model_name not in self._entries:
            raise ValueError(f"Model {model_name} not found in configurations")
        entry = self._entries[model_name]
        params = _resolve_placeholders(entry.default_params, self._context)

        if entry.extra.get("n_classes_objective"):
            params = self._apply_lightgbm_clf_defaults(params)

        if isinstance(params.get("hidden_layer_sizes"), list):
            params["hidden_layer_sizes"] = tuple(params["hidden_layer_sizes"])

        return params

    def create_model(self, model_name: str, params: Dict[str, Any]) -> Any:
        """Instantiate a model with thread injection and special-case handling."""
        if model_name not in self._entries:
            raise ValueError(f"Model {model_name} not found in configurations")
        entry = self._entries[model_name]
        params = params.copy()

        if isinstance(params.get("hidden_layer_sizes"), list):
            params["hidden_layer_sizes"] = tuple(params["hidden_layer_sizes"])

        params = _inject_threads(params, entry.thread_strategy, self.n_threads)

        if entry.extra.get("catboost_train_dir"):
            if self.catboost_train_dir:
                params.setdefault("train_dir", self.catboost_train_dir)
            else:
                params.setdefault("allow_writing_files", False)

        if entry.extra.get("inject_num_class"):
            if self.n_classes is not None and self.n_classes > 1:
                params["num_class"] = self.n_classes

        if entry.extra.get("n_classes_objective"):
            if self.n_classes is not None and self.n_classes > 2:
                params.setdefault("num_class", self.n_classes)
            else:
                params.pop("num_class", None)
                params.setdefault("objective", "binary")
                params.setdefault("metric", "binary_logloss")
                params.setdefault("is_unbalance", True)

        cls = _resolve_import_path(entry.import_path)
        return cls(**params)

    def filter_model_params(self, model_name: str, params: Dict[str, Any]) -> Dict[str, Any]:
        """Strip auxiliary params that are not valid model constructor args."""
        filtered = {k: v for k, v in params.items() if not k.startswith("_")}

        if model_name not in self._entries:
            return filtered

        entry = self._entries[model_name]
        if entry.extra.get("filter_n_units"):
            filtered.pop("n_layers", None)
            filtered = {k: v for k, v in filtered.items() if not k.startswith("n_units_l")}

        return filtered

    def _apply_lightgbm_clf_defaults(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Adjust LightGBM classifier defaults based on n_classes."""
        params = params.copy()
        if self.n_classes == 2:
            params["objective"] = "binary"
            params["metric"] = "binary_logloss"
            params["is_unbalance"] = True
            params.pop("num_class", None)
        else:
            params.setdefault("objective", "multiclass")
            params.setdefault("metric", "multi_logloss")
            if self.n_classes is not None:
                params["num_class"] = self.n_classes
        return params
