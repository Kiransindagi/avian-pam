from src.models.base_model import BaseAvianModel
from src.models.baselines import (
    DummyMeanPredictor,
    DummyMedianPredictor,
    ElasticNetModel,
    LassoRegressionModel,
    LinearRegressionModel,
    PoissonRegressionModel,
    RidgeRegressionModel,
)
from src.models.ensembles import StackingEnsembleModel, VotingEnsembleModel
from src.models.kernel_models import KNNRegressorModel, SVRModel
from src.models.model_registry import (
    ModelRegistryManager,
    get_model,
    list_registered_models,
    register_model,
)
from src.models.trees import (
    CatBoostModel,
    DecisionTreeModel,
    ExtraTreesModel,
    GradientBoostingModel,
    HistGradientBoostingModel,
    LightGBMModel,
    RandomForestModel,
    XGBoostModel,
)

__all__ = [
    "BaseAvianModel",
    "register_model",
    "get_model",
    "list_registered_models",
    "DummyMeanPredictor",
    "DummyMedianPredictor",
    "LinearRegressionModel",
    "RidgeRegressionModel",
    "LassoRegressionModel",
    "ElasticNetModel",
    "PoissonRegressionModel",
    "DecisionTreeModel",
    "RandomForestModel",
    "ExtraTreesModel",
    "GradientBoostingModel",
    "HistGradientBoostingModel",
    "XGBoostModel",
    "LightGBMModel",
    "CatBoostModel",
    "SVRModel",
    "KNNRegressorModel",
    "VotingEnsembleModel",
    "StackingEnsembleModel",
]
