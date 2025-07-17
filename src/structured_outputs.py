from pydantic import BaseModel, model_validator
from typing import Optional
from typing_extensions import Self


class Experiment(BaseModel):
    """
    Represents a single experiment with a title, objective, steps, and deliverables.

    Attributes:
        title (str): Title of the experiment
        objective (str): The main goal or objective of the experiment
        steps (str): List of steps to be followed to implement the experiment
        deliverables (str): List of expected outcomes or deliverables from the experiment
    """
    title: str
    objective: str
    steps: str
    deliverables: str


class ExperimentList(BaseModel):
    """
    A collection of experiments.

    Attributes:
        experiments (list[Experiment]): List of Experiment objects
    """
    experiments: list[Experiment]


class ExperimentCode(BaseModel):
    """
    Contains the code implementation for an experiment.

    Attributes:
        code (str): The actual code to be executed for the experiment
    """
    code: str


class ProgramCritique(BaseModel):
    """
    Feedback on experiment code implementation.

    Attributes:
        fixes (list[str]): List of suggested fixes or improvements for the code
    """
    fixes: list[str]


class ExperimentAnalyst(BaseModel):
    """
    Analysis of experiment results.

    Attributes:
        error (bool): Whether the experiment failed
        analysis (Optional[str]): Detailed analysis of the experiment outcomes
    """
    error: bool
    analysis: str

    @model_validator(mode='after')
    def error_required_on_failure(self) -> Self:
        if not self.error and self.analysis is None:
            raise ValueError('analysis is required when error is False')
        return self


class ExperimentReviewer(BaseModel):
    """
    Review of an experiment's execution and results.

    Attributes:
        success (bool): Whether the experiment was successful
        feedback (str | None): Required feedback when experiment fails, optional otherwise

    Raises:
        ValueError: If success is False and no feedback is provided
    """
    success: bool
    feedback: str

    @model_validator(mode='after')
    def feedback_required_on_failure(self) -> Self:
        if not self.success and self.feedback is None:
            raise ValueError('feedback is required when success is False')
        return self


class Relationship(BaseModel):
    """
    Represents a relationship between two variables in a hypothesis.

    Attributes:
        explanatory (str): The independent/explanatory variable in the relationship
        response (str): The dependent/response variable in the relationship 
        relationship (str): Description of how the explanatory variable affects the response variable
    """
    explanatory: str
    response: str
    relationship: str


class HypothesisDimensions(BaseModel):
    """
    Structured representation of the key dimensions of a hypothesis.

    Attributes:
        contexts (list[str]): List of boundary conditions and assumptions under which the hypothesis holds
        variables (list[str]): List of key concepts/variables involved in the hypothesis
        relationships (list[Relationship]): List of causal relationships between pairs of variables
    """
    contexts: list[str]
    variables: list[str]
    relationships: list[Relationship]


class Hypothesis(BaseModel):
    """
    A proposed explanation for a phenomenon or a prediction about the outcome of an experiment.

    Attributes:
        hypothesis (str): The hypothesis statement
        dimensions (HypothesisDimensions): Structured hypothesis dimensions
    """
    hypothesis: str
    dimensions: HypothesisDimensions


class ImageAnalysis(BaseModel):
    """
    Structured representation of plot axes and related analysis information.

    Attributes:
        title (str): The title of the plot
        x_axis_label (str): Label for the x-axis
        y_axis_label (str): Label for the y-axis
        x_axis_range (list[int | float]): Range of values on the x-axis
        y_axis_range (list[int | float]): Range of values on the y-axis
        data_trends (list[str]): List of observed trends in the data
        statistical_insights (list[str]): List of statistical observations and metrics
        annotations_and_legends (list[str]): List of plot annotations and legend descriptions
    """
    plot_type: str
    title: str
    x_axis_label: str
    y_axis_label: str
    x_axis_range: list[int] | list[float]
    y_axis_range: list[int] | list[float]
    data_trends: list[str]
    statistical_insights: list[str]
    annotations_and_legends: list[str]


class ExecutionResult(BaseModel):
    exit_code: int
    result: str
