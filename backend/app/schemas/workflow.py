from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any
from datetime import datetime

class WorkflowBase(BaseModel):
    name: str = Field(..., max_length=255, description="Name of the workflow")
    description: Optional[str] = Field(None, description="Description of the workflow")
    definition: Dict[str, Any] = Field(..., description="Workflow structure in JSON format")
    status: str = Field("active", max_length=50, description="Status of the workflow (active, inactive)")
    created_at: datetime = Field(default_factory=datetime.utcnow, description="Creation timestamp")
    updated_at: datetime = Field(default_factory=datetime.utcnow, description="Last update timestamp")

class WorkflowCreate(WorkflowBase):
    """Schema for creating a new workflow."""
    name: str = Field(..., max_length=255, description="Name of the workflow")
    description: Optional[str] = Field(None, description="Description of the workflow")
    definition: Dict[str, Any] = Field(..., description="Workflow structure in JSON format")
    status: str = Field("active", max_length=50, description="Status of the workflow (active, inactive)")

class WorkflowUpdate(WorkflowBase):
    """Schema for updating an existing workflow."""
    name: Optional[str] = Field(None, max_length=255, description="Updated name of the workflow")
    description: Optional[str] = Field(None, description="Updated description of the workflow")
    status: Optional[str] = Field(None, max_length=50, description="Updated status of the workflow")    

class Workflow(WorkflowBase):
    """Schema for a workflow with additional fields."""
    id: int = Field(..., description="Unique identifier for the workflow")
    avg_duration: Optional[float] = Field(None, description="Average duration of workflow executions")
    success_rate: Optional[float] = Field(None, description="Success rate of workflow executions")
    optimization_score: Optional[float] = Field(None, description="Score indicating optimization potential")

    class Config:
        orm_mode = True 

class WorkflowExecutionBase(BaseModel):
    workflow_id: int = Field(..., description="ID of the associated workflow")
    status: str = Field(..., max_length=50, description="Execution status (running, completed, failed)")
    start_time: datetime = Field(default_factory=datetime.utcnow, description="Execution start time")
    end_time: Optional[datetime] = Field(None, description="Execution end time")
    duration: Optional[float] = Field(None, description="Duration of the execution in seconds")
    input_data: Dict[str, Any] = Field(..., description="Input data for the execution")
    output_data: Optional[Dict[str, Any]] = Field(None, description="Output data from the execution")

class WorkflowExecutionCreate(WorkflowExecutionBase):
    """Schema for creating a new workflow execution."""
    workflow_id: int = Field(..., description="ID of the associated workflow")
    status: str = Field(..., max_length=50, description="Execution status (running, completed, failed)")
    input_data: Dict[str, Any] = Field(..., description="Input data for the execution")

class WorkflowExecutionUpdate(WorkflowExecutionBase):
    """Schema for updating an existing workflow execution."""
    status: Optional[str] = Field(None, max_length=50, description="Updated execution status")
    end_time: Optional[datetime] = Field(None, description="Updated execution end time")
    duration: Optional[float] = Field(None, description="Updated duration of the execution in seconds")
    output_data: Optional[Dict[str, Any]] = Field(None, description="Updated output data from the execution")
class WorkflowExecution(WorkflowExecutionBase):
    """Schema for a workflow execution with additional fields."""
    id: int = Field(..., description="Unique identifier for the workflow execution")
    execution_path: Optional[Dict[str, Any]] = Field(None, description="Path of executed steps in the workflow")
    bottlenecks: Optional[Dict[str, Any]] = Field(None, description="Identified bottlenecks during execution")
    optimization_opportunities: Optional[Dict[str, Any]] = Field(None, description="Opportunities for optimization identified by AI")

    class Config:
        orm_mode = True

class ExecutionStepBase(BaseModel):
    execution_id: int = Field(..., description="ID of the associated workflow execution")
    step_name: str = Field(..., max_length=255, description="Name of the execution step")
    step_type: str = Field(..., max_length=100, description="Type of the execution step")
    status: str = Field(..., max_length=50, description="Status of the step (running, completed, failed)")
    start_time: datetime = Field(default_factory=datetime.utcnow, description="Step start time")
    end_time: Optional[datetime] = Field(None, description="Step end time")
    duration: Optional[float] = Field(None, description="Duration of the step in seconds")
    input_data: Dict[str, Any] = Field(..., description="Input data for the step")
    output_data: Optional[Dict[str, Any]] = Field(None, description="Output data from the step")

class ExecutionStepCreate(ExecutionStepBase):
    """Schema for creating a new execution step."""
    execution_id: int = Field(..., description="ID of the associated workflow execution")
    step_name: str = Field(..., max_length=255, description="Name of the execution step")
    step_type: str = Field(..., max_length=100, description="Type of the execution step")
    status: str = Field(..., max_length=50, description="Status of the step (running, completed, failed)")
    input_data: Dict[str, Any] = Field(..., description="Input data for the step")

class ExecutionStepUpdate(ExecutionStepBase):
    """Schema for updating an existing execution step."""
    status: Optional[str] = Field(None, max_length=50, description="Updated status of the step")
    end_time: Optional[datetime] = Field(None, description="Updated step end time")
    duration: Optional[float] = Field(None, description="Updated duration of the step in seconds")
    output_data: Optional[Dict[str, Any]] = Field(None, description="Updated output data from the step")

class ExecutionStep(ExecutionStepBase):
    """Schema for an execution step with additional fields."""
    id: int = Field(..., description="Unique identifier for the execution step")

    class Config:
        orm_mode = True 

class WorkflowOptimizationBase(BaseModel):
    workflow_id: int = Field(..., description="ID of the associated workflow")
    optimization_type: str = Field(..., max_length=100, description="Type of optimization (e.g., performance, cost)")
    details: Dict[str, Any] = Field(..., description="Details of the optimization")
    created_at: datetime = Field(default_factory=datetime.utcnow, description="Creation timestamp")

class WorkflowOptimizationCreate(WorkflowOptimizationBase):
    """Schema for creating a new workflow optimization."""
    workflow_id: int = Field(..., description="ID of the associated workflow")
    optimization_type: str = Field(..., max_length=100, description="Type of optimization (e.g., performance, cost)")
    details: Dict[str, Any] = Field(..., description="Details of the optimization")
class WorkflowOptimizationUpdate(WorkflowOptimizationBase):
    """Schema for updating an existing workflow optimization."""
    optimization_type: Optional[str] = Field(None, max_length=100, description="Updated type of optimization")
    details: Optional[Dict[str, Any]] = Field(None, description="Updated details of the optimization")

class WorkflowOptimization(WorkflowOptimizationBase):
    """Schema for a workflow optimization with additional fields."""
    id: int = Field(..., description="Unique identifier for the workflow optimization")
    score: Optional[float] = Field(None, description="Score indicating the effectiveness of the optimization")

    class Config:
        orm_mode = True
class WorkflowExecutionWithSteps(WorkflowExecution):
    """Schema for a workflow execution including its steps."""
    steps: List[ExecutionStep] = Field(..., description="List of execution steps associated with this execution")

    class Config:
        orm_mode = True
        use_enum_values = True
        arbitrary_types_allowed = True
class WorkflowWithExecutions(Workflow):
    """Schema for a workflow including its executions."""
    executions: List[WorkflowExecutionWithSteps] = Field(..., description="List of executions associated with this workflow")

    class Config:
        orm_mode = True
        use_enum_values = True
        arbitrary_types_allowed = True
class WorkflowOptimizationWithWorkflow(WorkflowOptimization):
    """Schema for a workflow optimization including its associated workflow."""
    workflow: Workflow = Field(..., description="Associated workflow for this optimization")

    class Config:
        orm_mode = True
        use_enum_values = True
        arbitrary_types_allowed = True
class WorkflowExecutionWithOptimizations(WorkflowExecution):
    """Schema for a workflow execution including its optimizations."""
    optimizations: List[WorkflowOptimizationWithWorkflow] = Field(..., description="List of optimizations associated with this execution")

    class Config:
        orm_mode = True
        use_enum_values = True
        arbitrary_types_allowed = True
class WorkflowWithOptimizations(Workflow):
    """Schema for a workflow including its optimizations."""
    optimizations: List[WorkflowOptimizationWithWorkflow] = Field(..., description="List of optimizations associated with this workflow")

    class Config:
        orm_mode = True
        use_enum_values = True
        arbitrary_types_allowed = True