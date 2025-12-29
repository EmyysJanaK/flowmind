from sqlalchemy import Column, Integer, String, DateTime, JSON, Float, Boolean, ForeignKey, Text
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func
from datetime import datetime

Base = declarative_base()

class Workflow(Base):
    __tablename__ = "workflows"

    id = Column(Integer, primary_key=True, index=True)
    name = Column(String(255), nullable=False)
    description = Column(Text)
    definition = Column(JSON)  # Workflow structure
    status = Column(String(50), default="active")
    created_at = Column(DateTime, default=func.now())
    updated_at = Column(DateTime, default=func.now(), onupdate=func.now())
    
    # Performance metrics
    avg_duration = Column(Float)
    success_rate = Column(Float)
    optimization_score = Column(Float)
    
    # Relationships
    executions = relationship("WorkflowExecution", back_populates="workflow")
    optimizations = relationship("WorkflowOptimization", back_populates="workflow")

class WorkflowExecution(Base):
    __tablename__ = "workflow_executions"

    id = Column(Integer, primary_key=True, index=True)
    workflow_id = Column(Integer, ForeignKey("workflows.id"))
    status = Column(String(50))  # running, completed, failed
    start_time = Column(DateTime)
    end_time = Column(DateTime)
    duration = Column(Float)
    input_data = Column(JSON)
    output_data = Column(JSON)
    execution_path = Column(JSON)  # Track which steps were executed
    
    # AI-generated insights
    bottlenecks = Column(JSON)
    optimization_opportunities = Column(JSON)
    
    # Relationships
    workflow = relationship("Workflow", back_populates="executions")
    steps = relationship("ExecutionStep", back_populates="execution")

class ExecutionStep(Base):
    __tablename__ = "execution_steps"

    id = Column(Integer, primary_key=True, index=True)
    execution_id = Column(Integer, ForeignKey("workflow_executions.id"))
    step_name = Column(String(255))
    step_type = Column(String(100))
    status = Column(String(50))
    start_time = Column(DateTime)
    end_time = Column(DateTime)
    duration = Column(Float)
    input_data = Column(JSON)
    output_data = Column(JSON)
    error_message = Column(Text)
    
    # AI metrics
    performance_score = Column(Float)
    anomaly_score = Column(Float)
    
    # Relationships
    execution = relationship("WorkflowExecution", back_populates="steps")

class WorkflowOptimization(Base):
    __tablename__ = "workflow_optimizations"

    id = Column(Integer, primary_key=True, index=True)
    workflow_id = Column(Integer, ForeignKey("workflows.id"))
    optimization_type = Column(String(100))
    suggestion = Column(Text)
    impact_score = Column(Float)
    confidence_score = Column(Float)
    status = Column(String(50))  # pending, applied, rejected
    created_at = Column(DateTime, default=func.now())
    applied_at = Column(DateTime)
    
    # Performance impact
    expected_improvement = Column(Float)
    actual_improvement = Column(Float)
    
    # Relationships
    workflow = relationship("Workflow", back_populates="optimizations")