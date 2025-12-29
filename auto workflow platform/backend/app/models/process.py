# individual process steps
from sqlalchemy import Column, Integer, String, Float, DateTime, ForeignKey, JSON, Text
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func
from datetime import datetime
from .base import Base

class Process(Base):
    __tablename__ = "processes"

    id = Column(Integer, primary_key=True, autoincrement=True)
    name = Column(String(255), nullable=False)
    description = Column(Text, nullable=True)
    created_at = Column(DateTime, default=func.now())
    updated_at = Column(DateTime, default=func.now(), onupdate=func.now())

    steps = relationship("ProcessStep", back_populates="process")
    
class ProcessStep(Base):
    __tablename__ = "process_steps"

    id = Column(Integer, primary_key=True, autoincrement=True)
    process_id = Column(Integer, ForeignKey("processes.id"), nullable=False)
    name = Column(String(255), nullable=False)
    description = Column(Text, nullable=True)
    order = Column(Integer, nullable=False)
    created_at = Column(DateTime, default=func.now())
    updated_at = Column(DateTime, default=func.now(), onupdate=func.now())

    process = relationship("Process", back_populates="steps")