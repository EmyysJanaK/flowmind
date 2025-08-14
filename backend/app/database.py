#db engine and session management for MySQL
from sqlalchemy import create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import QueuePool
from sqlalchemy import MetaData
from app.config import settings
import os

# Use MySQL connection URL
DATABASE_URL = settings.mysql_url

# Create engine with MySQL optimizations
engine = create_engine(
    DATABASE_URL, 
    poolclass=QueuePool,
    pool_size=10,
    max_overflow=20,
    pool_pre_ping=True,
    pool_recycle=3600,
    echo=settings.DEBUG  # Enable SQL logging in debug mode
)

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()
