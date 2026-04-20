from sqlalchemy import Column, Integer, String, Float, Boolean, ForeignKey, DateTime, JSON
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func
from app.db.database import Base

class Patient(Base):
    __tablename__ = "patients"

    id = Column(Integer, primary_key=True, index=True)
    name = Column(String, index=True, nullable=False)
    age = Column(Integer, nullable=False)
    insurance_provider = Column(String, index=True, nullable=False)
    
    claims = relationship("Claim", back_populates="patient")

class Claim(Base):
    __tablename__ = "claims"

    id = Column(Integer, primary_key=True, index=True)
    patient_id = Column(Integer, ForeignKey("patients.id"), nullable=False)
    diagnosis = Column(String, nullable=False) # Extracted raw diagnosis
    procedure = Column(String, nullable=False) # Extracted raw procedure
    icd_code = Column(String, nullable=True)   # Mapped ICD-10 Code
    cpt_code = Column(String, nullable=True)   # Mapped CPT Code
    status = Column(String, default="draft", index=True) # draft, validated, corrected, submitted
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    
    patient = relationship("Patient", back_populates="claims")
    denial_predictions = relationship("Denial", back_populates="claim")
    agent_logs = relationship("AgentLog", back_populates="claim")

class Denial(Base):
    __tablename__ = "denials"

    id = Column(Integer, primary_key=True, index=True)
    claim_id = Column(Integer, ForeignKey("claims.id"), nullable=False)
    reason = Column(String, nullable=False)
    probability = Column(Float, nullable=False)
    corrected = Column(Boolean, default=False)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    
    claim = relationship("Claim", back_populates="denial_predictions")

class AgentLog(Base):
    """
    Key Differentiator: Audit trail of multi-agent interactions.
    """
    __tablename__ = "agent_logs"

    id = Column(Integer, primary_key=True, index=True)
    claim_id = Column(Integer, ForeignKey("claims.id"), nullable=False)
    agent_name = Column(String, index=True, nullable=False)
    input_data = Column(JSON, nullable=False)
    output_data = Column(JSON, nullable=False)
    timestamp = Column(DateTime(timezone=True), server_default=func.now())
    
    claim = relationship("Claim", back_populates="agent_logs")
