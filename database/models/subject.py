import uuid

from sqlalchemy import Boolean, Column, ForeignKey, Integer, String
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import relationship

from database.connection import Base


class Subject(Base):
    __tablename__ = "subjects"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    dataset_id = Column(UUID(as_uuid=True), ForeignKey("datasets.id"), nullable=False)
    source_subject_id = Column(String, nullable=False)
    age = Column(Integer, nullable=True)
    gender = Column(String, nullable=True)
    respiratory_condition = Column(String, nullable=True)
    has_fever = Column(Boolean, nullable=True)
    is_smoker = Column(Boolean, nullable=True)

    # Relationships
    dataset = relationship("Dataset", backref="subjects")
    recordings = relationship("Recording", back_populates="subject")

    def __repr__(self):
        return f"<Subject(source_id='{self.source_subject_id}', age={self.age}, gender='{self.gender}')>"
