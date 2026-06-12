import uuid

from sqlalchemy import Boolean, Column, Float, ForeignKey, String
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import relationship

from database.connection import Base


class Recording(Base):
    __tablename__ = "recordings"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    subject_id = Column(UUID(as_uuid=True), ForeignKey("subjects.id"), nullable=False)
    file_path = Column(String, nullable=False)
    duration = Column(Float, nullable=True)
    equipment = Column(String, nullable=True)
    is_cough = Column(Boolean, nullable=True)

    # Relationships
    subject = relationship("Subject", back_populates="recordings")

    def __repr__(self):
        return f"<Recording(file='{self.file_path}', duration={self.duration}, is_cough={self.is_cough})>"
