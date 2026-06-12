import uuid

from sqlalchemy import Column, String
from sqlalchemy.dialects.postgresql import UUID

from database.connection import Base


class Dataset(Base):
    __tablename__ = "datasets"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    name = Column(String, nullable=False, unique=True)
    version = Column(String, nullable=False)
    description = Column(String, nullable=True)

    def __repr__(self):
        return f"<Dataset(name='{self.name}', version='{self.version}')>"
