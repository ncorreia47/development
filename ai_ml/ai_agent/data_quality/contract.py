from datetime import datetime
from typing import Tuple
from pydantic import BaseModel, EmailStr


class Insurance(BaseModel):
    username: EmailStr
    insurance_plan: str
    initial_date: datetime
    final_date: datetime
    group_name: str