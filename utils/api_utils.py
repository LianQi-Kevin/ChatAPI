from typing import Literal, Any

from pydantic import BaseModel, model_validator, conint


class RESTfulAPI(BaseModel):
    """将数据按照RESTful格式封装"""
    code: conint(gt=100, lt=600)
    status: Literal["success", "fail", "error"]
    message: str = None
    data: Any = None

    @model_validator(mode='after')
    def validate_message(cls, m: 'RESTfulAPI'):
        """当status值不为success时, 要求提供原因"""
        if m.status != "success" and m.message is None:
            raise ValueError("message is required when status is not success")
