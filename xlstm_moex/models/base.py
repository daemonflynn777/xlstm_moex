import abc


class BaseModel(abc.ABC):
    @abc.abstractmethod
    def __init__(self, **kwargs):
        raise NotImplementedError
    
    # @abc.abstractmethod
    # def fit(self, **kwargs):
    #     raise NotImplementedError
    
    # @abc.abstractmethod
    # def forecast_sequence(self, **kwargs):
    #     raise NotImplementedError