from abc import ABC, abstractmethod

class Ensemble(ABC):
    @abstractmethod
    #return dictionay of the predictions
    def forward(self, batch, train=False) -> dict:
        pass
    
    


    