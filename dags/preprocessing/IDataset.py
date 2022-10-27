from abc import ABC

class IDataset(ABC):
    def __init__(self) -> None:
        super().__init__()

    def return_data(self):
        pass

    def return_labels(self):
        pass 
